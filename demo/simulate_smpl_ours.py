#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:59:21 2024

This file is created to view DFAUST and corresponding rigid SMPL model 
side by side. We then add helper bones to jiggle the SMPL tissues to 
compare the computed jigglings with the ground truth DFAUST data.

Color coded error metrics display Euclidean distance differences between
DFAUST vs. SMPL (middle) and DFAUST vs. Ours (right). Note that since the
basic SMPL model has more topological difference on hands and feet, the 
distance error isn't on the jiggling tissues. 

@author: bartu
"""

import numpy as np

import __init__
from src import skinning
from src.kinematics import inverse_kinematics
from src.helper_handler import HelperBonesHandler
from src.utils.linalg_utils import normalize_arr_np
from src.data.skeleton_data import get_smpl_skeleton
from src.skeleton import create_skeleton, add_helper_bones, extract_headtail_locations
from src.data.smpl_sequence import get_gendered_smpl_model, get_anim_sequence, get_smpl_rest_data


# =============================================================================
# # --------------------------------------------------------------------------
# # Config Parameters 
# # --------------------------------------------------------------------------
# =============================================================================

# Simulation parameters
FRAME_RATE = 24 #24
TIME_STEP = 1./FRAME_RATE  
MASS = 1.
STIFFNESS = 100 #200.
DAMPING = 10 #50.            
MASS_DSCALE = 0.2        # Scales mass velocity (Use [0.0, 1.0] range to slow down)
SPRING_DSCALE = 3.0     # Scales spring forces (increase for more jiggling)

COMPLIANCE = 0.0 # Set positive number for soft constraint, 0 for hard constraints
EDGE_CONSTRAINT = False # Only available for PBD integration
FIXED_SCALE = True # Set true if you want the jiggle bone to preserve its length
POINT_SPRING = False # Set true for less jiggling (point spring at the tip), set False to jiggle the whole bone as a spring.
EXCLUDE_ROOT = True # Set true in order not to render the invisible root bone (it's attached to origin)
DEGREES = False # Set true if pose is represented with degrees as Euler angles. 
                # WARNING: For SMPL it is False, i.e. radians.

INTEGRATION = "PBD" # "PBD" or "Euler" 
ALGO = "T" # T or RST, if T is selected, only translations will be concerned. Note that RST fails at current stage. TODO: investigate it.
NORMALIZE_WEIGHTS = True # Set true to automatically normalize the weights. Unnormalized weights might cause artifacts.

JIGGLE_SCALE = 1.0      # Set it greater than 1 to exaggerate the jiggling impact
if JIGGLE_SCALE != 3.0: print(f"WARNING: Jiggle scaling is set to {JIGGLE_SCALE}, use 1.0 for normal settings.")


 # =============================================================================
 # Compute difference errors
 # =============================================================================
def get_smpl_distance_error(V_smpl, V_dyn):
     n_frames = len(V_smpl)
     base_verts = V_smpl
     distance_err_dyn = np.linalg.norm(base_verts - V_dyn, axis=-1)  # (n_frames, n_verts)
     
     tot_err_dyn =  np.sum(distance_err_dyn)
     print(">> Total error: ", np.round(tot_err_dyn,4))
     avg_err_dyn = tot_err_dyn / n_frames
     print(">> Average error: ", np.round(avg_err_dyn, 4))
     
     normalized_dists_dyn = normalize_arr_np(distance_err_dyn) 
     err_codes_dyn = normalized_dists_dyn
     return err_codes_dyn
 

def get_simulated_smpl(SIMULATE_HELPERS, SELECTED_SUBJECT, SELECTED_POSE):
    #SIMULATE_HELPERS = False # If set to false, helpers will not be simulated, to be used to feed in CPBD point handle movement

    # =============================================================================
    # # --------------------------------------------------------------------------
    # # Load animation sequence for the selected subject and pose
    # # --------------------------------------------------------------------------
    # =============================================================================
    
    smpl_model = get_gendered_smpl_model(subject_id=SELECTED_SUBJECT, device="cpu")
    F = np.array(smpl_model.faces, dtype=int)
    W_smpl = smpl_model.weights.numpy()
    smpl_kintree = get_smpl_skeleton()
    V_gt, V_smpl, J, bpt = get_anim_sequence(SELECTED_SUBJECT, SELECTED_POSE, smpl_model, return_numpy=True)
    V_rest, J_rest = get_smpl_rest_data(smpl_model, bpt[0][0], bpt[-1][0], return_numpy=True)
    
    # =============================================================================
    # # --------------------------------------------------------------------------
    # # Setup a helper rig
    # # --------------------------------------------------------------------------
    # =============================================================================
    HELPER_RIG_PATH = "../data/helper_rig_data_50004.npz"
    HELPER_RIG_SUBJECT = HELPER_RIG_PATH.split("_")[-1].split(".")[0]
    if SELECTED_SUBJECT != HELPER_RIG_SUBJECT: print(">> WARNING: Selected subject does not match with imported rig target.")
    
    # Load joint locations, kintree and weights
    with np.load(HELPER_RIG_PATH) as data:
         helper_W = data["weights"]
         helper_joints = data["joints"]
         helper_kintree_bl = data["kintree"]
    
    # Configure helper rig kintree
    n_verts = V_smpl.shape[1]
    n_rigid_bones = J.shape[1]
    n_helper_bones = helper_joints.shape[0]
    HELPER_ROOT_PARENT = 9 # See https://www.researchgate.net/figure/Layout-of-23-joints-in-the-SMPL-models_fig2_351179264
    
    assert HELPER_ROOT_PARENT < n_rigid_bones, f"Please select a valid parent index from: [0,..,{n_rigid_bones-1}]."
    assert helper_kintree_bl[0,0] == -1, "Expected first entry to be the root bone."
    assert helper_kintree_bl.shape[1] == 2, f"Expected helper kintree to have shape (n_helpers, 2), got {helper_kintree_bl.shape}."
    assert helper_joints.shape == (n_helper_bones,2,3), f"Expected helper joints to have shape (n_helpers, 2, 3), got {helper_joints.shape}."
    assert helper_W.shape == (n_verts, n_helper_bones), f"Expected helper bone weights to have shape ({n_verts},{n_helper_bones}), got {helper_W.shape}."
    
    helper_kintree = helper_kintree_bl + n_rigid_bones # Update helper indices to match with current skeleton
    helper_kintree[0,0] = HELPER_ROOT_PARENT            # Bind the helper tree to a bone in rigid skeleton
    
    helper_parents = helper_kintree[:,0]                  # Extract parents (TODO: could you require less steps to setup helpers please?)
    assert len(helper_parents) == n_helper_bones, f"Expected all helper bones to have parents. Got {len(helper_parents)} parents instead of {n_helper_bones}."
    helper_endpoints = helper_joints[:,-1,:]            # TODO: we should be able to insert helper bones with head,tail data
                                                        # We can do that by start_points but also we should be able to provide [n_bones,2,3] shape, that is treated as headtail automatically.
    
    # Initiate rigid rig and add helper bones on it
    initial_skel_J = J[0] #J_rest
    skeleton = create_skeleton(initial_skel_J, smpl_kintree)
    helper_idxs = add_helper_bones(skeleton, 
                                   helper_endpoints, 
                                   helper_parents,
                                   )
    
    helper_idxs = np.array(helper_idxs, dtype=int)
    if SIMULATE_HELPERS:
        helper_rig = HelperBonesHandler(skeleton, 
                                        helper_idxs,
                                        mass          = MASS, 
                                        stiffness     = STIFFNESS,
                                        damping       = DAMPING,
                                        mass_dscale   = MASS_DSCALE,
                                        spring_dscale = SPRING_DSCALE,
                                        point_spring  = POINT_SPRING,
                                        edge_constraint   = EDGE_CONSTRAINT,
                                        compliance    = COMPLIANCE,
                                        fixed_scale = FIXED_SCALE,
                                        simulation_mode = INTEGRATION) 
    
    # =============================================================================
    # # --------------------------------------------------------------------------
    # # Simulate data
    # # --------------------------------------------------------------------------
    # =============================================================================
    # Loop over frames:
    J_dyn = []
    V_dyn = V_smpl.copy() 
    n_frames = V_smpl.shape[0]
    
    _, poses, translations = bpt
    helper_poses = np.zeros((n_helper_bones, 3))                  # (10,3)
    rest_bone_locations = skeleton.get_rest_bone_locations(exclude_root=False)
    prev_J = rest_bone_locations
    prev_V = V_smpl[0]
    
    n_bones = len(skeleton.rest_bones)
    n_bones_rigid = n_bones - len(helper_idxs)
    for frame in range(n_frames):
        diff = J[frame] - J[0]
        theta = np.zeros((n_bones_rigid, 3))      # (24,3)
        theta = np.vstack((theta, helper_poses))  # (34,3)
        t = np.zeros((n_bones,3))
        for i in range(n_bones):
            if i in helper_idxs:
                continue
            t[i] = diff[i]
    
        rigidly_posed_locations = skeleton.pose_bones(theta, t, degrees=DEGREES) 
        smpl_J_frame = extract_headtail_locations(J[frame], smpl_kintree, exclude_root=False)
        rigidly_posed_locations[:len(smpl_J_frame)] = smpl_J_frame 
        
        # 1.1 - Compute dynamic joint locations via simulation
        if SIMULATE_HELPERS:
            dyn_posed_locations = helper_rig.update_bones(rigidly_posed_locations) # Update the rigidly posed locations
        else: dyn_posed_locations = rigidly_posed_locations
        
        J_dyn.append(dyn_posed_locations)
        
        if ALGO == "RST":
            # 1.2 - Get the transformations through IK (cancelled for SMPL)
            M = inverse_kinematics.get_absolute_transformations(prev_J, dyn_posed_locations, return_mat=True, algorithm="RST")
            M = M[helper_idxs] # TODO: you may need to change it after excluding root bone? make sure you're retrieving correct transformations
            
            # 1.3 - Feed them to skinning and obtain dynamically deformed vertices. (cancelled for SMPL)
            delta_jiggle = skinning.LBS_from_mat(prev_V, helper_W, M, use_normalized_weights=NORMALIZE_WEIGHTS) 
            
        else:
            # Compute the translations and add them on SMPL mesh
            prev_helper_tips = prev_J[2 * helper_idxs + 1]
            cur_helper_tips = dyn_posed_locations[2 * helper_idxs + 1]
            
            delta = cur_helper_tips - prev_helper_tips
            delta_jiggle = helper_W @ delta 
        
        V_dyn[frame] += delta_jiggle * JIGGLE_SCALE
        
        prev_J = dyn_posed_locations
        prev_V = V_smpl[frame]
    
    J_dyn = np.array(J_dyn, dtype=float)[:,2:,:] # TODO: get rid of the root bone
    
    smpl_bundle = (F, J, smpl_kintree, V_gt, V_smpl, W_smpl)
    ours_bundle = (J_dyn, skeleton, helper_idxs, V_dyn, helper_W, helper_kintree_bl)
    return smpl_bundle, ours_bundle
   
        

