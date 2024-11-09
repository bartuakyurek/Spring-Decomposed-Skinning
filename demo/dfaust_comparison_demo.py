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
import pyvista as pv

import __init__
from src import skinning
from src.kinematics import inverse_kinematics
from src.helper_handler import HelperBonesHandler
from src.utils.linalg_utils import normalize_arr_np
from src.data.skeleton_data import get_smpl_skeleton
from src.skeleton import create_skeleton, add_helper_bones, extract_headtail_locations
from src.global_vars import subject_ids, pose_ids, RESULT_PATH
from src.data.smpl_sequence import get_gendered_smpl_model, get_anim_sequence, get_smpl_rest_data
from src.render.pyvista_render_tools import (add_mesh, 
                                             add_skeleton, 
                                             set_mesh_color_scalars,
                                             set_mesh_color)

# -----------------------------------------------------------------------------
# Config Parameters 
# -----------------------------------------------------------------------------
FIXED_SCALE = False # Set true if you want the jiggle bone to preserve its length
POINT_SPRING = False # Set true for less jiggling (point spring at the tip), set False to jiggle the whole bone as a spring.
EXCLUDE_ROOT = True # Set true in order not to render the invisible root bone (it's attached to origin)
DEGREES = False # Set true if pose is represented with degrees as Euler angles. 
                # WARNING: For SMPL it is False, i.e. radians.

FRAME_RATE = 24 #24
TIME_STEP = 1./FRAME_RATE  
MASS = 2.
STIFFNESS = 10 #200.
DAMPING = 15. #50.            
MASS_DSCALE = 0.5       # Scales mass velocity (Use [0.0, 1.0] range to slow down)
SPRING_DSCALE = 1.0     # Scales spring forces (increase for more jiggling)

ERR_MODE = "SMPL" # "DFAUST" or "SMPL", determines which mesh to take as reference for error distances
COLOR_CODE = True
RENDER_MESH_RIGID, RENDER_MESH_DYN = True, True # Turn on/off mesh for SMPL and/or SDDS
RENDER_SKEL_RIGID, RENDER_SKEL_DYN = True, True # Turn on/off mesh for SMPL and/or SDDS
OPACITY = 0.8
JIGGLE_SCALE = 1.0      # Set it greater than 1 to exaggerate the jiggling impact
NORMALIZE_WEIGHTS = False # Set true to automatically normalize the weights. Unnormalized weights might cause artifacts.
WINDOW_SIZE = (16*50*3, 16*80) # Divisible by 16 for ffmeg writer
ADD_GLOBAL_T = False    # Add the global translation given in the dataset 
                        # (Note that it'll naturally jiggle the helper bones but it doesn't mean 
                        #  the jiggling of helper bones are intiated with rigid movement 
                        #  so it might be misleading, could be better to keep it False.)
     
if JIGGLE_SCALE != 1.0:
    print(f"WARNING: Jiggle scaling is set to {JIGGLE_SCALE}, use 1.0 for normal settings.")
# -----------------------------------------------------------------------------
# Load animation sequence for the selected subject and pose
# -----------------------------------------------------------------------------

SELECTED_SUBJECT, SELECTED_POSE = subject_ids[0], pose_ids[10]

smpl_model = get_gendered_smpl_model(subject_id=SELECTED_SUBJECT, device="cpu")
F = np.array(smpl_model.faces, dtype=int)
smpl_kintree = get_smpl_skeleton()
V_gt, V_smpl, J, bpt = get_anim_sequence(SELECTED_SUBJECT, SELECTED_POSE, smpl_model, return_numpy=True)
V_rest, J_rest = get_smpl_rest_data(smpl_model, bpt[0][0], bpt[-1][0], return_numpy=True)


# -----------------------------------------------------------------------------
# Setup a helper rig
# -----------------------------------------------------------------------------

# Setup helper bones data
HELPER_RIG_PATH = "../data/helper_rig_data_50004.npz"
HELPER_RIG_SUBJECT = HELPER_RIG_PATH.split("_")[-1].split(".")[0]
if SELECTED_SUBJECT != HELPER_RIG_SUBJECT: print(">> WARNING: Selected subject does not match with imported rig target.")

# Load joint locations, kintree and weights
with np.load(HELPER_RIG_PATH) as data:
     helper_W = data["weights"]
     helper_joints = data["joints"]
     helper_kintree = data["kintree"]

# Configure helper rig kintree
n_verts = V_smpl.shape[1]
n_rigid_bones = J.shape[1]
n_helper_bones = helper_joints.shape[0]
HELPER_ROOT_PARENT = 6

assert HELPER_ROOT_PARENT < n_rigid_bones, f"Please select a valid parent index from: [0,..,{n_rigid_bones-1}]."
assert helper_kintree[0,0] == -1, "Expected first entry to be the root bone."
assert helper_kintree.shape[1] == 2, f"Expected helper kintree to have shape (n_helpers, 2), got {helper_kintree.shape}."
assert helper_joints.shape == (n_helper_bones,2,3), f"Expected helper joints to have shape (n_helpers, 2, 3), got {helper_joints.shape}."
assert helper_W.shape == (n_verts, n_helper_bones), f"Expected helper bone weights to have shape ({n_verts},{n_helper_bones}), got {helper_W.shape}."
helper_kintree = helper_kintree + n_rigid_bones # Update helper indices to match with current skeleton
helper_kintree[0,0] = HELPER_ROOT_PARENT            # Bind the helper tree to a bone in rigid skeleton

helper_parents = helper_kintree[:,0]                  # Extract parents (TODO: could you require less steps to setup helpers please?)
helper_endpoints = helper_joints[:,-1,:]            # TODO: we should be able to insert helper bones with head,tail data
                                                    # We can do that by start_points but also we should be able to provide [n_bones,2,3] shape, that is treated as headtail automatically.

initial_skel_J = J[0] #J_rest
helper_rig_t = initial_skel_J[HELPER_ROOT_PARENT] - helper_endpoints[0] * 0.5
#helper_endpoints += helper_rig_t # Translate the helper rig

assert len(helper_parents) == n_helper_bones, f"Expected all helper bones to have parents. Got {len(helper_parents)} parents instead of {n_helper_bones}."

# Initiate rigid rig and add helper bones on it

skeleton = create_skeleton(initial_skel_J, smpl_kintree)
helper_idxs = add_helper_bones(skeleton, 
                               helper_endpoints, 
                               helper_parents,
                               )
helper_idxs = np.array(helper_idxs)

helper_rig = HelperBonesHandler(skeleton, 
                                helper_idxs,
                                mass          = MASS, 
                                stiffness     = STIFFNESS,
                                damping       = DAMPING,
                                mass_dscale   = MASS_DSCALE,
                                spring_dscale = SPRING_DSCALE,
                                dt            = TIME_STEP,
                                point_spring  = POINT_SPRING,
                                fixed_scale   = FIXED_SCALE) 

# -----------------------------------------------------------------------------
# Simulate data
# -----------------------------------------------------------------------------

# Loop over frames:
J_dyn = []
V_dyn = V_smpl.copy() 
n_frames = V_smpl.shape[0]

_, poses, translations = bpt
helper_poses = np.zeros((n_helper_bones, 3))                  # (10,3)
rest_bone_locations = skeleton.get_rest_bone_locations(exclude_root=False)
prev_J = rest_bone_locations
prev_V = V_smpl[0]
for frame in range(n_frames):
    # WARNING:I'm using pose parameters in the dataset to apply FK for helper bones
    # but when the bones are actually posed with these parameters, the skeleton 
    # is not the same as the provided joint locations from the SMPL model. That is
    # because SMPL use a regressor to estimate the bone locations.
    theta = np.reshape(poses[frame].numpy(),newshape=(-1, 3)) # (24,3)
    theta = np.vstack((theta, helper_poses))                  # (34,3)
    rigidly_posed_locations = skeleton.pose_bones(theta, degrees=DEGREES) 
    
    global_trans = translations[frame].numpy()            # (3,)
    if ADD_GLOBAL_T: rigidly_posed_locations += global_trans
    
    # Since the regressed joint locations of SMPL is different, keep them as given in the dataset
    # (Try to comment the couple lines right below this to see the effect.)
    smpl_J_frame = extract_headtail_locations(J[frame], smpl_kintree, exclude_root=False)
    rigidly_posed_locations[:len(smpl_J_frame)] = smpl_J_frame 
    
    # 1.1 - Compute dynamic joint locations via simulation
    dyn_posed_locations = helper_rig.update_bones(rigidly_posed_locations) # Update the rigidly posed locations
    J_dyn.append(dyn_posed_locations)
    
    # 1.2 - Get the transformations through IK (cancelled for SMPL)
    #M = inverse_kinematics.get_absolute_transformations(rest_bone_locations, dyn_posed_locations, return_mat=True, algorithm="RST")
    #M = inverse_kinematics.get_absolute_transformations(prev_J, dyn_posed_locations, return_mat=True, algorithm="RST")
    #M = M[helper_idxs] # TODO: you may need to change it after excluding root bone? make sure you're retrieving correct transformations
    
    # 1.3 - Feed them to skinning and obtain dynamically deformed vertices. (cancelled for SMPL)
    #mesh_points = skinning.LBS_from_mat(prev_V, helper_W, M, use_normalized_weights=NORMALIZE_WEIGHTS) 
    
    # Compute the translations and add them on SMPL mesh
    prev_helper_tips = prev_J[2 * helper_idxs + 1]
    cur_helper_tips = dyn_posed_locations[2 * helper_idxs + 1]
    
    delta = cur_helper_tips - prev_helper_tips
    delta_jiggle = helper_W @ delta 
    V_dyn[frame] += delta_jiggle * JIGGLE_SCALE
    
    prev_J = dyn_posed_locations
    prev_V = V_smpl[frame]

J_dyn = np.array(J_dyn, dtype=float)[:,2:,:] # TODO: get rid of the root bone
# TODO: Report simulation timing

# -----------------------------------------------------------------------------
# Create a plotter object and add meshes
# -----------------------------------------------------------------------------
RENDER = True
plotter = pv.Plotter(notebook = False, 
                     off_screen = not RENDER, 
                     shape = (1,3),
                     border = False,
                     window_size = WINDOW_SIZE)

# Data to initialize mesh objects
initial_J, initial_smpl_V, initial_gt_V = J[0], V_smpl[0], V_gt[0]


TEXT_POSITION = "lower_left"
# Add DFAUST Ground Truth Mesh
plotter.subplot(0, 0)
dfaust_mesh = add_mesh(plotter, initial_gt_V, F, opacity=1.0)
plotter.add_text("Ground Truth Deformation", TEXT_POSITION, font_size=18)
plotter.camera_position = [[-0.5,  1.5,  5.5],
                           [-0. ,  0.2,  0.3],
                           [ 0. ,  1. , -0.2]]

frame_text_actor = plotter.add_text("0", (600,0), font_size=18)

# Add SMPL Mesh 
plotter.subplot(0, 1)
rigid_skel_mesh, rigid_skel_actor = add_skeleton(plotter, initial_J, smpl_kintree, return_actor=True)
rigid_smpl_mesh, rigid_smpl_actor = add_mesh(plotter, initial_smpl_V, F, opacity=OPACITY, return_actor=True)
plotter.add_text("SMPL Rigid Deformation", TEXT_POSITION, font_size=18)
plotter.camera_position = [[-0.5,  1.5,  5.5],
                           [-0. ,  0.2,  0.3],
                           [ 0. ,  1. , -0.2]]

# Add SMPL mesh to be jiggled
plotter.subplot(0, 2)

assert J_dyn.shape[0] == J.shape[0], f"Expected first dimensions to share number of frames (n_frames, n_bones, 3). Got shapes {J_dyn.shape} and {J.shape}."
J_dyn_initial = J_dyn[0]
edges = len(skeleton.rest_bones)
line_segments = np.reshape(np.arange(0, 2*(edges-1)), (edges-1, 2))
dyn_skel_mesh, dyn_skel_actor = add_skeleton(plotter, J_dyn_initial, line_segments, return_actor=True)

dyn_smpl_mesh, dyn_smpl_actor = add_mesh(plotter, initial_smpl_V, F, opacity=OPACITY, return_actor=True)
plotter.add_text("Spring Deformation", TEXT_POSITION, font_size=18)
plotter.camera_position = [[-0.5,  1.5,  5.5],
                           [-0. ,  0.2,  0.3],
                           [ 0. ,  1. , -0.2]]

# Visibility settings
if not RENDER_MESH_DYN: dyn_smpl_actor.visibility = False
if not RENDER_SKEL_DYN: dyn_skel_actor.visibility = False

if not RENDER_MESH_RIGID: rigid_smpl_actor.visibility = False
if not RENDER_SKEL_RIGID: rigid_skel_actor.visibility = False
# -----------------------------------------------------------------------------
# Render and save results
# -----------------------------------------------------------------------------
    
result_fname = "dfaust_comparison" + "_" + str(SELECTED_SUBJECT) + "_" + str(SELECTED_POSE)
plotter.open_movie(RESULT_PATH + f"{result_fname}.mp4")

n_frames = len(V_smpl)
if ERR_MODE == "SMPL":
    base_verts = V_smpl
    distance_err_dyn = np.linalg.norm(base_verts - V_dyn, axis=-1)  # (n_frames, n_verts)
    tot_err_dyn =  np.sum(distance_err_dyn)
    print(">> Total error: ", np.round(tot_err_dyn,4))
    avg_err_dyn = tot_err_dyn / n_frames
    print(">> Average error: ", np.round(avg_err_dyn, 4))
    normalized_dists_dyn = normalize_arr_np(distance_err_dyn) 
    
elif ERR_MODE == "DFAUST":
    base_verts = V_gt
    distance_err_rigid = np.linalg.norm(base_verts - V_smpl, axis=-1)  # (n_frames, n_verts)
    distance_err_dyn = np.linalg.norm(base_verts - V_dyn, axis=-1)  # (n_frames, n_verts)
    
    tot_err_rigid =  np.sum(distance_err_rigid)
    tot_err_dyn =  np.sum(distance_err_dyn)
    print(">> Total error SMPL: ", np.round(tot_err_rigid,4))
    print(">> Total error Ours: ", np.round(tot_err_dyn,4))
    avg_err_rigid = tot_err_rigid / n_frames
    avg_err_dyn = tot_err_dyn / n_frames
    print(">> Average error SMPL: ", np.round(avg_err_rigid, 4))
    print(">> Average error Ours: ", np.round(avg_err_dyn, 4))
    normalized_dists_rigid = normalize_arr_np(distance_err_rigid) 
    normalized_dists_dyn = normalize_arr_np(distance_err_dyn) 

for frame in range(n_frames):
    rigid_skel_mesh.points = J[frame]   # Update mesh points in the renderer.
    dyn_skel_mesh.points = J_dyn[frame] # TODO: update it!
    
    dfaust_mesh.points = V_gt[frame]
    rigid_smpl_mesh.points = V_smpl[frame]
    dyn_smpl_mesh.points = V_dyn[frame]
     
    # Colorize meshes with respect to error distances
    if COLOR_CODE:
        if ERR_MODE == "DFAUST":
            set_mesh_color_scalars(rigid_smpl_mesh, normalized_dists_rigid[frame])  
        set_mesh_color_scalars(dyn_smpl_mesh, normalized_dists_dyn[frame])  
    
    frame_text_actor.input = str(frame+1)
    plotter.write_frame()               # Write a frame. This triggers a render.

plotter.close()


