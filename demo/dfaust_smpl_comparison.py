#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:59:21 2024

This file is created to view DFAUST and corresponding rigid SMPL model 
side by side. We then add helper bones to jiggle the SMPL tissues to 
compare the computed jigglings with the ground truth DFAUST data.

@author: bartu
"""

import numpy as np
import pyvista as pv

import __init__
from src.helper_handler import HelperBonesHandler
from src.data.skeleton_data import get_smpl_skeleton
from src.skeleton import create_skeleton, add_helper_bones
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
DEGREES = True # Set true if pose is represented with degrees as Euler angles.
N_REPEAT = 10
N_REST = N_REPEAT - 5
FRAME_RATE = 24 #24
TIME_STEP = 1./FRAME_RATE  
MASS = 1.
STIFFNESS = 300.
DAMPING = 50.            
MASS_DSCALE = 0.4       # Scales mass velocity (Use [0.0, 1.0] range to slow down)
SPRING_DSCALE = 1.0     # Scales spring forces (increase for more jiggling)

ADD_GLOBAL_T = False    # Add the global translation given in the dataset 
                        # (Note that it'll naturally jiggle the helper bones but it doesn't mean 
                        #  the jiggling of helper bones are intiated with rigid movement 
                        #  so it might be misleading, could be better to keep it False.)
# -----------------------------------------------------------------------------
# Load animation sequence for the selected subject and pose
# -----------------------------------------------------------------------------

SELECTED_SUBJECT, SELECTED_POSE = subject_ids[0], pose_ids[4]

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

helper_rig_t = J_rest[HELPER_ROOT_PARENT] - helper_endpoints[0] * 0.8
helper_endpoints += helper_rig_t # Translate the helper rig

assert len(helper_parents) == n_helper_bones, f"Expected all helper bones to have parents. Got {len(helper_parents)} parents instead of {n_helper_bones}."

# Initiate rigid rig and add helper bones on it
J_initial = J_rest #J[0] #J_rest
skeleton = create_skeleton(J_initial, smpl_kintree)
helper_idxs = add_helper_bones(skeleton, 
                               helper_endpoints, 
                               helper_parents,
                               )

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

def extract_headtail_locations(joints, kintree, exclude_root=False, collapse=True):
    # Given the joints and their connectivity
    # Extract two endpoints for each bone
    
    n_bones = len(kintree)
    if not exclude_root: n_bones += 1
    J = np.empty((n_bones, 2, 3))
    
    if not exclude_root: # WARNING: Assumes root node is at 0
        J[0] = np.array([[0,0,0], joints[0]])
        
    for pair in kintree:
        parent_idx, bone_idx = pair
        J[bone_idx] = np.array([joints[parent_idx], joints[bone_idx]]) 
        
    if collapse:
        J = np.reshape(J, (-1,3))
    return J

# Loop over frames:
J_dyn = []
V_dyn = V_smpl.copy() #[]
n_frames = V_smpl.shape[0]
all_J = skeleton.get_rest_bone_locations(exclude_root=False)
_, poses, translations = bpt
for frame in range(n_frames):
    
    theta = np.reshape(poses[frame].numpy(),newshape=(-1, 3)) # (24,3)
    helper_poses = np.zeros((n_helper_bones, 3))
    theta = np.vstack((theta, helper_poses))
    global_trans = translations[frame].numpy()                 # (3,)
    all_J = skeleton.pose_bones(theta, degrees=DEGREES, exclude_root=False) 
    
    if ADD_GLOBAL_T: all_J += global_trans
    
    # Since the regressed joint locations of SMPL is different, keep them as given in the dataset
    #smpl_J_frame = extract_headtail_locations(J[frame], smpl_kintree, exclude_root=False)
    #all_J[:len(smpl_J_frame)] = smpl_J_frame 
    
    # 1.1 - Compute dynamic joint locations via simulation
    posed_locations = helper_rig.update_bones(all_J) # Update the rigidly posed locations
    all_J = posed_locations.copy()
    J_dyn.append(all_J)
    
    # 1.2 - Get the transformations through IK
    # 1.3 - Feed them to skinning and obtain dynamically deformed vertices.

J_dyn = np.array(J_dyn, dtype=float)[:,2:,:] # TODO: get rid of the root bone
# TODO: add rest frames to see jiggling after motion? No because smpl data has no ground truth for that.
# TODO: Report simulation timing

# -----------------------------------------------------------------------------
# Create a plotter object and add meshes
# -----------------------------------------------------------------------------
RENDER = True
plotter = pv.Plotter(notebook = False, 
                     off_screen = not RENDER, 
                     shape = (1,3),
                     border = False,
                     window_size = (1504, 1408))

# Data to initialize mesh objects
initial_J, initial_smpl_V, initial_gt_V = J[0], V_smpl[0], V_gt[0]

# Add DFAUST Ground Truth Mesh
plotter.subplot(0, 0)
dfaust_mesh = add_mesh(plotter, initial_gt_V, F, opacity=1.0)
plotter.add_text("Ground Truth Deformation", font_size=18)
plotter.camera_position = [[-0.5,  1.5,  5.5],
                           [-0. ,  0.2,  0.3],
                           [ 0. ,  1. , -0.2]]

# Add SMPL Mesh 
plotter.subplot(0, 1)
rigid_skel_mesh = add_skeleton(plotter, initial_J, smpl_kintree)
rigid_smpl_mesh = add_mesh(plotter, initial_smpl_V, F, opacity=0.8)
plotter.add_text("SMPL Rigid Deformation", font_size=18)
plotter.camera_position = [[-0.5,  1.5,  5.5],
                           [-0. ,  0.2,  0.3],
                           [ 0. ,  1. , -0.2]]

# Add SMPL mesh to be jiggled
plotter.subplot(0, 2)

assert J_dyn.shape[0] == J.shape[0], f"Expected first dimensions to share number of frames (n_frames, n_bones, 3). Got shapes {J_dyn.shape} and {J.shape}."
J_dyn_initial = J_dyn[0]
edges = len(skeleton.rest_bones)
line_segments = np.reshape(np.arange(0, 2*(edges-1)), (edges-1, 2))
dyn_skel_mesh = add_skeleton(plotter, J_dyn_initial, line_segments)

dyn_smpl_mesh = add_mesh(plotter, initial_smpl_V, F, opacity=0.8)
plotter.add_text("Spring Deformation", font_size=18)
plotter.camera_position = [[-0.5,  1.5,  5.5],
                           [-0. ,  0.2,  0.3],
                           [ 0. ,  1. , -0.2]]

# ---------------------------------------------------------------------------------
# Set up Key Press Actions
# ---------------------------------------------------------------------------------
selected_bone_idx = -1
n_bones = n_helper_bones # n_rigid_bones + n_helper_bones 
bone_start_idx = 0 #n_rigid_bones - 1
weights = helper_W                     # TODO: extract interface?
mesh_to_be_colored = dyn_smpl_mesh     # TODO: extract interface?

def change_colors():
    global selected_bone_idx
    global n_bones
    
    if selected_bone_idx == -1:
        selected_bone_idx = bone_start_idx
        
    selected_bone_idx += 1
    if selected_bone_idx >= n_bones-1: # TODO: remove -1 when you get rid of root bone
        selected_bone_idx = -1
    
    if selected_bone_idx >= 0:
        print("INFO: Selected bone ", selected_bone_idx)
        print(">> Call set mesh colors...")
        selected_weights = weights[:,selected_bone_idx]
        set_mesh_color_scalars(mesh_to_be_colored, selected_weights)  

def deselect_bone():
    global selected_bone_idx
    selected_bone_idx = -1
    set_mesh_color(mesh_to_be_colored, [0.8, 0.8, 1.0])
    print(">> INFO: Bone deselected.")
    return

# When "B" key is pressed, show colors of the corresponding bone's weights
plotter.add_key_event("B", change_colors)
plotter.add_key_event("b", change_colors)
plotter.add_key_event("N", deselect_bone)
plotter.add_key_event("n", deselect_bone)

# -----------------------------------------------------------------------------
# Render and save results
# -----------------------------------------------------------------------------
    
result_fname = "dfaust_comparison"
plotter.open_movie(RESULT_PATH + f"{result_fname}.mp4")

n_frames = V_smpl.shape[0]
for frame in range(n_frames):
    rigid_skel_mesh.points = J[frame]   # Update mesh points in the renderer.
    dyn_skel_mesh.points = J_dyn[frame] # TODO: update it!
    
    dfaust_mesh.points = V_gt[frame]
    rigid_smpl_mesh.points = V_smpl[frame]
    dyn_smpl_mesh.points = V_dyn[frame]
     
    plotter.write_frame()               # Write a frame. This triggers a render.
    
plotter.close()