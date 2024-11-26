#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file is created to visualize a comparison between plain SMPL,
our helper bones with SMPL, and Wu et al.'s method on SMPL (with the
same helper handles but simulated with their method).

Created on Tue Nov 26 10:34:45 2024

@author: bartu
"""

import os
import numpy as np
import pyvista as pv
from matplotlib import cm

from src.global_vars import subject_ids, pose_ids, RESULT_PATH, DATA_PATH
from src.render.pyvista_render_tools import (add_mesh, 
                                             add_skeleton,
                                             add_skeleton_from_Skeleton, 
                                             set_mesh_color_scalars,
                                             )
# =============================================================================
# # --------------------------------------------------------------------------
# # Render parameters
# # --------------------------------------------------------------------------
# =============================================================================
err_cmap = cm.jet #winter, jet, brg, gnuplot2, autumn, viridis or see https://matplotlib.org/stable/users/explain/colors/colormaps.html
COLOR_CODE = True
RENDER_MESH_RIGID, RENDER_MESH_DYN = True, True # Turn on/off mesh for SMPL and/or SDDS
RENDER_SKEL_RIGID, RENDER_SKEL_DYN = True, True # Turn on/off mesh for SMPL and/or SDDS
OPACITY = 0.3
WINDOW_SIZE = (16*50*2, 16*80) # Divisible by 16 for ffmeg writer


SELECTED_SUBJECT, SELECTED_POSE = subject_ids[0], pose_ids[8]

# =============================================================================
# # --------------------------------------------------------------------------
# # Load related data (Ours)
# # --------------------------------------------------------------------------
# =============================================================================
import simulate_smpl_ours

# Get our data
smpl_bundle, ours_bundle = simulate_smpl_ours.get_simulated_smpl(True, 
                                                                 SELECTED_SUBJECT, 
                                                                 SELECTED_POSE)

F, J, smpl_kintree, _, V_smpl, W_smpl = smpl_bundle
J_dyn, skeleton, helper_idxs, V_dyn, helper_W = ours_bundle

if COLOR_CODE: err_codes_dyn = simulate_smpl_ours.get_smpl_distance_error(V_smpl, V_dyn)


# create model path if not existing
# base_pose = ... save it as .obj A-pose or T-pose (V[0])
# .tgf for skeleton data at base_pose (J[0])
# weights

# Prepare data for Wu et al. simulation
# -------------------------------------------
_, rigidly_simulated_bundle = simulate_smpl_ours.get_simulated_smpl(False, 
                                                                 SELECTED_SUBJECT, 
                                                                 SELECTED_POSE)
J_helpers_rigid, _, _, _, _ = rigidly_simulated_bundle

n_frames = len(V_smpl)
point_handles_anim = np.reshape(J_helpers_rigid,(n_frames,-1,2,3))[:,:,1,:] # (n_frames, n_handles, 3)
assert point_handles_anim.shape[0] == n_frames

# Check if the data directory exists, else create it
modelname = "smpl_" + str(SELECTED_SUBJECT) + "_" + str(SELECTED_POSE)
asset_path = os.path.join(DATA_PATH, modelname)
if not os.path.exists(asset_path):
    os.makedirs(asset_path)

# Write weights to a file
# WARNING: Assumes helper indices are all at the end
W_wu = np.append(W_smpl, helper_W, axis=-1)
np.savetxt(os.path.join(asset_path, f'{modelname}_w.txt'), W_wu)

# Write .tgf in its most simplistic form
J_rest = point_handles_anim[0]

with open(os.path.join(asset_path,f"{modelname}.tgf"), "w") as f:
    for i,point_coord in enumerate(J_rest):
        x,y,z = point_coord
        f.write(str(i) + " " + str(x) + " " + str(y) + " " + str(z) + "\n")
        
    f.write('#')
    
# Write obj
import igl
V_rest = V_smpl[0]
igl.write_obj(os.path.join(asset_path,f"{modelname}.obj"), V_rest, F)

# =============================================================================
# # --------------------------------------------------------------------------
# # Load related data (Wu et al.)
# # --------------------------------------------------------------------------
# =============================================================================
import simulate_smpl_wu

V_wu, J_wu, fixed_handles = simulate_smpl_wu.get_simulated_smpl(modelname, 
                                                                asset_path,
                                                                point_handles_anim,
                                                                start_frame=0,
                                                                tetmesh=False, save_path= None)

# =============================================================================
# # ---------------------------------------------------------------------------
# # Create a plotter and add subplots
# # ---------------------------------------------------------------------------
# =============================================================================
RENDER = True
plotter = pv.Plotter(notebook = False, 
                     off_screen = not RENDER, 
                     shape = (1,2),
                     border = False,
                     window_size = WINDOW_SIZE)

# Data to initialize mesh objects
initial_J, initial_smpl_V = J[0], V_smpl[0]

# =============================================================================
# Add SMPL 
# =============================================================================
plotter.subplot(0, 0)
rigid_skel_mesh, rigid_skel_actor = add_skeleton(plotter, initial_J, smpl_kintree, bone_color="white", return_actor=True)

rigid_smpl_mesh, rigid_smpl_actor = add_mesh(plotter, initial_smpl_V, F, opacity=OPACITY, return_actor=True)
plotter.camera_position = [[-0.5,  1.5,  5.5],
                           [-0. ,  0.2,  0.3],
                           [ 0. ,  1. , -0.2]]

# =============================================================================
# Add Ours
# =============================================================================
plotter.subplot(0, 1)

assert J_dyn.shape[0] == J.shape[0], f"Expected first dimensions to share number of frames (n_frames, n_bones, 3). Got shapes {J_dyn.shape} and {J.shape}."
#J_dyn_initial = J_dyn[0]
#edges = len(skeleton.rest_bones)
#line_segments = np.reshape(np.arange(0, 2*(edges-1)), (edges-1, 2))
dyn_skel_mesh, dyn_skel_actor = add_skeleton_from_Skeleton(plotter, skeleton, helper_idxs, is_smpl=True, return_actor=True)

dyn_smpl_mesh, dyn_smpl_actor = add_mesh(plotter, initial_smpl_V, F, opacity=OPACITY, return_actor=True)
plotter.camera_position = [[-0.5,  1.5,  5.5],
                           [-0. ,  0.2,  0.3],
                           [ 0. ,  1. , -0.2]]


# =============================================================================
# Visibility settings
# =============================================================================
if not RENDER_MESH_DYN: dyn_smpl_actor.visibility = False
if not RENDER_SKEL_DYN: 
    dyn_skel_actor[0].visibility = False
    dyn_skel_actor[1].visibility = False

if not RENDER_MESH_RIGID: rigid_smpl_actor.visibility = False
if not RENDER_SKEL_RIGID: 
    rigid_skel_actor[0].visibility = False
    rigid_skel_actor[1].visibility = False

# =============================================================================
# Visualize results    
# =============================================================================
result_fname = "dfaust_comparison" + "_" + str(SELECTED_SUBJECT) + "_" + str(SELECTED_POSE)
plotter.open_movie(RESULT_PATH + f"{result_fname}.mp4")

for frame in range(n_frames):
    rigid_skel_mesh.points = J[frame]   # Update mesh points in the renderer.
    dyn_skel_mesh.points = J_dyn[frame] # TODO: update it!
    
    rigid_smpl_mesh.points = V_smpl[frame]
    dyn_smpl_mesh.points = V_dyn[frame]
     
    # Colorize meshes with respect to error distances
    if COLOR_CODE:
        set_mesh_color_scalars(dyn_smpl_mesh, err_codes_dyn[frame], err_cmap)  
    
    #frame_text_actor.input = str(frame+1)
    plotter.write_frame()               # Write a frame. This triggers a render.

plotter.close()
