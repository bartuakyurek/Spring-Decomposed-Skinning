#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:34:45 2024

@author: bartu
"""

import numpy as np
import pyvista as pv
from matplotlib import cm

from src.global_vars import subject_ids, pose_ids, RESULT_PATH
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

SIMULATE_HELPERS = False
SELECTED_SUBJECT, SELECTED_POSE = subject_ids[0], pose_ids[8]

# =============================================================================
# # --------------------------------------------------------------------------
# # Load related data
# # --------------------------------------------------------------------------
# =============================================================================
from simulate_smpl import get_simulated_smpl, get_smpl_distance_error

F, J, J_dyn, smpl_kintree, skeleton, helper_idxs, V_gt, V_smpl, V_dyn = get_simulated_smpl(SIMULATE_HELPERS, 
                                                                                    SELECTED_SUBJECT, 
                                                                                    SELECTED_POSE)
err_codes_dyn = get_smpl_distance_error(V_smpl, V_dyn)

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
initial_J, initial_smpl_V, initial_gt_V = J[0], V_smpl[0], V_gt[0]


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

n_frames = len(V_smpl)
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
