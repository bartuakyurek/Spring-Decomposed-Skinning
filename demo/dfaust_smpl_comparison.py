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

from src.global_vars import subject_ids, pose_ids, RESULT_PATH
from src.data.skeleton_data import get_smpl_skeleton
from src.render.pyvista_render_tools import add_skeleton, add_mesh
from src.data.smpl_sequence import get_gendered_smpl_model, get_anim_sequence

# -----------------------------------------------------------------------------
# Load animation sequence for the selected subject and pose
# -----------------------------------------------------------------------------

SELECTED_SUBJECT, SELECTED_POSE = subject_ids[0], pose_ids[0]

smpl_model = get_gendered_smpl_model(subject_id=SELECTED_SUBJECT, device="cpu")
F, kintree = smpl_model.faces, get_smpl_skeleton()
V_gt, V_smpl, J, _ = get_anim_sequence(SELECTED_SUBJECT, SELECTED_POSE, smpl_model, return_numpy=True)


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
rigid_skel_mesh = add_skeleton(plotter, J[0], kintree)
rigid_smpl_mesh = add_mesh(plotter, initial_smpl_V, F, opacity=0.8)
plotter.add_text("SMPL Rigid Deformation", font_size=18)
plotter.camera_position = [[-0.5,  1.5,  5.5],
                           [-0. ,  0.2,  0.3],
                           [ 0. ,  1. , -0.2]]

# -----------------------------------------------------------------------------
# Setup a helper rig and simulate 
# -----------------------------------------------------------------------------

# TODO: Precomputation here...
# Note that we could do the computation inside the render loop
# TODO: Report timing
J_dyn = J.copy()
V_dyn = V_smpl.copy()

# Add SMPL mesh to be jiggled
plotter.subplot(0, 2)
dyn_skel_mesh = add_skeleton(plotter, J_dyn[0], kintree)
dyn_smpl_mesh = add_mesh(plotter, initial_smpl_V, F, opacity=0.8)
plotter.add_text("Spring Deformation", font_size=18)
plotter.camera_position = [[-0.5,  1.5,  5.5],
                           [-0. ,  0.2,  0.3],
                           [ 0. ,  1. , -0.2]]

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