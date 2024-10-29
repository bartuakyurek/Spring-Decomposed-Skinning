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

from src.global_vars import subject_ids, pose_ids
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
# Create a plotter object
# -----------------------------------------------------------------------------
RENDER = True
plotter = pv.Plotter(notebook=False, off_screen=not RENDER, shape=(1,2))

initial_J, initial_smpl_V, initial_gt_V = J[0], V_smpl[0], V_gt[0]


# Add DFAUST Ground Truth Mesh
plotter.subplot(0, 0)
dfaust_mesh = add_mesh(plotter, initial_gt_V, F, opacity=1.0)
plotter.add_text("No Eye-Dome Lighting", font_size=24)
plotter.camera_position = [[-0.5,  1.5,  5.5],
                           [-0. ,  0.2,  0.3],
                           [ 0. ,  1. , -0.2]]

# Add SMPL Mesh 
plotter.subplot(0, 1)
skel_mesh = add_skeleton(plotter, J[0], kintree)
smpl_mesh = add_mesh(plotter, initial_smpl_V, F, opacity=1.0)
plotter.add_text("Eye-Dome Lighting", font_size=24)
plotter.camera_position = [[-0.5,  1.5,  5.5],
                           [-0. ,  0.2,  0.3],
                           [ 0. ,  1. , -0.2]]


# Show plot
def foo(value):
    print("===")
    print(np.round(plotter.camera_position, 1))
    print("---")
plotter.add_slider_widget(foo, [5, 100], title='Resolution')
plotter.show()