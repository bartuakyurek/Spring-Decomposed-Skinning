#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:51:49 2024

@author: bartu
"""

import open3d as o3d

import os
import numpy as np
import torch

from smpl_torch_batch import SMPLModel
from skeleton_data import get_smpl_skeleton

training_data = torch.load('./data/50004_dataset.pt')
data_loader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=False)

device = "cpu"
smpl_model = SMPLModel(device=device, model_path='./body_models/smpl/female/model.pkl')
kintree = get_smpl_skeleton()
F = smpl_model.faces

for data in data_loader:
   beta_pose_trans_seq = data[0].squeeze().type(torch.float64)
   betas, pose, trans = beta_pose_trans_seq[:,:10], beta_pose_trans_seq[:,10:82], beta_pose_trans_seq[:,82:] 
   target_verts = data[1].squeeze()
   smpl_verts, joints = smpl_model(betas, pose, trans)
   break
   
V = smpl_verts.detach().cpu().numpy()
J = joints.detach().cpu().numpy()
n_frames, n_verts, n_dims = target_verts.shape

# ---------------------------------------------------------------------------- 
import numpy as np
import pyvista as pv

from pyvista_render_tools import add_skeleton, add_mesh


## TODO: Run spring simulation here and save the mass locations for rendering later




## TODO: Render the baked simulation using Pyvista
## ----------------------------------------------------------------------------
# Create a plotter object and set the scalars to the Z height

RENDER = True
plotter = pv.Plotter(notebook=False, off_screen=not RENDER)
plotter.camera_position = 'zy'
plotter.camera.azimuth = -90

skel_mesh = add_skeleton(plotter, J[0], kintree)

# Open a gif
plotter.open_movie("./results/smpl-skeleton.mp4")

n_frames = 200 
n_repeats = 10
for _ in range(n_repeats):
    for frame in range(n_frames-1):
        # TODO: Reset particle system
        
        # TODO: Update particle system
        
        # TODO: Update mesh points
        skel_mesh.points = J[frame]
        
        # Write a frame. This triggers a render.
        plotter.write_frame()

# Closes and finalizes movie
plotter.close()
plotter.deep_clean()
