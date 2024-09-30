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
from optimal_rigid_motion import get_optimal_rigid_motion
from cost import MSE_np



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

## TODO: Add springs



## TODO: Simulate springs w.r.t. rigid motion (and save simulation)



## TODO: Render the baked simulation using Pyvista
## ----------------------------------------------------------------------------
# Create a plotter object and set the scalars to the Z height
plotter = pv.Plotter(notebook=False, off_screen=False)
plotter.camera_position = 'zy'
plotter.camera.azimuth = -90

#add_mesh(plotter, verts=V[0].copy(), faces=F)
add_skeleton(plotter, J[0].copy(), kintree)

# Open a gif
plotter.open_gif("./results/sample.gif")

n_frames = 200 
n_repeats = 5
for _ in range(n_repeats):
    for frame in range(n_frames-1):
        
        pts = J[frame].copy()
        plotter.update_coordinates(pts, render=False)
        #plotter.update_scalars(z.ravel(), render=False) # updates colors
    
        # Write a frame. This triggers a render.
        plotter.write_frame()

# Closes and finalizes movie
plotter.close()
