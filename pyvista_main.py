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

from spring import Spring
from pyvista_render_tools import add_skeleton, add_mesh


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
    
    ## TODO: Add springs

    spring_coords = np.array([
                                [0.5, 1.0, 0.5],
                                ])

    # This is the same as adding another mass and connecting it to the first
    # mass with a spring vector. 
    # TODO: maybe we can change the implementation
    # for a more intuitive mass-spring lattice creation
    spring_rest_vectors = np.array([
                                    [1.0, 0.0, 0.0],
                                    ])

    # TODO: could we create a shallow copy array, to combine SMPL joints plus spring coordinates? 
    spring_instances = []
    spring_coords_bake = []

    # Loop create springs 
    n_springs = 1
    for i in range(n_springs):
       pass
    ## TODO: Simulate springs w.r.t. rigid motion (and save simulation)
    
    for frame in range(n_frames-1):
        
        pts = J[frame].copy()
        plotter.update_coordinates(pts, render=False)
        #plotter.update_scalars(z.ravel(), render=False) # updates colors
    
        # Write a frame. This triggers a render.
        plotter.write_frame()

# Closes and finalizes movie
plotter.close()
