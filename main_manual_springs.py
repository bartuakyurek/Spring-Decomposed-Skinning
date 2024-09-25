#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:54:30 2024

@author: bartu

This file extracts SMPL poses from ground truth of DFAUST dataset,
to inspect the difference between scanned human registrations and 
SMPL model output.

DISCLAIMER: The base code is borrowed from https://github.com/haoranchen06/DBS
"""

import os
import numpy as np
import torch

from smpl_torch_batch import SMPLModel
from skeleton_data import get_smpl_skeleton
from optimal_rigid_motion import get_optimal_rigid_motion
from cost import MSE_np

from scene_node import *
from matplot_viewer import Matplot_Viewer

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
   
V = smpl_verts.detach().cpu().numpy()#[SELECTED_FRAME]
J = joints.detach().cpu().numpy()#[SELECTED_FRAME]
n_frames, n_verts, n_dims = target_verts.shape



SELECTED_FRAME = 10

# Declare two sets P and Q, to compute rigid motion P -> Q
# and the weights for all points
P = t_pose = smpl_model(betas, torch.zeros_like(pose) ,trans)[0][0].detach().cpu().numpy()
Q = deformed_pose = V[SELECTED_FRAME]
W = np.ones(P.shape[0])

err = MSE_np(Q, P)
print("Error value before optimized: ", err)

# Compute the optimal rotation and translation and apply it to the first set
R, t = get_optimal_rigid_motion(P, Q, W)
P_star = P @ R + t

# Compute Mean Squared Error between two sets
err = MSE_np(Q, P_star)
print("Error value: ", err)

# Visualize the results
viewer = Matplot_Viewer()

mesh_T_node = Mesh(P, F)
mesh_optimized_node = Mesh(P_star, F)
mesh_target_node = Mesh(Q, F)

viewer.add_scene_node(mesh_T_node)
#viewer.add_scene_node(mesh_optimized_node)
#viewer.add_scene_node(mesh_target_node)

viewer.run()


