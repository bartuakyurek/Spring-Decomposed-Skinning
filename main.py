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
from plots import *
from viewer import *
from particle import *

def get_max_error_vertex_idx(mesh_array, num_max_verts=1):
    num_frames = mesh_array.shape[0] 
    vert_error = torch.zeros(mesh_array.shape[1])
    for frame in range(num_frames):
        single_frame_diff = target_verts[frame] - smpl_verts[frame].numpy() 
        # Take the square as absolute value (L1 norm) is not differentiable?
        error = torch.sum(torch.square(single_frame_diff), dim=1)
        vert_error += error
    
    max_errors_idx = torch.empty(num_max_verts, dtype=int)
    for i in range(num_max_verts):
        idx = torch.argmax(vert_error)
        max_errors_idx[i] = idx
        vert_error = torch.cat((vert_error[:idx], vert_error[idx+1:]))
    return max_errors_idx

# TODO: use './data/female_bpts2dbs.pt' 
# TODO: turn shuffle on for training dataset
# TODO: create validation and test splits as well
bpts2dbs_data = torch.load('./data/50004_dataset.pt')
training_data = bpts2dbs_data 
data_loader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=False)


# TODO: search for available devices (in colab demo)
# TODO: gather every path variable in a path.py file, e.g. SMPL_PATH = './path/to/smpl'
device = "cpu"
smpl_model = SMPLModel(device=device, model_path='./body_models/smpl/female/model.pkl')

for data in data_loader:
   beta_pose_trans_seq = data[0].squeeze().type(torch.float64)
   betas = beta_pose_trans_seq[:,:10]
   pose = beta_pose_trans_seq[:,10:82]
   trans = beta_pose_trans_seq[:,82:]

   target_verts = data[1].squeeze()
   smpl_verts, joints = smpl_model(betas, pose, trans)
   
   
   SELECTED_FRAME = 150
   joint_locations = joints[SELECTED_FRAME].numpy()
   kintree = get_smpl_skeleton()
   
   # -----------------------------------------------------------------------
   # -----------------------------------------------------------------------
   
   
   max_error_vert_idx = get_max_error_vertex_idx(smpl_verts, 1000)
   print(max_error_vert_idx)
   # Highligh the max vert idx
   viewer = Viewer()
   viewer.add_mesh(smpl_verts[SELECTED_FRAME].numpy(), smpl_model.faces)
   viewer.add_points(smpl_verts[SELECTED_FRAME].numpy()[max_error_vert_idx], point_size=15)
   viewer.run()
   
   # -----------------------------------------------------------------------
   # -----------------------------------------------------------------------
   
   #viewer = Viewer()
   #viewer.add_animated_mesh(smpl_verts.numpy(), smpl_model.faces)
   
   #faces = smpl_model.faces
   #verts = smpl_verts[SELECTED_FRAME].numpy()
   #plot_verts(verts, faces)
   #plot_obj_w_skeleton("./results/smpl_result.obj", joint_locations, kintree)
   #matplot_skeleton(joint_locations, kintree)
   #smpl_model.write_obj(smpl_verts[SELECTED_FRAME], './smpl_result.obj')
   #smpl_model.write_obj(target_verts[SELECTED_FRAME], './target.obj')
    
   # criterion = nn.MSELoss()
   # smpl_loss = criterion(smpl_verts+trans, target_verts)
   break


