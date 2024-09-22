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
   
#V = smpl_verts.detach().cpu().numpy()#[SELECTED_FRAME]
#J = joints.detach().cpu().numpy()#[SELECTED_FRAME]
n_frames, n_verts, n_dims = target_verts.shape

### Manual Spring Data 
P = np.array([
                [0.5, 3.0, 0.5],
                [2.0, 3.0, 0.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ])
S = np.array([
                [0, 2],
                [1, 2],
                [2, 4],
                [2, 3],
                [3, 4]
            ])
   
viewer = Matplot_Viewer()
viewer.run()

