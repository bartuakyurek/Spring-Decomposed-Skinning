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
   
   # Write one frame of the resulting .objs
   SELECTED_FRAME = 150
   joint_locations = joints[SELECTED_FRAME].numpy()
   kintree = get_smpl_skeleton()
   
   #matplot_skeleton(joint_locations, kintree)
   #smpl_model.write_obj(smpl_verts[SELECTED_FRAME], './smpl_result.obj')
   #smpl_model.write_obj(target_verts[SELECTED_FRAME], './target.obj')
    
   plot_obj_w_skeleton("./results/smpl_result.obj", joint_locations, kintree)
   
   # criterion = nn.MSELoss()
   # smpl_loss = criterion(smpl_verts+trans, target_verts)
   break


