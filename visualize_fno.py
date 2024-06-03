#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 10:43:51 2024

@author: bartu
"""

import os
import numpy as np
import torch
from neuralop.models import TFNO

from smpl_torch_batch import SMPLModel
from skeleton_data import get_smpl_skeleton
from viewer import Viewer

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
   
   # -----------------------------------------------------------------------

   ### Declare data we want to visualize
   v = smpl_verts.detach().cpu().numpy()
   f = smpl_model.faces   

   break

PATH = './saved_models/epoch_400.pkl'
"""
model = TFNO(n_modes=(16, 16),
             hidden_channels=64,
             in_channels=3,
             out_channels=3,
             factorization='tucker',
             implementation='factorized',
             rank=0.05)
model.to('cpu')
"""
model = torch.load(PATH, map_location=torch.device('cpu'))
model.eval()

v = smpl_verts.permute(2,0,1)[None,:,:,:].type(torch.float32)
v_fno = (model(v).detach().cpu() + v)[0].permute(1, 2, 0).numpy()

## Create viewer canvas and add a mesh (only single mesh is allowed rn)
single_mesh_viewer = Viewer()
single_mesh_viewer.set_mesh_animation(v_fno, f)
single_mesh_viewer.set_mesh_opacity(0.6)

## Run animation
single_mesh_viewer.run_animation() #, jpg_dir=jpg_path+"{}.jpg")


