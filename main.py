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
   
   SELECTED_FRAME = 150
   joint_locations = joints[SELECTED_FRAME].numpy()
   kintree = get_smpl_skeleton()
   
   # -----------------------------------------------------------------------

   ### Declare data we want to visualize
   v = smpl_verts.detach().cpu().numpy()
   j = joints.detach().cpu().numpy()
   f = smpl_model.faces
   jpg_path = "./results/rendered_jpgs/"
   
   
   #############
   upto_this_frame = 10
   f1 = 10
   f2 = 100
   v_short = np.repeat(np.expand_dims(v[f1], axis=0), 100, axis=0)
   v_short[upto_this_frame:] = v[f2]
   
   j_short = np.repeat(np.expand_dims(j[f1], axis=0), 100, axis=0)
   j_short[upto_this_frame:] = j[f2]
   
   v = v_short
   j = j_short
   #############
   
   
   ### Manuel Spring Data
   spring_rest_locations = np.array([[0.4, 0.2, 0.0],
                            [0.1, 0.2, 0.3]])
   spring_parents =  [6, 20]
   
   ## Create viewer canvas and add a mesh (only single mesh is allowed rn)
   single_mesh_viewer = Viewer()
   single_mesh_viewer.set_mesh_animation(v, f)
   single_mesh_viewer.set_mesh_opacity(0.6)
   
   ## Skeleton and springs
   single_mesh_viewer.set_skeletal_animation(j, kintree)
   single_mesh_viewer.set_spring_rig(spring_parents, kintree)
   
   ## Run animation
   single_mesh_viewer.run_animation() #, jpg_dir=jpg_path+"{}.jpg")
   
   # -----------------------------------------------------------------------
   # Break from for loop since we only wanna visualize one mesh rn
   break

     
