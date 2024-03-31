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
from signal_filtering import *

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
   
   
   viewer = Viewer()
   viewer.add_animated_mesh(smpl_verts.numpy(), smpl_model.faces)
   viewer.run()
   
   """
   selected_vert = 4000
   
   target_x = target_verts[:,selected_vert, 0]
   target_y = target_verts[:,selected_vert, 1]
   target_z = target_verts[:,selected_vert, 2]
   
   smpl_x = smpl_verts[:,selected_vert, 0]
   smpl_y = smpl_verts[:,selected_vert, 1]
   smpl_z = smpl_verts[:,selected_vert, 2]
   
   draw_same_length_signals([target_x, target_y, target_z], " Original for vertex " + str(selected_vert))
   draw_same_length_signals([smpl_x, smpl_y, smpl_z], " SMPL for vertex " + str(selected_vert))
   
   target_x_freqs, target_x_amps = get_FFT(target_x)
   smpl_x_freqs, smpl_x_amps = get_FFT(smpl_x)
   
   draw_FFT(target_x_freqs, target_x_amps, show=False)
   draw_FFT(smpl_x_freqs, smpl_x_amps, color='red')
   """
   
   """
   # Draw joint motion
   selected_joint = 5
   joint_x = joints[:, selected_joint, 0] 
   joint_y = joints[:, selected_joint, 1] 
   joint_z = joints[:, selected_joint, 2] 
   draw_same_length_signals([joint_x, joint_y, joint_z], " Joint " + str(selected_joint))

   # Draw global translation components 
   draw_same_length_signals([trans[:,0] , trans[:,1], trans[:,2]], " Global Translations ")
   """
   
   # -----------------------------------------------------------------------
   # -----------------------------------------------------------------------
   
   
   #faces = smpl_model.faces
   #verts = smpl_verts[SELECTED_FRAME].numpy()
   #plot_verts(verts, faces)
   #plot_obj_w_skeleton("./results/smpl_result.obj", joint_locations, kintree)
   #matplot_skeleton(joint_locations, kintree)
   #smpl_model.write_obj(smpl_verts[SELECTED_FRAME], './smpl_result.obj')
   #smpl_model.write_obj(target_verts[SELECTED_FRAME], './target.obj')

   break


