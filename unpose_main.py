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
import meshplot 

from smpl_torch_batch import SMPLModel
from dmpl_torch_batch import DMPLModel

from skinning import *
from skeleton_data import get_smpl_skeleton
from viewer import Viewer
from colors import SMPL_JOINT_COLORS

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
#dmpl_model = DMPLModel(device=device, model_path='./body_models/dmpls/male/model.npz')

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
   
   """
   smpl_verts_T, joints_T = smpl_model(betas, torch.zeros_like(pose), trans)
   
   theta = np.reshape(pose[SELECTED_FRAME], (24,3)) #### sample pose
   v = smpl_verts_T.detach().cpu().numpy()[SELECTED_FRAME]
   j = joints_T.detach().cpu().numpy()[SELECTED_FRAME]
   f = smpl_model.faces
   w = np.array(smpl_model.weights.detach().cpu())
   #np.savez("./results/skinning_T_pose_data.npz", v, f, j, theta, kintree, w)
   """
  
   """
   AYNISI....
   sequence_len = target_verts.shape[0]
   f = smpl_model.faces
   _, _, LBS_T = smpl_model(betas, pose, trans, return_T=True)

   v_unposed_seq = []
   for i in range(sequence_len):
       SELECTED_FRAME = i
       T = LBS_T[SELECTED_FRAME]
       T_inv = torch.inverse(T)
       
       tmp_verts = smpl_verts[SELECTED_FRAME] #target_verts[SELECTED_FRAME] 
       tmp_verts -= trans[SELECTED_FRAME]
       
       v_homo = torch.cat([tmp_verts,torch.ones([6890, 1])], dim=-1)
       v_unposed = torch.matmul(T_inv, torch.unsqueeze(v_homo, dim=-1)).squeeze()[ :,:3]
       
       v_unposed_seq.append(v_unposed)
       
   np.savez("./results/unposed_seq_smpl.npz", np.array(v_unposed_seq))
   
   #meshplot.offline()
   #meshplot.plot(np.array(v_unposed), f, filename="./unposed-dfaust.html")
   """
   
   _, _, LBS_T = smpl_model(betas, pose, trans, return_T=True)

   def unpose_verts_batch(target_verts, T):
       T_inv = torch.inverse(T)
       num_batch = target_verts.shape[0]
       
       tmp_verts = target_verts 
       tmp_verts -= torch.reshape(trans, (num_batch, 1, 3)) 
       
       v_homo = torch.cat([tmp_verts,torch.ones([num_batch, 6890, 1])], dim=2)
       v_unposed = torch.matmul(T_inv, torch.unsqueeze(v_homo, dim=-1)).squeeze()[:, :,:3]
       
       return v_unposed

   v_unposed = unpose_verts_batch(target_verts, LBS_T)
   np.savez("./results/unposed_dfaust.npz", v_unposed)
   

   """
   ### Declare data we want to visualize
   smpl_verts_T, joints_T = smpl_model(betas, torch.zeros_like(pose), trans)

   theta = np.reshape(pose[SELECTED_FRAME], (24,3))
   v = smpl_verts.detach().cpu().numpy()[SELECTED_FRAME]
   j = joints.detach().cpu().numpy()[SELECTED_FRAME]
   f = smpl_model.faces
   w = np.array(smpl_model.weights.detach().cpu())
   """
   
   """
   # Color the weights data of SMPL to verify bone-vertex weights.
   w = np.array(smpl_model.weights.detach().cpu())
   joint_colors = np.array(SMPL_JOINT_COLORS)
   
   vert_colors = w @ joint_colors
   np.savez("smpl_weight_vertex_colors.npz", vert_colors)
   """
   
   """
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
   """
   """
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
   """
   
   # -----------------------------------------------------------------------
   # Break from for loop since we only wanna visualize one mesh rn
   break

     
