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
import matplotlib.pyplot as plt

from smpl_torch_batch import SMPLModel

device = "cpu"

training_data = torch.load('./data/50004_dataset.pt')
data_loader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=False)
smpl_model = SMPLModel(device=device, model_path='./body_models/smpl/female/model.pkl')

for data in data_loader:
   beta_pose_trans_seq = data[0].squeeze().type(torch.float64)
   betas = beta_pose_trans_seq[:,:10]
   pose = beta_pose_trans_seq[:,10:82]
   trans = beta_pose_trans_seq[:,82:] 
   
   target_verts = data[1].squeeze()
   # -----------------------------------------------------------------------
   
   init_betas = torch.clone(betas)
   
   betas_param = torch.nn.Parameter(betas, requires_grad=True)
   pose_param = torch.nn.Parameter(pose, requires_grad=True)
   trans_param = torch.nn.Parameter(trans, requires_grad=True)

   n_iters = 70
   learning_rate = 1e-6
    
   criterion = torch.nn.MSELoss()
   optimizer = torch.optim.SGD([betas_param, pose_param, trans_param], learning_rate)

   smpl_model.eval() # ????
   train_loss = []
   for i in range(n_iters):
       
       smpl_verts, joints = smpl_model(betas_param, pose_param, trans_param)
       loss = criterion(target_verts, smpl_verts)
       
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()
       
       train_loss.append(loss.item())
       print(f"loss ({i}): {loss.item()}")
       
       #print(torch.sum(tmp_betas - betas_param))
    
    
   plt.plot(train_loss)
   
   # -----------------------------------------------------------------------
   # Break from for loop since we only wanna visualize one mesh rn
   break

     
