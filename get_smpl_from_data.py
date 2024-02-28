#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:54:30 2024

@author: bartu

This file extracts SMPL poses from ground truth of DFAUST dataset,
to inspect the difference between scanned human registrations and 
SMPL model output.

"""

import os
import numpy as np
import torch
from smpl_torch_batch import SMPLModel
#from smpl_loader_dbs import SMPLModel

bpts2dbs_data = torch.load('female_bpts2dbs.pt')

training_data = bpts2dbs_data 

data_loader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=False)

device = "cpu"
smpl_model = SMPLModel(device=device)

for data in data_loader:
   beta_pose_trans_seq = data[0].squeeze().type(torch.float64)
   betas = beta_pose_trans_seq[:,:10]
   pose = beta_pose_trans_seq[:,10:82]
   trans = beta_pose_trans_seq[:,82:]

   target_verts = data[1].squeeze()

   smpl_result = smpl_model(betas, pose, trans)
   
   # Write one frame of the resulting .objs
   # TODO: why write_obj is at smpl? it has nothing to do with it.
   smpl_model.write_obj(smpl_result[150], './smpl_result.obj')
   smpl_model.write_obj(target_verts[150], './target.obj')
   break
    
   #outmesh_path = './smpl_torch_{}.obj'
    # model.write_obj(result, outmesh_path.format(i))
