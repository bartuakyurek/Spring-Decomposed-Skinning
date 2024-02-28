#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Borrowed from DBS;

This file extracts female poses in DFAUST dataset.
"""

import os
import glob
import h5py
import numpy as np
import torch
from torch import nn
from smpl_torch_batch import SMPLModel
from time import time
from tqdm import tqdm


"""
sids = ['50004', '50020', '50021', '50022', '50025',
        '50002', '50007', '50009', '50026', '50027']

pids = ['hips', 'knees', 'light_hopping_stiff', 'light_hopping_loose',
        'jiggle_on_toes', 'one_leg_loose', 'shake_arms', 'chicken_wings',
        'punching', 'shake_shoulders', 'shake_hips', 'jumping_jacks',
        'one_leg_jump', 'running_on_spot']

femaleids = sids[:5]
maleids = sids[5:]
"""

# -------------------------------------------------------------
# ------------------------ WARNING ----------------------------
# -------------------------------------------------------------
femaleids = ['50004']
pids = ['one_leg_loose']
# -------------------------------------------------------------
# -------------------------------------------------------------

regis_dir = './dyna/dyna_dataset_f.h5'
data_dir = './DFaust_67'

f = h5py.File(regis_dir, 'r')

comp_device = torch.device('cpu')
smplmodel = SMPLModel(device=comp_device)
# dmplmodel = DMPLModel(device=comp_device)
# dbsmodel = DBSModel(device=comp_device)

dataset = []


for femaleid in femaleids:
    print('\n{} now is being processed:'.format(femaleid))
    data_fnames = glob.glob(os.path.join(data_dir, femaleid, '*_poses.npz'))
    for data_fname in tqdm(data_fnames):
                
        sidpid = data_fname[18:-10]
        verts = f[sidpid][()].transpose([2, 0, 1])
        verts = torch.Tensor(verts).type(torch.float64)
        bdata = np.load(data_fname)
        betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).squeeze().type(torch.float64)
        pose_body = torch.Tensor(bdata['poses'][:, 3:72]).squeeze().type(torch.float64)
        pose_body = torch.cat((torch.zeros(pose_body.shape[0], 3).type(torch.float64),pose_body),1)
        if pose_body.shape[0]-verts.shape[0] != 0:
            # print(data_fname, pose_body.shape[0]-verts.shape[0])
            continue
        trans = torch.Tensor(bdata['trans']).type(torch.float64)
        #dmpls = torch.Tensor(bdata['dmpls']).type(torch.float64)
        num_frame = pose_body.shape[0]
        betas = betas.repeat(num_frame,1)
        

        smpl_vert = smplmodel(betas, pose_body, trans)
       
        translation = torch.mean(verts - smpl_vert, 1).view(num_frame,1,3)
        tar_bs = verts - translation
        # criterion = nn.MSELoss()
        # smpl_loss = criterion(smpl_vert+translation, verts)
        
        bpts = torch.cat((betas,pose_body,trans),1)
        dataset.append((bpts,tar_bs))
        
    
    
torch.save(dataset, 'female_bpts2dbs.pt')

    
        
        
        