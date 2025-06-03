#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DISCLAIMER: This file is borrowed from https://github.com/haoranchen06/DBS 
(compute_target_pose.py), slightly changed according to project needs.

This file extracts poses of DFAUST dataset in SMPL format,
and saves a dataset that includes [beta, pose, trans] for every instance.


[!] WARNING: Right now it only extracts female poses
"""

import os
import glob
import h5py
import torch
import numpy as np
from torch import nn
from time import time
from tqdm import tqdm

import sys
sys.path.append('../') # For parent directory packages
from models.smpl_torch_batch import SMPLModel


sids = ['50004', '50020', '50021', '50022', '50025',
        '50002', '50007', '50009', '50026', '50027']

pids = ['hips', 'knees', 'light_hopping_stiff', 'light_hopping_loose',
        'jiggle_on_toes', 'one_leg_loose', 'shake_arms', 'chicken_wings',
        'punching', 'shake_shoulders', 'shake_hips', 'jumping_jacks',
        'one_leg_jump', 'running_on_spot']

femaleids = sids[:5]
maleids = sids[5:]

regis_dir = '../../data/dyna/dyna_dataset_f.h5'
data_dir = '../../data/DFaust_67'

f = h5py.File(regis_dir, 'r')

comp_device = torch.device('cpu')
smplmodel = SMPLModel(model_path='../models/body_models/smpl/female/model.pkl',device=comp_device)

def get_instance_data(sidpid):
    verts = f[sidpid][()].transpose([2, 0, 1])
    verts = torch.Tensor(verts).type(torch.float64)
    bdata = np.load(data_fname)
    betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).squeeze().type(torch.float64)
    pose_body = torch.Tensor(bdata['poses'][:, 3:72]).squeeze().type(torch.float64)
    pose_body = torch.cat((torch.zeros(pose_body.shape[0], 3).type(torch.float64),pose_body),1)
    
    trans = torch.Tensor(bdata['trans']).type(torch.float64)
    num_frame = pose_body.shape[0]
    betas = betas.repeat(num_frame,1)
    
    smpl_vert, _ = smplmodel(betas, pose_body, trans) 
    translation = torch.mean(verts - smpl_vert, 1).view(num_frame,1,3)
    target_verts = verts - translation
    assert target_verts.shape == verts.shape
    return betas, pose_body, trans, target_verts

dataset = []
for femaleid in femaleids:
    print('\n{} now is being processed:'.format(femaleid))
    data_fnames = glob.glob(os.path.join(data_dir, femaleid, '*_poses.npz'))
    
    for data_fname in tqdm(data_fnames):
        
        sidpid = os.path.basename(data_fname)[:-len("_poses.npz")] 
        betas, pose_body, trans, target_verts = get_instance_data(sidpid)
        
        if pose_body.shape[0] != target_verts.shape[0]:
            print(">> Found erronous data, skipping...")
            continue
        bpts = torch.cat((betas,pose_body,trans),1)
        dataset.append((bpts,target_verts))
        
    
torch.save(dataset, 'female_bpts2dbs.pt')

        
        
        