#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is created to load an animation sequence from SMPL dataset. 

Input:
      - Subject name
      - Sequence number
     
Output:
      For T pose:
      -----------
      - V_smpl_rest: SMPL vertices at rest pose, has shape (n_verts, 3)
      - F: Faces of body models has shape (n_faces, 3)
      - J_rest : Joint locations at rest pose, has shape (n_joints, 3)
      - kintree: Tree hierarchy for joint kinematics, has shape (n_bones, 2) 
                                                      where n_bones = n_joints - 1
      
      For animation sequence:
      -----------------------
      - V_gt : DFAUST vertices, has shape (n_frames, n_verts, 3) as ground truth dynamic deformation
      - V_smpl : SMPL vertices, has shape (n_frames, n_verts, 3) as rigid deformation
      - J : Joint locations, has shape (n_frames, n_joints, 3)
      
Created on Tue Oct 29 07:34:53 2024
@author: bartu
"""

import os
import torch
import numpy as np
import sys
sys.path.append('../') # For parent directory packages

from models.smpl_torch_batch import SMPLModel
from skeleton_data import get_smpl_skeleton
from global_vars import DATA_PATH, MODEL_PATH, DFAUST_PATH


subject_ids = ['50004', '50020', '50021', '50022', '50025',
               '50002', '50007', '50009', '50026', '50027']

pose_ids = ['hips', 'knees', 'light_hopping_stiff', 'light_hopping_loose',
            'jiggle_on_toes', 'one_leg_loose', 'shake_arms', 'chicken_wings',
            'punching', 'shake_shoulders', 'shake_hips', 'jumping_jacks',
            'one_leg_jump', 'running_on_spot']

femaleids = subject_ids[:5]
maleids = subject_ids[5:]

                    
def get_smpl_rest_data(betas, trans):
    T_pose = np.zeros((1,72))
    V_rest, J_rest = smpl_model(betas, T_pose, trans)
    kintree = get_smpl_skeleton()
    F = smpl_model.faces
    return V_rest, J_rest, F, kintree

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
    
def get_anim_sequence(subject_id, pose_id, device="cpu"):
    
    if sequence_id in femaleids:
        model_gender = "female"
    elif sequence_id in maleids:
        model_gender = "male"
    else:
        raise ValueError("Neutral models not implemented yet.")
    
    model_path = MODEL_PATH + 'smpl/'+ model_gender + '/model.pkl'
    smpl_model = SMPLModel(device=device, model_path=model_path)
    
    f = os.path.join(DFAUST_PATH, subject_id, "_", pose_id,'_poses.npz')
    
    for data in training_data:
       beta_pose_trans_seq = data[0].squeeze().type(torch.float64)
       betas, pose, trans = beta_pose_trans_seq[:,:10], beta_pose_trans_seq[:,10:82], beta_pose_trans_seq[:,82:] 
       V_dfaust = data[1].squeeze()
       smpl_verts, joints = smpl_model(betas, pose, trans)
       break

    V_smpl = smpl_verts.detach().cpu().numpy()
    J = joints.detach().cpu().numpy()
    
    return V_dfaust, V_smpl, J, (betas, pose, trans)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    
    
    SELECTED_SUBJECT = femaleids[0]
    SELECTED_POSE = pose_ids[0]
    V_gt, V_smpl, J, bpt = get_anim_sequence(SELECTED_SUBJECT, SELECTED_POSE)