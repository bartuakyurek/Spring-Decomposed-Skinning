#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is created to load an animation sequence from SMPL dataset. 
Available methods are:
                        get_gendered_smpl_model()
                        get_anim_sequence()
                        get_smpl_rest_data()

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
import h5py
import torch
import numpy as np

try:
    from ..models.smpl_torch_batch import SMPLModel
    from ..global_vars import MODEL_PATH, DFAUST_PATH, MODEL_REGIS_PATH
except:
    import sys
    sys.path.append('../') # For parent directory packages
    from models.smpl_torch_batch import SMPLModel
    from global_vars import MODEL_PATH, DFAUST_PATH, MODEL_REGIS_PATH


subject_ids = ['50004', '50020', '50021', '50022', '50025',
               '50002', '50007', '50009', '50026', '50027']

pose_ids = ['hips', 'knees', 'light_hopping_stiff', 'light_hopping_loose',
            'jiggle_on_toes', 'one_leg_loose', 'shake_arms', 'chicken_wings',
            'punching', 'shake_shoulders', 'shake_hips', 'jumping_jacks',
            'one_leg_jump', 'running_on_spot']

femaleids = subject_ids[:5]
maleids = subject_ids[5:]

                    
def get_smpl_rest_data(smpl_model, betas, trans):
    T_pose = np.zeros((1,72))
    V_rest, J_rest = smpl_model(betas, T_pose, trans)
    return V_rest, J_rest

def _retrieve_bpt_and_v(subject_id, pose_id, regis_dir):
    sidpid = subject_id + "_" + pose_id
    data_fname = os.path.join(DFAUST_PATH, subject_id, sidpid + '_poses.npz')

    f = h5py.File(regis_dir, 'r')
    verts = f[sidpid][()].transpose([2, 0, 1]) # np.ndarray
   
    verts = torch.Tensor(verts).type(torch.float64)
    bdata = np.load(data_fname)
    betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).squeeze().type(torch.float64)
    pose_body = torch.Tensor(bdata['poses'][:, 3:72]).squeeze().type(torch.float64)
    pose_body = torch.cat((torch.zeros(pose_body.shape[0], 3).type(torch.float64),pose_body),1)
    
    trans = torch.Tensor(bdata['trans']).type(torch.float64)
    num_frame = pose_body.shape[0]
    betas = betas.repeat(num_frame,1)
    
    return betas, pose_body, trans, verts

def _orientate_target_verts_to_smpl(verts, smpl_verts):
    """
    Translates the given vertices to the mean of smpl_vertices.
    TODO: There's nothing special about smpl_verts here, we could just
    implement this function as match_mean(u, v) etc.
    
    Parameters
    ----------
    verts : TYPE
        DESCRIPTION.
    smpl_verts : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    target_verts : TYPE
        DESCRIPTION.

    """
    assert verts.shape == smpl_verts.shape, f"Found erronous data, please try another pose id."
    assert len(verts.shape) == 3 and verts.shape[-1] == 3, f"Expected vertices to have shape (n_batch, n_frames, 3) or (n_frames, 3)\
                                   Got {verts.shape}."

    num_frame = verts.shape[0] # (n_frames, n_verts, 3)
    translation = torch.mean(verts - smpl_verts, 1).view(num_frame,1,3)
    target_verts = verts - translation
    return target_verts

def get_gendered_smpl_model(device, subject_id=None, model_gender=None):
    """
    Given the subject_id or model_gender, load the SMPL model and return it.
    
    
    Parameters
    ----------
    device : str
        Device to load SMPL model on. "cpu" or "cuda" options are available.
        
    subject_id : str
        Retrieve the related SMPL model given the subject ID. This option
        is optional, if not provided please provide model_gender parameter.
        Available options are:
            '50004', '50020', '50021', '50022', '50025', -> For female model
            '50002', '50007', '50009', '50026', '50027'. -> For male model
        Default is None. If provided, model_gender option will be overridden.
        
    model_gender : str
        If subject_id is not provided, this parameter is considered to 
        determine which SMPL model to be used. Currently "female" and "male"
        options are available.
        
    Returns
    -------
    smpl_model : SMPLModel
    
    """
    if subject_id in femaleids:
        model_gender = "female"
    elif subject_id in maleids:
        model_gender = "male"
    else:
        assert subject_id is None, f"subject_id {subject_id} is not available. Please use one of these {subject_ids} or provide model_gender"
        assert model_gender == "female" or model_gender == "male", f"model_gender {model_gender} is not available. Please select either male or female (sorry nonbinary folks :/ )."
    
    model_path = MODEL_PATH + 'smpl/'+ model_gender + '/model.pkl'
    smpl_model = SMPLModel(device=device, model_path=model_path)
    return smpl_model
    
def get_anim_sequence(subject_id, pose_id, smpl_model, return_numpy=True):
    """
    Retrieve the vertex and joints data of a particular animation sequence
    given the subject_id and pose_id.

    Parameters
    ----------
    subject_id : str
        Available options are:
            '50004', '50020', '50021', '50022', '50025',
            '50002', '50007', '50009', '50026', '50027'.
    pose_id : str
        Available options are:
            'hips', 'knees', 'light_hopping_stiff', 'light_hopping_loose',
            'jiggle_on_toes', 'one_leg_loose', 'shake_arms', 'chicken_wings',
            'punching', 'shake_shoulders', 'shake_hips', 'jumping_jacks',
            'one_leg_jump', 'running_on_spot'.

    Returns
    -------
    V_dfaust : torch.Tensor or numpy.ndarray
        Vertices of dynamically scanned human model, shape (n_frames, n_verts, 3).
    V_smpl : torch.Tensor or numpy.ndarray
        Vertices of SMPL model with one to one correspondence, shape (n_frames, n_verts, 3).
    J : torch.Tensor or numpy.ndarray
        Joint locations of the SMPL model, shape (n_frames, n_joints, 3)
    (betas, pose, trans) : tuple of torch.Tensors
        Parameters that fed in SMPL model, if needed.
    """
    
    betas, pose, trans, verts = _retrieve_bpt_and_v(subject_id, pose_id, regis_dir=MODEL_REGIS_PATH)
    V_smpl, J = smpl_model(betas, pose, trans) 
    V_dfaust = _orientate_target_verts_to_smpl(verts, V_smpl)
       
    if return_numpy:
        V_dfaust = V_dfaust.detach().cpu().numpy()
        V_smpl = V_smpl.detach().cpu().numpy()
        J = J.detach().cpu().numpy()
        
    return V_dfaust, V_smpl, J, (betas, pose, trans)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(">> Testing...")
    SELECTED_SUBJECT = femaleids[0]
    SELECTED_POSE = pose_ids[0]
    
    smpl_model = get_gendered_smpl_model(subject_id=SELECTED_SUBJECT, device="cpu")
    V_gt, V_smpl, J, _ = get_anim_sequence(SELECTED_SUBJECT, SELECTED_POSE, smpl_model, return_numpy=True)
    
    print(">> End of test.")
    