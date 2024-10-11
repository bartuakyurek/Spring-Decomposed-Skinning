#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:24:07 2024

This file loads an SMPL animation, and creates a Skeleton class
by using SMPL's joint locations and kinematic tree data. The skeleton class
then is used to call forward kinematics to pose the skeleton bones with
respect to relative transformation data SMPL is providing. The skeletal 
animation is displayed via PyVista's rendering interface.

Note that the actual SMPL joint locations differ from our forward kinematics
based joints. That is because SMPL uses a regressor to estimate joint locations
that is not the same with computing the locations analitically.

@author: bartu
"""

import torch
import numpy as np
import pyvista as pv

import __init__
from skeleton import Skeleton
from smpl_torch_batch import SMPLModel
from skeleton_data import get_smpl_skeleton
from pyvista_render_tools import add_skeleton
from global_vars import DATA_PATH, MODEL_PATH, RESULT_PATH

# ---------------------------------------------------------------------------- 
# Load SMPL animation file and get the mesh and associated rig data
# ---------------------------------------------------------------------------- 
data_loader = torch.utils.data.DataLoader(torch.load(DATA_PATH+'50004_dataset.pt'), batch_size=1, shuffle=False)
smpl_model = SMPLModel(device="cpu", model_path = MODEL_PATH +'smpl/female/model.pkl')
kintree = get_smpl_skeleton()
for data in data_loader:
   beta_pose_trans_seq = data[0].squeeze().type(torch.float64)
   betas, pose, trans = beta_pose_trans_seq[:,:10], beta_pose_trans_seq[:,10:82], beta_pose_trans_seq[:,82:] 
   target_verts = data[1].squeeze()
   smpl_verts, joints = smpl_model(betas, pose, trans)
   break
V = smpl_verts.detach().cpu().numpy()
J = joints.detach().cpu().numpy()
n_frames, n_verts, n_dims = target_verts.shape

# Get rest pose SMPL data
rest_verts, rest_joints = smpl_model(betas, torch.zeros_like(pose), trans)
J_rest = rest_joints.numpy()[0]

# ---------------------------------------------------------------------------- 
# Create skeleton based on rest pose SMPL data
# ---------------------------------------------------------------------------- 
smpl_skeleton = Skeleton(root_vec = J_rest[0])
for edge in kintree:
    parent_idx, bone_idx = edge
    smpl_skeleton.insert_bone(endpoint_location = J_rest[bone_idx], 
                              parent_node_idx = parent_idx)
    
# ---------------------------------------------------------------------------- 
# Create plotter 
# ---------------------------------------------------------------------------- 
RENDER = True
plotter = pv.Plotter(notebook=False, off_screen=not RENDER)
plotter.camera_position = 'zy'
plotter.camera.azimuth = -90

# ---------------------------------------------------------------------------- 
# Add skeleton mesh based on T-pose locations
# ---------------------------------------------------------------------------- 
n_bones = len(smpl_skeleton.bones)
rest_bone_locations = smpl_skeleton.get_rest_bone_locations(exclude_root=True)
line_segments = np.reshape(np.arange(0, 2*(n_bones-1)), (n_bones-1, 2))

skel_mesh = add_skeleton(plotter, rest_bone_locations, line_segments)
plotter.open_movie(RESULT_PATH + "smpl-skeleton.mp4")

n_repeats = 1
n_frames = len(J)
for _ in range(n_repeats):
    for frame in range(n_frames):
        
        # TODO: Update mesh points
        theta = np.reshape(pose[frame].numpy(), newshape=(-1, 3))
        t = trans[frame].numpy()
        # TODO: (because it's global t, should be only applied to root)
        # we're not using t, we should handle it after correcting the FK.
     
        posed_bone_locations = smpl_skeleton.pose_bones(theta, exclude_root=True)
        skel_mesh.points = posed_bone_locations
        
        # Write a frame. This triggers a render.
        plotter.write_frame()
        
# Closes and finalizes movie
plotter.close()
plotter.deep_clean()