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

training_data = torch.load('./data/50004_dataset.pt')
data_loader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=False)

device = "cpu"
smpl_model = SMPLModel(device=device, model_path='./body_models/smpl/female/model.pkl')
kintree = get_smpl_skeleton()
F = smpl_model.faces

for data in data_loader:
   beta_pose_trans_seq = data[0].squeeze().type(torch.float64)
   betas, pose, trans = beta_pose_trans_seq[:,:10], beta_pose_trans_seq[:,10:82], beta_pose_trans_seq[:,82:] 
   target_verts = data[1].squeeze()
   smpl_verts, joints = smpl_model(betas, pose, trans)
   break
   
V = smpl_verts.detach().cpu().numpy()#[SELECTED_FRAME]
J = joints.detach().cpu().numpy()#[SELECTED_FRAME]
n_frames, n_verts, n_dims = target_verts.shape

### Manual Spring Data 
P = np.array([
                [0.5, 3.0, 0.5],
                [2.0, 3.0, 0.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ])
S = np.array([
                [0, 2],
                [1, 2],
                [2, 4],
                [2, 3],
                [3, 4]
            ])
   
import open3d as o3d

mesh_animation_list = []
for i in range(n_frames):
    mesh_faces = o3d.utility.Vector3iVector(F)
    mesh_verts = o3d.utility.Vector3dVector(V[i])
    mesh = o3d.geometry.TriangleMesh(mesh_verts, mesh_faces)
    mesh.compute_vertex_normals()
    mesh_animation_list.append(mesh)

""""
def test_animation_callback():
    oct = o3d.geometry.TriangleMesh.create_octahedron()
    def cb_test(vis, time):
        print('in cb_test, time =', time)
        oct.paint_uniform_color(np.random.rand(3))
        
    o3d.visualization.draw({'name': 'oct', 'geometry': oct},
                           on_animation_frame = cb_test,
                           animation_time_step = 1 / 60,
                           animation_duration = 1000000,
                           show_ui=True)
    
test_animation_callback()
"""

current_frame = 0
def custom_draw_geometry_with_rotation(pcd_list):
    global current_frame
    
    if current_frame >= n_frames: 
        print("Animation ended.")
        current_frame = 0
    else: 
        current_frame += 1
    
    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd_list[current_frame]],
                                                              rotate_view)

custom_draw_geometry_with_rotation(mesh_animation_list)



"""

vis = o3d.visualization.Visualizer()
vis.create_window()

# geometry is the point cloud used in your animation
geometry = o3d.geometry.TriangleMesh()
#geometry.triangles = F
#geometry.vertices = V[0]
vis.add_geometry(geometry)


for i in range(num_frames):
    # now modify the points of your geometry
    # you can use whatever method suits you best, this is just an example
    geometry.points = pcd_list[i].points
    vis.update_geometry(geometry)
    vis.poll_events()
    vis.update_renderer()
"""

     
