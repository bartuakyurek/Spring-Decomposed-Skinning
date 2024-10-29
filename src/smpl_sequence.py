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

import torch
import numpy as np
import pyvista as pv

from src.models.smpl_torch_batch import SMPLModel
from data.skeleton_data import get_smpl_skeleton

# ---------------------------------------------------------------------------- 
# Load SMPL animation file and get the mesh and associated rig data
# ---------------------------------------------------------------------------- 

training_data = torch.load('../data/50004_dataset.pt')
data_loader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=False)

device = "cpu"
smpl_model = SMPLModel(device=device, model_path='../src/models/body_models/smpl/female/model.pkl')
kintree = get_smpl_skeleton()
F = smpl_model.faces

for data in data_loader:
   beta_pose_trans_seq = data[0].squeeze().type(torch.float64)
   betas, pose, trans = beta_pose_trans_seq[:,:10], beta_pose_trans_seq[:,10:82], beta_pose_trans_seq[:,82:] 
   target_verts = data[1].squeeze()
   smpl_verts, joints = smpl_model(betas, pose, trans)
   break
   
V = smpl_verts.detach().cpu().numpy()
J = joints.detach().cpu().numpy()
n_frames, n_verts, n_dims = target_verts.shape

# -----------------------------------------------------------------------------
# Create a plotter object and set the scalars to the Z height
# -----------------------------------------------------------------------------
RENDER = True
plotter = pv.Plotter(notebook=False, off_screen=not RENDER)
plotter.camera_position = 'zy'
plotter.camera.azimuth = -90

skel_mesh = add_skeleton(plotter, J[0], kintree)
smpl_mesh = add_mesh(plotter, V[0], F, opacity=0.6)
# -----------------------------------------------------------------------------
# Initiate mass-spring system container to be attached to animated rig
# -----------------------------------------------------------------------------
dt = 1. / 24
mass_spring_system = MassSpringSystem(dt)

# Add masses to container and connect them, and fixate some of them.
n_masses =  2
mass = 10
# ---------------------------------------------------------------------
# TODO: Fix the masses to the bones...
# !!!!   !!!!   !!!!!!!   !!!!   !!!!!!!   !!!!   !!!!!!!   !!!!   !!!
#
# --------------------------------------------------------------------
mass_spring_system.add_mass(mass_coordinate=np.array([0,0,0]), mass=mass)
mass_spring_system.add_mass(mass_coordinate=np.array([0,0,1.0]), mass=mass)
mass_spring_system.connect_masses(0, 1)
mass_spring_system.fix_mass(0)
    
# Add masses with their initial locations to PyVista Plotter
initial_mass_locations = mass_spring_system.get_mass_locations()
mass_point_cloud = pv.PolyData(initial_mass_locations)
_ = plotter.add_mesh(mass_point_cloud, render_points_as_spheres=True,
                 show_vertices=True)
 
# Add springs connections actors in between to PyVista Plotter
spring_meshes = mass_spring_system.get_spring_meshes()
for spring_mesh in spring_meshes:
    plotter.add_mesh(spring_mesh)

# -----------------------------------------------------------------------------
# Run the mass-spring simulation based on rig motion
# -----------------------------------------------------------------------------
n_frames = 200 
for frame in range(n_frames):
    # ---------------------------------------------------------------------
    # Step 1 - TODO: Translate the masses that are connected to a bone 
    # !!!!   !!!!   !!!!!!!   !!!!   !!!!!!!   !!!!   !!!!!!!   !!!!   !!!
    #
    # ---------------------------------------------------------------------
    mass_spring_system.simulate()
    
    # Step 2 - Get current mass positions and update rendered particles
    cur_mass_locations = mass_spring_system.get_mass_locations()
    mass_point_cloud.points = cur_mass_locations 
    
    # Step 3 - Update the renderd connections based on new locations
    for i, mass_idx_tuple in enumerate(mass_spring_system.connections):
        spring_meshes[i].points[0] = cur_mass_locations[mass_idx_tuple[0]]
        spring_meshes[i].points[1] = cur_mass_locations[mass_idx_tuple[1]]

# -----------------------------------------------------------------------------
# Render the baked simulation using Pyvista
# -----------------------------------------------------------------------------
plotter.open_movie("../results/smpl-skeleton.mp4")

n_repeats = 3
for _ in range(n_repeats):
    for frame in range(n_frames-1):
        
        # TODO: Update mesh points
        skel_mesh.points = J[frame]
        smpl_mesh.points = V[frame]
        # Write a frame. This triggers a render.
        plotter.write_frame()

# Closes and finalizes movie
plotter.close()
plotter.deep_clean()