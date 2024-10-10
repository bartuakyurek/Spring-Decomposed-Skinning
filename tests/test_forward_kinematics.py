#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file is created to test the Skeleton class and it's forward kinematics
to see if the kinematics are implemented correctly.


Created on Thu Oct 10 14:34:34 2024
@author: bartu
"""
import igl
import __init__
import numpy as np
import pyvista as pv
from pyvista_render_tools import add_skeleton
from skeleton import Skeleton

TGF_PATH = "/Users/bartu/Desktop/MS/simplified-deltamush-libigl/libigl-data/arm.tgf"
joint_locations, kintree, _, _, _, _ = igl.read_tgf(TGF_PATH)

# ---------------------------------------------------------------------------- 
# Create skeleton based on loaded data
# ---------------------------------------------------------------------------- 
# TODO: if you always create root_vec based on your first data, why don't you 
#       include it in the insert_bone phase to make it easier to read?
test_skeleton = Skeleton(root_vec = joint_locations[0])
for edge in kintree:
     parent_idx, bone_idx = edge
     test_skeleton.insert_bone(endpoint_location = joint_locations[bone_idx], 
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
n_bones = len(test_skeleton.bones)
rest_bone_locations = test_skeleton.get_rest_bone_locations(exclude_root = True)
line_segments = np.reshape(np.arange(0, 2*(n_bones-1)), (n_bones-1, 2))

skel_mesh = add_skeleton(plotter, rest_bone_locations, line_segments)
plotter.open_movie("../results/smpl-skeleton.mp4")

n_repeats = 100
n_frames = 2
pose = np.array([
                [
                 [0.,0.,0.],
                 [0.,0.,0.],
                 [0., 0., 0.],
                 [0.,0.,0.],
                 [0.,0.,0.],
                ],
                [
                 [0.,0.,0.],
                 [0.,0.,0.],
                 [45., 45., 0.],
                 [0.,0.,0.],
                 [0.,0.,0.],
                ]
                ])
for _ in range(n_repeats):
    for frame in range(n_frames):
        
        # TODO: Update mesh points
        theta = np.reshape(pose[frame], newshape=(-1, 3))
        posed_bone_locations = test_skeleton.pose_bones(theta)
       
        current_skel_data = np.reshape(posed_bone_locations[2:], (2*(n_bones-1), 3))
        skel_mesh.points = current_skel_data
        
        # Write a frame. This triggers a render.
        plotter.write_frame()

# Closes and finalizes movie
plotter.close()
plotter.deep_clean()
    