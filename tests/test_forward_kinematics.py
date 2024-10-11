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


pose = np.array([
                [
                 [0.,0.,0.],
                 [0.,0.,0.],
                 [0., 0., 0.],
                 [0.,0.,0.],
                 [0.,0.,0.],
                ],
                [
                 [10.,0.,0.],
                 [30.,0. ,0.],
                 [45., 45., 0.],
                 [0.,0.,10.],
                 [0.,0.,0.],
                ]
                ])
"""
d_t = np.zeros((n_bones-1,3))
d_q = np.zeros((n_bones-1,4))
from scipy.spatial.transform import Rotation
d_q[1] = Rotation.from_euler('xyz', pose[1][2]).as_quat()
P = igl.directed_edge_parents(kintree)
abs_rot, abs_t = igl.forward_kinematics(joint_locations, kintree, P, d_q, d_t)
"""
n_repeats = 24
n_frames = 2
for _ in range(n_repeats):    
    for frame in range(n_frames):
            
        for _ in range(24):
            
            theta = pose[frame]
            posed_bone_locations = test_skeleton.pose_bones(theta, degrees=True, exclude_root=True)
            skel_mesh.points = posed_bone_locations
            
            # Write a frame. This triggers a render.
            plotter.write_frame()

# Closes and finalizes movie
plotter.close()
plotter.deep_clean()
    