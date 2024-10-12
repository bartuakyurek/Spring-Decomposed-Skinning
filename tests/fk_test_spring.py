#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is created to test the Skeleton class and it's forward kinematics
to see if the kinematics are implemented correctly.
Created on Thu Oct 10 14:34:34 2024
@author: bartu
"""
import igl
import numpy as np
import pyvista as pv

import __init__
from skeleton import Skeleton
from pyvista_render_tools import add_skeleton
from global_vars import IGL_DATA_PATH, RESULT_PATH

TGF_PATH = IGL_DATA_PATH + "arm.tgf"
joint_locations, kintree, _, _, _, _ = igl.read_tgf(TGF_PATH)

# ---------------------------------------------------------------------------- 
# Create skeleton based on loaded data
# ---------------------------------------------------------------------------- 
# TODO: if you always create root_vec based on your first data, why don't you 
#       include it in the insert_bone phase to make it easier to read?
test_skeleton = Skeleton(root_vec = joint_locations[0])
for edge in kintree:
     parent_idx, bone_idx = edge
     test_skeleton.insert_bone(endpoint = joint_locations[bone_idx], 
                               parent_idx = parent_idx)

helper_bone_endpoints = np.array([ joint_locations[2] + [0.0, 0.2, 0.0] ])
helper_bone_parents = [2]
n_helper = len(helper_bone_parents)

for i in range(n_helper):
    test_skeleton.insert_bone(endpoint = helper_bone_endpoints[i],
                              parent_idx = helper_bone_parents[i],
                              at_the_tip=False,
                              offset_ratio = 1.0,
                              #startpoint = test_skeleton.bones[2].end_location-test_skeleton.bones[2].start_location
                              )

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
EXCLUDE_ROOT = True
n_bones = len(test_skeleton.bones)
rest_bone_locations = test_skeleton.get_rest_bone_locations(exclude_root=EXCLUDE_ROOT)
line_segments = np.reshape(np.arange(0, 2*(n_bones-1)), (n_bones-1, 2))

skel_mesh = add_skeleton(plotter, rest_bone_locations, line_segments)
plotter.open_movie(RESULT_PATH + "/igl-skeleton.mp4")

n_repeats = 20
n_frames = 2
pose = np.array([
                [
                 [0.,0.,0.],
                 [0.,0.,0.],
                 [0., 0., 0.],
                 [0.,0.,0.],
                 [0.,0.,0.],
                 [0.,0.,0.],
                ],
                [
                 [0.,0.,0.],
                 [0.,0.,0.],
                 [0., 0., 0.],
                 [0.,0.,0.],
                 [0.,0.,0.],
                 [0.,0.,0.],
                ]
                ])
"""
trans = np.array([
                [
                 [0.,0.,0.],
                 [0.,0.,0.],
                 [0., 0., 0.],
                 [0.,0.,0.],
                 [0.,0.,0.],
                ],
                [
                 [0.0,0.,0.],
                 [0.,0.,0.],
                 [0., 0., 0.],
                 [0.,0.,0.0],
                 [0.,0.,0.],
                ]
                ])
"""
for _ in range(n_repeats):
    for frame in range(n_frames):
        for _ in range(24):
            theta = pose[frame]
            #t = trans[frame]
            posed_bone_locations = test_skeleton.pose_bones(theta, degrees=True, exclude_root=EXCLUDE_ROOT)
            skel_mesh.points = posed_bone_locations
    
            # Write a frame. This triggers a render.
            plotter.write_frame()

# Closes and finalizes movie
plotter.close()
plotter.deep_clean()