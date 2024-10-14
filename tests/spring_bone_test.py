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
from helper_rig import HelperBonesHandler
from pyvista_render_tools import add_skeleton
from global_vars import IGL_DATA_PATH, RESULT_PATH

# ---------------------------------------------------------------------------- 
# Declare helper functions
# ---------------------------------------------------------------------------- 
# TODO: Could we move these functions to skeleton class so that every other test
# can utilize them?
def create_skeleton(joint_locations, kintree):
    test_skeleton = Skeleton(root_vec = joint_locations[0])
    for edge in kintree:
         parent_idx, bone_idx = edge
         test_skeleton.insert_bone(endpoint = joint_locations[bone_idx], 
                                   parent_idx = parent_idx)   
    return test_skeleton

def add_helper_bones(test_skeleton, helper_bone_endpoints, helper_bone_parents,
                     offset_ratio=0.0, startpoints=[]):
    n_helper = len(helper_bone_parents)
    if len(startpoints)==0: startpoints = np.repeat([None],n_helper)
    
    helper_indices = []
    for i in range(n_helper):
        bone_idx = test_skeleton.insert_bone(endpoint = helper_bone_endpoints[i],
                                              parent_idx = helper_bone_parents[i],
                                              at_the_tip=False,
                                              offset_ratio = offset_ratio,
                                              startpoint = startpoints[i]
                                              )
        helper_indices.append(bone_idx)
    return helper_indices

def lerp(arr1, arr2, ratio):
    
    return ((1.0 - ratio) * arr1) + (ratio * arr2)

# ---------------------------------------------------------------------------- 
# Set skeletal animation data
# ---------------------------------------------------------------------------- 
TGF_PATH = IGL_DATA_PATH + "arm.tgf"
joint_locations, kintree, _, _, _, _ = igl.read_tgf(TGF_PATH)

helper_bone_endpoints = np.array([ joint_locations[2] + [0.0, 0.2, 0.0] ])
helper_bone_parents = [2]

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
                 [0., 10., 40.],
                 [0.,0.,0.],
                 [0.,0.,0.],
                 [0.,0.,0.],
                ]
                ])

DEGREES = True # Set true if pose is represented with degrees as Euler angles.
MODE = "Dynamic" #"Rigid" or "Dynamic"
MASS = 0.1
STIFFNESS = 0.3
MASS_DSCALE = 1.0        # Range [0.0, 1.0] Scales mass velocity
SPRING_DSCALE = 1.0      # Range [0.0, 1.0]
DAMPING = 0.4            # TODO: Why increasing damping makes the stability worse?
TIME_STEP = 1/30
POINT_SPRING = False
FRAME_RATE = 60 #24

# ---------------------------------------------------------------------------- 
# Create rig and set helper bones
# ---------------------------------------------------------------------------- 

test_skeleton = create_skeleton(joint_locations, kintree)
helper_indices = add_helper_bones(test_skeleton, helper_bone_endpoints, 
                                     helper_bone_parents, #offset_ratio=0.0,
                                     startpoints=helper_bone_endpoints-1e-6)

helper_rig = HelperBonesHandler(test_skeleton, 
                                helper_indices,
                                mass          = MASS, 
                                stiffness     = STIFFNESS,
                                damping       = DAMPING,
                                mass_dscale   = MASS_DSCALE,
                                spring_dscale = SPRING_DSCALE,
                                dt            = TIME_STEP,
                                point_spring  = POINT_SPRING) 

# TODO: you could also add insert_point_handle() to Skeleton class
# that creates a zero-length bone (we need to render bone tips as spheres to see that)

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
n_bones = len(test_skeleton.rest_bones)
rest_bone_locations = test_skeleton.get_rest_bone_locations(exclude_root=EXCLUDE_ROOT)
line_segments = np.reshape(np.arange(0, 2*(n_bones-1)), (n_bones-1, 2))
# TODO: rename get_rest_bone_locations() to get_rest_bones() that will also return
# line_segments based on exclude_root variable
# (note that you need to re-run other skeleton tests)

skel_mesh = add_skeleton(plotter, rest_bone_locations, line_segments)
plotter.open_movie(RESULT_PATH + "/igl-skeleton.mp4")

n_repeats = 10
n_poses = pose.shape[0]
trans = None # TODO: No relative translation yet...
for _ in range(n_repeats):
    for pose_idx in range(n_poses):
        for frame_idx in range(FRAME_RATE):
            
            if pose_idx:
                theta = lerp(pose[pose_idx-1], pose[pose_idx], frame_idx/FRAME_RATE)
            else:
                theta = lerp(pose[-1], pose[pose_idx], frame_idx/FRAME_RATE)
            
            if MODE == "Rigid":
                rigid_bone_locations = test_skeleton.pose_bones(theta, trans, degrees=DEGREES, exclude_root=EXCLUDE_ROOT)
                skel_mesh.points = rigid_bone_locations
            else:
                simulated_bone_locations = helper_rig.update(theta, trans, degrees=DEGREES, exclude_root=EXCLUDE_ROOT)
                skel_mesh.points = simulated_bone_locations
    
            # Write a frame. This triggers a render.
            plotter.write_frame()

# Closes and finalizes movie
plotter.close()
plotter.deep_clean()