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

def add_helper_bones(test_skeleton, 
                     helper_bone_endpoints, 
                     helper_bone_parents,
                     offset_ratio=0.0, 
                     startpoints=[]):
    
    n_helper = len(helper_bone_parents)
    if len(startpoints)==0: startpoints = np.repeat([None],n_helper)
    
    helper_idxs = []
    for i in range(n_helper):
        bone_idx = test_skeleton.insert_bone( endpoint = helper_bone_endpoints[i],
                                              parent_idx = helper_bone_parents[i],
                                              offset_ratio = offset_ratio,
                                              startpoint = startpoints[i]
                                              )
        helper_idxs.append(bone_idx)
    return helper_idxs

def lerp(arr1, arr2, ratio):
    # TODO: Please make it more robust? Like asserting array shapes etc...
    return ((1.0 - ratio) * arr1) + (ratio * arr2)

# ---------------------------------------------------------------------------- 
# Set skeletal animation data
# ---------------------------------------------------------------------------- 
TGF_PATH = IGL_DATA_PATH + "arm.tgf"
joint_locations, kintree, _, _, _, _ = igl.read_tgf(TGF_PATH)

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
                ],
                [
                 [0.,0.,0.],
                 [0.,0.,0.],
                 [0., 0., 0.],
                 [0.,0.,0.],
                 [0.,0.,0.],
                 [0.,0.,0.],
                ],
                ])

MODE = "Dynamic" #"Rigid" or "Dynamic"

FIXED_SCALE = False
POINT_SPRING = True 
EXCLUDE_ROOT = True
DEGREES = True # Set true if pose is represented with degrees as Euler angles.

N_REPEAT = 10
FRAME_RATE = 60 #24
TIME_STEP = 1./FRAME_RATE  

MASS = 1.
STIFFNESS = 200.
MASS_DSCALE = 0.1        # Range [0.0, 1.0] Scales mass velocity
SPRING_DSCALE = 10.0      # Range [0.0, 1.0]
DAMPING = 50.            

# ---------------------------------------------------------------------------- 
# Create rig and set helper bones
# ---------------------------------------------------------------------------- 
PARENT_IDX = 2
helper_bone_endpoints = np.array([ joint_locations[PARENT_IDX] + [0.0, 0.2, 0.0] ])
helper_bone_parents = [PARENT_IDX]

test_skeleton = create_skeleton(joint_locations, kintree)
helper_idxs = add_helper_bones(test_skeleton, helper_bone_endpoints, 
                                     helper_bone_parents,
                                     offset_ratio=0.5,
                                     #startpoints=helper_bone_endpoints-1e-6
                                     )

helper_rig = HelperBonesHandler(test_skeleton, 
                                helper_idxs,
                                mass          = MASS, 
                                stiffness     = STIFFNESS,
                                damping       = DAMPING,
                                mass_dscale   = MASS_DSCALE,
                                spring_dscale = SPRING_DSCALE,
                                dt            = TIME_STEP,
                                point_spring  = POINT_SPRING,
                                fixed_scale   = FIXED_SCALE) 

# TODO: you could also add insert_point_handle() to Skeleton class
# that creates a zero-length bone (we need to render bone tips as spheres to see that)

# ---------------------------------------------------------------------------- 
# Create plotter 
# ---------------------------------------------------------------------------- 
RENDER = True
plotter = pv.Plotter(notebook=False, off_screen=not RENDER)
plotter.camera_position = 'zy'
plotter.camera.azimuth = 90
plotter.camera.view_angle = 90 # This works like zoom actually

# ---------------------------------------------------------------------------- 
# Add skeleton mesh based on T-pose locations
# ---------------------------------------------------------------------------- 
n_bones = len(test_skeleton.rest_bones)
rest_bone_locations = test_skeleton.get_rest_bone_locations(exclude_root=EXCLUDE_ROOT)
line_segments = np.reshape(np.arange(0, 2*(n_bones-1)), (n_bones-1, 2))
# TODO: rename get_rest_bone_locations() to get_rest_bones() that will also return
# line_segments based on exclude_root variable
# (note that you need to re-run other skeleton tests)

skel_mesh = add_skeleton(plotter, rest_bone_locations, line_segments)
plotter.open_movie(RESULT_PATH + "/igl-skeleton.mp4")

n_poses = pose.shape[0]
trans = None # TODO: No relative translation yet...

try:
    for rep in range(N_REPEAT):
        init_locations = helper_rig.init_pose(theta=pose[0], trans=trans, degrees=DEGREES)
        for pose_idx in range(n_poses):
            for frame_idx in range(FRAME_RATE):
                
                if pose_idx: # If not the first pose
                    theta = lerp(pose[pose_idx-1], pose[pose_idx], frame_idx/FRAME_RATE)
                else:        # Lerp with the last pose for boomerang
                    theta = lerp(pose[pose_idx], pose[-1], frame_idx/FRAME_RATE)
                    
                    if rep==0 and frame_idx==0 and rep==0:
                        rigid_locations = test_skeleton.pose_bones(theta, trans, degrees=DEGREES, exclude_root=False)
                        assert np.linalg.norm(init_locations - rigid_locations) < 1e-20, "ERROR: Initial pose does not match with helper bone's initial settings. Please fix it first in order not to run into stabilization errors."
                 
                if MODE == "Rigid":
                    rigid_bone_locations = test_skeleton.pose_bones(theta, trans, degrees=DEGREES, exclude_root=EXCLUDE_ROOT)
                    skel_mesh.points = rigid_bone_locations
                else:
                    simulated_bone_locations = helper_rig.update(theta, trans, degrees=DEGREES, exclude_root=EXCLUDE_ROOT)
                    skel_mesh.points = simulated_bone_locations
        
                # Write a frame. This triggers a render.
                plotter.write_frame()
except AssertionError:
    print(">>>> Caught assertion, stopping execution...")
    plotter.close()
    raise
    
# Closes and finalizes movie
plotter.close()
plotter.deep_clean()