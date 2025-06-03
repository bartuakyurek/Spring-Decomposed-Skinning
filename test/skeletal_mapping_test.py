#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is created to test the computed optimal rigid motions. Given two sets 
of points, optimal rigid motion maps the first source set to the second target 
set with optimal rotation and translations.

In our case, we're dealing with line segments. We like to find the optimal rigid
motion between two line segments. However, since the optimal rigid motion algorithm
we're dealing with doesn't work correctly on 2 points sets, we'll be providing an
extra middle point from these bone line segments. There are also other ways to do
it however as I observed previously, the rotation obtained from these algorithms
can fail in Dual Quaternion Skinning. (Though I don't know if this implementation
                                       will work with DQS yet.)

For our skinning purposes, we have source set coming from T-pose skeleton and 
target set that is the posed skeleton. Every bone in the skeleton constitutes 
from 2 joints. We like to find the mapping for each bone that can take the bone
from T-pose and locate it to its posed locations.

Note that if you set FIXED_SCALE = False, the bones will be disconnected because
transformations do not handle scaling of the bones yet. This effect is best observed
on POINT_SPRINGS = False case, i.e. when the whole bone is a spring.

Created on Thu Oct 10 14:34:34 2024
@author: bartu
"""
import igl
import numpy as np
import pyvista as pv

import __init__
from src import skinning
from data import poses
from src.utils.linalg_utils import lerp
from src.helper_handler import HelperBonesHandler
from src.global_vars import IGL_DATA_PATH, RESULT_PATH
from src.render.pyvista_render_tools import add_skeleton
from src.skeleton import Skeleton, create_skeleton, add_helper_bones

# ---------------------------------------------------------------------------- 
# Set skeletal animation data 
# ---------------------------------------------------------------------------- 
TGF_PATH = IGL_DATA_PATH + "arm.tgf"
joint_locations, kintree, _, _, _, _ = igl.read_tgf(TGF_PATH)
pose = poses.igl_arm_pose

# ----------------------------------------------------------------------------
# Declare parameters
# ----------------------------------------------------------------------------
MODE = "Dynamic" #"Rigid" or "Dynamic" 
FIXED_SCALE = True # Set true if you want the jiggle bone to preserve its length 
POINT_SPRING = False # Set true for less jiggling (point spring at the tip), set False to jiggle the whole bone as a spring.
EXCLUDE_ROOT = True # Set true in order not to render the invisible root bone (it's attached to origin)
DEGREES = True # Set true if pose is represented with degrees as Euler angles.
N_REPEAT = 10
FRAME_RATE = 24 #24
TIME_STEP = 1./FRAME_RATE  
MASS = 1.
STIFFNESS = 300.
DAMPING = 50.            
MASS_DSCALE = 0.4       # Scales mass velocity (Use [0.0, 1.0] range to slow down)
SPRING_DSCALE = 1.0     # Scales spring forces (increase for more jiggling)

# ---------------------------------------------------------------------------- 
# Create rig and add helper bones
# ---------------------------------------------------------------------------- 
PARENT_IDX = 2
helper_bone_endpoints = np.array([ joint_locations[PARENT_IDX] + [0.0, 0.2, 0.0] ])
helper_bone_parents = [PARENT_IDX]

test_skeleton = create_skeleton(joint_locations, kintree)
helper_idxs = add_helper_bones(test_skeleton, 
                               helper_bone_endpoints, 
                               helper_bone_parents,
                               offset_ratio=0.5,
                               #startpoints=helper_bone_endpoints-1e-6
                               )

# Add helpers to the tip of the previously added helpers 
another_helper_idxs = add_helper_bones(test_skeleton,
                                       helper_bone_endpoints * 1.25, 
                                       helper_bone_parents = helper_idxs,
                                       offset_ratio=0.0,
                                       )

another_helper_idxs2 = add_helper_bones(test_skeleton,
                                       helper_bone_endpoints * 2, 
                                       helper_bone_parents = another_helper_idxs,
                                       offset_ratio=0.0,
                                       )

all_helper_idxs = helper_idxs + another_helper_idxs + another_helper_idxs2
helper_rig = HelperBonesHandler(test_skeleton, 
                                all_helper_idxs,
                                mass          = MASS, 
                                stiffness     = STIFFNESS,
                                damping       = DAMPING,
                                mass_dscale   = MASS_DSCALE,
                                spring_dscale = SPRING_DSCALE,
                                dt            = TIME_STEP,
                                point_spring  = POINT_SPRING,
                                fixed_scale   = FIXED_SCALE) 

# ---------------------------------------------------------------------------- 
# Create plotter 
# ---------------------------------------------------------------------------- 
RENDER = True
OPACITY = 0.5
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
skel_mesh = add_skeleton(plotter, rest_bone_locations, line_segments)
 
# ---------------------------------------------------------------------------------
# Render Loop
# ---------------------------------------------------------------------------------
plotter.open_movie(RESULT_PATH + f"/helper-jiggle-m{MASS}-k{STIFFNESS}-kd{DAMPING}-mds{MASS_DSCALE}-sds{SPRING_DSCALE}-fixedscale-{FIXED_SCALE}-pointspring-{POINT_SPRING}.mp4")
n_poses = pose.shape[0]
trans = np.zeros((n_bones, 3))

def render_loop():
    for rep in range(N_REPEAT):         # This can be refactored too as it's not related to render
        for pose_idx in range(n_poses): # Loop keyframes, this could be refactored.
            for frame_idx in range(FRAME_RATE):
                
                if pose_idx: # If not the first pose
                        theta = lerp(pose[pose_idx-1], pose[pose_idx], frame_idx/FRAME_RATE)
                else:        # Lerp with the last pose for boomerang
                        theta = lerp(pose[pose_idx], pose[-1], frame_idx/FRAME_RATE)
                      
                if MODE=="Rigid":
                    posed_locations = skinning.get_skel_points(test_skeleton, theta, trans, degrees=DEGREES, exclude_root=False, combine_points=True)
                else:
                    posed_locations = skinning.get_skel_points(helper_rig, theta, trans, degrees=DEGREES, exclude_root=False, combine_points=True)
               
                # WARNING: If there's scaling in the bones, these transformations will not handle it! 
                # So it works under FIXED_SCALE = True case.
                abs_rot_quat, abs_trans = helper_rig.get_absolute_transformations(posed_locations)
                estimated_locations = test_skeleton.compute_bone_locations(abs_rot_quat, abs_trans)
                diff = np.linalg.norm(estimated_locations - posed_locations)
                assert diff < 1e-12, f"Expected difference to be less than 1e-12, got {diff}."
               
                skel_mesh_points = estimated_locations[2:] # exclude root
                assert skel_mesh_points.shape == ( (n_bones-EXCLUDE_ROOT) * 2, 3)
                
                skel_mesh.points = skel_mesh_points # Update mesh points in the renderer.
                plotter.write_frame()          # Write a frame. This triggers a render.
                
try:
   render_loop()
except AssertionError:
    print(">>>> Caught assertion, stopping execution...")
    plotter.close()
    raise
    
# ---------------------------------------------------------------------------------
# Quit the renderer and close the movie.
# ---------------------------------------------------------------------------------
plotter.close()
plotter.deep_clean()