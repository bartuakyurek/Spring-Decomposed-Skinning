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
from src.utils.linalg_utils import lerp
from src.helper_handler import HelperBonesHandler
from src.global_vars import IGL_DATA_PATH, RESULT_PATH
from src.render.pyvista_render_tools import add_skeleton, add_mesh
from src.skeleton import Skeleton, create_skeleton, add_helper_bones

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
                 [0.,0.,0.],
                 [0.,0.,0.],
                ],
                ])

MODE = "Dynamic " #"Rigid" or "Dynamic"

FIXED_SCALE = False # Set true if you want the jiggle bone to preserve its length
POINT_SPRING = False # Set true for less jiggling (point spring at the tip), set False to jiggle the whole bone as a spring.
EXCLUDE_ROOT = True # Set true in order not to render the invisible root bone (it's attached to origin)
DEGREES = True # Set true if pose is represented with degrees as Euler angles.

N_REPEAT = 10
N_REST = N_REPEAT - 5
FRAME_RATE = 24 #24
TIME_STEP = 1./FRAME_RATE  

MASS = 1.
STIFFNESS = 300.
DAMPING = 50.            
MASS_DSCALE = 0.4       # Scales mass velocity (Use [0.0, 1.0] range to slow down)
SPRING_DSCALE = 1.0     # Scales spring forces (increase for more jiggling)

# ---------------------------------------------------------------------------- 
# Create rig and set helper bones
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

n_poses = pose.shape[0]
trans = None 

# ---------------------------------------------------------------------------------
# Helper routine to obtain posed mesh vertices
# ---------------------------------------------------------------------------------
def _get_skel_points(mode, combine_points=True):
    if mode == "Rigid":
        rigid_bone_locations = test_skeleton.pose_bones(theta, trans, degrees=DEGREES, exclude_root=EXCLUDE_ROOT)
        skel_mesh_points = rigid_bone_locations
    else:
        simulated_bone_locations = helper_rig.pose_bones(theta, trans, degrees=DEGREES, exclude_root=EXCLUDE_ROOT)
        skel_mesh_points = simulated_bone_locations
    
    if combine_points:
        skel_mesh_points = np.reshape(skel_mesh_points, (-1,3)) # Combine all the 3D points into one dimension
    return skel_mesh_points

# ---------------------------------------------------------------------------------
# Render Loop
# ---------------------------------------------------------------------------------
plotter.open_movie(RESULT_PATH + f"/helper-jiggle-m{MASS}-k{STIFFNESS}-kd{DAMPING}-mds{MASS_DSCALE}-sds{SPRING_DSCALE}-fixedscale-{FIXED_SCALE}-pointspring-{POINT_SPRING}.mp4")
try:
    for rep in range(N_REPEAT):
        for pose_idx in range(n_poses):
            for frame_idx in range(FRAME_RATE):
                
                if rep < N_REST:  
                    if pose_idx: # If not the first pose
                        theta = lerp(pose[pose_idx-1], pose[pose_idx], frame_idx/FRAME_RATE)
                    else:        # Lerp with the last pose for boomerang
                        theta = lerp(pose[pose_idx], pose[-1], frame_idx/FRAME_RATE)
                         
                skel_mesh_points = _get_skel_points(MODE, combine_points=True)
                assert skel_mesh_points.shape == ( (n_bones-EXCLUDE_ROOT) * 2, 3)
                
                skel_mesh.points = skel_mesh_points # Update mesh points in the renderer.
                plotter.write_frame()          # Write a frame. This triggers a render.
except AssertionError:
    print(">>>> Caught assertion, stopping execution...")
    plotter.close()
    raise
    
# ---------------------------------------------------------------------------------
# Quit the renderer and close the movie.
# ---------------------------------------------------------------------------------
plotter.close()
plotter.deep_clean()