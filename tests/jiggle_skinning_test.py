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
import skinning
from data import poses
from linalg_utils import lerp
from helper_rig import HelperBonesHandler
from global_vars import IGL_DATA_PATH, RESULT_PATH
from pyvista_render_tools import add_skeleton, add_mesh
from skeleton import Skeleton, create_skeleton, add_helper_bones

# ---------------------------------------------------------------------------- 
# Set skeletal animation data (TODO: Can we do it in another script and retrieve the data with 1-2 lines?)
# ---------------------------------------------------------------------------- 
TGF_PATH = IGL_DATA_PATH + "arm.tgf"
OBJ_PATH = IGL_DATA_PATH + "arm.obj"
DMAT_PATH = IGL_DATA_PATH + "arm-weights.dmat"
joint_locations, kintree, _, _, _, _ = igl.read_tgf(TGF_PATH)
arm_verts_rest, _, _, arm_faces, _, _ =  igl.read_obj(OBJ_PATH)
pose = poses.igl_arm_pose

# ----------------------------------------------------------------------------
# Declare parameters
# ----------------------------------------------------------------------------
MODE = "Rigid" #"Rigid" or "Dynamic" TODO: could you use more robust way to set it?
FIXED_SCALE = False # Set true if you want the jiggle bone to preserve its length
POINT_SPRING = True # Set true for less jiggling (point spring at the tip), set False to jiggle the whole bone as a spring.
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
# TODO: This wasn't the way we supposed to add helpers. Can we change it to a single call?
another_helper_idxs2 = add_helper_bones(test_skeleton,
                                       helper_bone_endpoints * 2, 
                                       helper_bone_parents = another_helper_idxs,
                                       offset_ratio=0.0,
                                       )
# TODO: Again, can we change the adding of the helpers a single call by declaring
#  the additional data manually in another script?
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

# TODO: you could also add insert_point_handle() to Skeleton class
# that creates a zero-length bone (we need to render bone tips as spheres to see that)

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
# TODO: rename get_rest_bone_locations() to get_rest_bones() that will also return
# line_segments based on exclude_root variable
# (note that you need to re-run other skeleton tests)
n_bones = len(test_skeleton.rest_bones)
rest_bone_locations = test_skeleton.get_rest_bone_locations(exclude_root=EXCLUDE_ROOT)
line_segments = np.reshape(np.arange(0, 2*(n_bones-1)), (n_bones-1, 2))

skel_mesh = add_skeleton(plotter, rest_bone_locations, line_segments)
arm_mesh = add_mesh(plotter, arm_verts_rest, arm_faces, opacity=OPACITY)

# ---------------------------------------------------------------------------- 
# Bind T-pose vertices to unposed skeleton
# ---------------------------------------------------------------------------- 
if MODE == "Rigid":
    weights = igl.read_dmat(DMAT_PATH)
    # Add zero weights for the helper bone 
    # TODO: could you just not use the helper bones for rigid animation?
    n_verts, n_orig_bones = weights.shape
    n_helpers = n_bones - n_orig_bones - 1 # TODO: Minus one is for the invisible root bone, we better remove this feature in order not to complicatte things 
    zero_weights = np.zeros((n_verts, n_helpers))
    weights = np.append(weights, zero_weights, axis=-1)
elif MODE == "Dynamic":
    weights = skinning.bind_weights(arm_verts_rest, rest_bone_locations)
else:
    print(f">> ERROR: Unexpected skinning mode {MODE}")
    raise ValueError
    
# ---------------------------------------------------------------------------------
# Render Loop
# ---------------------------------------------------------------------------------
plotter.open_movie(RESULT_PATH + f"/helper-jiggle-m{MASS}-k{STIFFNESS}-kd{DAMPING}-mds{MASS_DSCALE}-sds{SPRING_DSCALE}-fixedscale-{FIXED_SCALE}-pointspring-{POINT_SPRING}.mp4")
n_poses = pose.shape[0]
trans = np.zeros((n_bones, 3))
try:
    # TODO: refactor this render loop such that we don't have to check N_REST from the
    # inside, rather we'll just render static image once we're done with animation
    # (or we can insert more rest poses via editing the pose array)
    # TODO: refactor the lerp() to be out of render loop such that poses will be precomputed
    # for each frame, we'll only retrieve the current pose daha and skinning data for rendering
    # these two refactoring should remove a conditional and a loop 
    for rep in range(N_REPEAT):         # This can be refactored too as it's not related to render
        for pose_idx in range(n_poses): # Loop keyframes, this could be refactored.
            for frame_idx in range(FRAME_RATE):
                
                if rep < N_REST:  
                    if pose_idx: # If not the first pose
                        theta = lerp(pose[pose_idx-1], pose[pose_idx], frame_idx/FRAME_RATE)
                    else:        # Lerp with the last pose for boomerang
                        theta = lerp(pose[pose_idx], pose[-1], frame_idx/FRAME_RATE)
                
                    if MODE == "Rigid": skeleton = test_skeleton
                    else: skeleton = helper_rig
                
                # TODO: we're repeating ourselves, could we separate pose bones and abs T parts?
                abs_rot_quat, abs_trans = skeleton.get_absolute_transformations(theta, trans, DEGREES)
                abs_rot_quat = abs_rot_quat[1:] # TODO: get rid of root bone convention
                abs_trans = abs_trans[1:]       # TODO: get rid of root bone convention
                mesh_points = skinning.skinning(arm_verts_rest, abs_rot_quat, abs_trans, weights, skinning_type="LBS")
                arm_mesh.points = mesh_points
                
                skel_mesh_points = skinning.get_skel_points(skeleton, theta, trans, degrees=DEGREES, exclude_root=EXCLUDE_ROOT, combine_points=True)
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