#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is created to tranfer jiggling bones' transformations to the 
surface via skinning (LBS by default).


Note: You first need to interact with the plotter via mouse click before
you can call key press events.

Available key presses
---------------------
B : Select a bone and show its binding colors on the surface


Created on Thu Oct 10 14:34:34 2024
@author: bartu
"""
# TODO: you can sanity check your weights by _assert_normalized_weights() here as well
# but you may not want normalized weights for entire skeleton...
# TODO: we could put a scalar to tune the weights of the helper bones weights
#       especially if we're computing weights separately for helper bones.
# Note that since we're aiming a framework to be on top of the existing animation
# we don't want to change the existing rig's weights anyway. So it's better to
# keep the weights separate even if they aren't add up to 1.0

import igl
import numpy as np
import pyvista as pv

import __init__
from src.data import poses
from src import skinning
from src.utils.linalg_utils import lerp
from src.kinematics import inverse_kinematics
from src.helper_handler import HelperBonesHandler
from src.global_vars import IGL_DATA_PATH, RESULT_PATH
from src.render.pyvista_render_tools import add_skeleton, add_mesh
from src.skeleton import Skeleton, create_skeleton, add_helper_bones
from src.render.pyvista_render_tools import (add_mesh, 
                                             add_skeleton, 
                                             set_mesh_color_scalars,
                                             set_mesh_color)

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
MODE = "Dynamic" #"Rigid" or "Dynamic" TODO: could you use more robust way to set it?
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

ENVELOPE = 10. # For weights

# ---------------------------------------------------------------------------- 
# Create rig and add helper bones
# ---------------------------------------------------------------------------- 
test_skeleton = create_skeleton(joint_locations, kintree)

helper_bone_endpoints = np.array([[ 0.1016,  0.481821, -0.31808 ],
                                  [ 0.1086,  0.4821, -0.0808 ],
                                  [ 0.1637,  0.39714, -0.08928]])
helper_bone_parents = [2, 5, 6]

helper_idxs = add_helper_bones(test_skeleton, 
                               helper_bone_endpoints, 
                               helper_bone_parents,
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
OPACITY = 1.0
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
arm_mesh, arm_mesh_actor = add_mesh(plotter, arm_verts_rest, arm_faces, opacity=OPACITY, return_actor=True)

# ---------------------------------------------------------------------------- 
# Bind T-pose vertices to unposed skeleton
# ---------------------------------------------------------------------------- 
weights = igl.read_dmat(DMAT_PATH)     # Read the weights for existing rigid rig
n_verts, n_orig_bones = weights.shape
n_helpers = n_bones - n_orig_bones - 1 # TODO: Minus one is for the invisible root bone, 
                                       #       we better remove this feature in order not to complicatte things 
if MODE == "Rigid":
    helper_weights = np.zeros((n_verts, n_helpers))
elif MODE == "Dynamic":
    #helper_weights = skinning.bind_weights(arm_verts_rest, helper_bone_rest_locations, envelope=ENVELOPE)
    helper_weights = np.zeros((n_verts, n_helpers))
    print(">> WARNING: helper weights are initialized to zero.")
    
    # Insert loaded bone weights to the helper bone weights for a single bone
    # This is done manually for now, it should be automatized....
    with np.load("../data/single-bone-weights-0.npz") as data:
        w = data["arr_0"]
        idxs = data["arr_1"]
    single_bone_weights = np.zeros((n_verts))
    single_bone_weights[idxs] = w
    SELECTED_HELPER = 1 # 0 1 2
    helper_weights[:,SELECTED_HELPER] = single_bone_weights
    # end of insert bone weights
else:
    print(f">> ERROR: Unexpected skinning mode {MODE}")
    raise ValueError
    
assert helper_weights.shape == (n_verts, n_helpers), f"Helper bones weights are expected to have shape {(n_verts, n_helpers)}, got {helper_weights.shape}."
weights = np.append(weights, helper_weights, axis=-1)

# ---------------------------------------------------------------------------------
# Set up Key Press Actions
# ---------------------------------------------------------------------------------
# When "B" key is pressed, show colors of the corresponding bone's weights
selected_bone_idx = -1
def change_colors():
    global selected_bone_idx
    global n_bones
    selected_bone_idx += 1
    if selected_bone_idx >= n_bones-1: # TODO: remove -1 when you get rid of root bone
        selected_bone_idx = -1
    
    if selected_bone_idx >= 0:
        print("INFO: Selected bone ", selected_bone_idx)
        print(">> Call set mesh colors...")
        selected_weights = weights[:,selected_bone_idx]
        set_mesh_color_scalars(arm_mesh, selected_weights)  

def deselect_bone():
    global selected_bone_idx
    selected_bone_idx = -1
    set_mesh_color(arm_mesh, [0.8, 0.8, 1.0])
    print(">> INFO: Bone deselected.")
    return

plotter.add_key_event("B", change_colors)
plotter.add_key_event("b", change_colors)
plotter.add_key_event("N", deselect_bone)
plotter.add_key_event("n", deselect_bone)
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
                
                posed_locations = test_skeleton.pose_bones(theta, trans, degrees=DEGREES)
                if MODE=="Rigid":
                    posed_locations = np.reshape(posed_locations, (-1,3)) # Combine all the 3D points into one dimension
                    skel_mesh_points = posed_locations[2:] # TODO: get rid of root bone convention

                    abs_rot_quat, abs_trans = test_skeleton.get_absolute_transformations(theta, trans, degrees=DEGREES)
                    mesh_points = skinning.LBS_from_quat(arm_verts_rest, weights, abs_rot_quat[1:], abs_trans[1:]) # TODO: get rid of root
                else:
                    
                    posed_locations = helper_rig.update_bones(posed_locations) # Update the rigidly posed locations
                    posed_locations = np.reshape(posed_locations, (-1,3)) # Combine all the 3D points into one dimension
                    
                    skel_mesh_points = posed_locations[2:] # TODO: get rid of root bone convention
                                                           # TODO: directly set skel_mesh.points = posed
                    # TODO: keep getting transforms from rigid skeleton, only update the helpers' transforms.
                    #abs_rot_quat, abs_trans = test_skeleton.get_absolute_transformations(theta, trans, degrees=DEGREES)
                    #M = helper_rig.get_absolute_transformations(posed_locations, return_mat=True, algorithm="RST")
                    rest_bone_locations = test_skeleton.get_rest_bone_locations(exclude_root=False) # TODO: Remove this line from here
                    M = inverse_kinematics.get_absolute_transformations(rest_bone_locations, posed_locations, return_mat=True, algorithm="RST")
                    mesh_points = skinning.LBS_from_mat(arm_verts_rest, weights, M[1:]) # TODO: get rid of root
                              
                # Set data for renderer
                arm_mesh.points = mesh_points
                skel_mesh.points = skel_mesh_points # Update mesh points in the renderer.
                plotter.write_frame()               # Write a frame. This triggers a render.
except AssertionError:
    print(">>>> Caught assertion, stopping execution...")
    plotter.close()
    raise
    
# ---------------------------------------------------------------------------------
# Quit the renderer and close the movie.
# ---------------------------------------------------------------------------------
plotter.close()
plotter.deep_clean()