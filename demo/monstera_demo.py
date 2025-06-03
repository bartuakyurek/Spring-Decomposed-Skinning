#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is created to demonstrate dynamic animation of a plant pot.

DEPRECATED, use lbs_comparison_demo.py

Created on Thu Oct 10 14:34:34 2024
@author: bartu
"""


import igl
import numpy as np
import pyvista as pv

import __init__
from src.data import poses
from src import skinning
from src.utils.linalg_utils import lerp
from src.kinematics import inverse_kinematics
from src.helper_handler import HelperBonesHandler
from src.global_vars import DATA_PATH, RESULT_PATH
from src.skeleton import Skeleton, create_skeleton_from
from src.render.pyvista_render_tools import (add_mesh, 
                                             add_skeleton_from_Skeleton, 
                                             set_mesh_color_scalars,
                                             set_mesh_color)

# ----------------------------------------------------------------------------
# Declare parameters
# ----------------------------------------------------------------------------
MODE = "Dynamic" #"Rigid" or "Dynamic" 
INTEGRATION = "PBD" # PBD or Euler
ALGO = "RST" # RST, SVD, T
NORMALIZE_WEIGHTS = True

OPACITY = 1.0
SPRING_BONE_COLOR = "blue"
WINDOW_SIZE = (1200, 1200)
RENDER_MESH = True
RENDER_SKEL = False
RENDER_PHYS_BASED = True
MATERIAL_METALLIC = 0.2
MATERIAL_ROUGHNESS = 0.5

FIXED_SCALE = True # Set true if you want the jiggle bone to preserve its length
POINT_SPRING = True # Set true for less jiggling (point spring at the tip), set False to jiggle the whole bone as a spring.
EXCLUDE_ROOT = True # Set true in order not to render the invisible root bone (it's attached to origin)
DEGREES = True # Set true if pose is represented with degrees as Euler angles.
N_REPEAT = 2
N_REST = 1

FRAME_RATE = 24 # 24, 30, 60
TIME_STEP = 1./FRAME_RATE  
MASS = 3
STIFFNESS = 150.
DAMPING = 10.            
MASS_DSCALE = 0.6       # Scales mass velocity (Use [0.0, 1.0] range to slow down)
SPRING_DSCALE = 1.0     # Scales spring forces (increase for more jiggling)

FNAME = "monstera"      # For i/o files
DATA_PATH = DATA_PATH + FNAME + "/"
OBJ_PATH = DATA_PATH + FNAME + ".obj"
MONSTERA_RIG_PATH = DATA_PATH + FNAME + "_rig_data.npz"

# ---------------------------------------------------------------------------- 
# Set skeletal animation data 
# ---------------------------------------------------------------------------- 
V_rest, _, _, F, _, _ =  igl.read_obj(OBJ_PATH)
keyframe_poses = poses.monstera_rig_pose

with np.load(MONSTERA_RIG_PATH) as data:
     W = data["weights"] # (n_verts, n_bones)
     B = data["joints"] # (n_bones, 2, 3)
     kintree = data["kintree"] # (n_bones, 2)

# ---- Rotate, translate, scale the rig data if needed -------

# Rotate 
from scipy.spatial.transform import Rotation
r = Rotation.from_euler('x', -90, degrees=True)
for i in range(len(B)):   
    B[i] = r.apply(B[i])
    
# Translate to origin
B = B - np.array([0, 0, 0.6])

# Scale
B = B * 0.37

# ---------------------------------------------------------------------------- 
# Define helper spring bones on the existing skeleton
# ---------------------------------------------------------------------------- 
n_bones = len(kintree)
skeleton = create_skeleton_from(B, kintree)
helper_idxs = np.array([i for i in range(1, n_bones)])

helper_rig = HelperBonesHandler(skeleton, 
                                helper_idxs,
                                mass          = MASS, 
                                stiffness     = STIFFNESS,
                                damping       = DAMPING,
                                mass_dscale   = MASS_DSCALE,
                                spring_dscale = SPRING_DSCALE,
                                dt            = TIME_STEP,
                                point_spring  = POINT_SPRING,
                                fixed_scale   = FIXED_SCALE, 
                                simulation_mode = INTEGRATION) 

# ---------------------------------------------------------------------------- 
# Create plotter 
# ---------------------------------------------------------------------------- 
RENDER = True
plotter = pv.Plotter(notebook=False, off_screen=not RENDER, window_size = WINDOW_SIZE)
plotter.camera_position = 'zy'
plotter.camera.position = [-8.0, 2.0, 0]
plotter.camera.view_angle = 20 # This works like zoom actually
plotter.camera.focal_point = (0.0, 0.65, 0.0)

# Add light
light = pv.Light(position=(0, 1.5, 1.5), light_type='scene light')
plotter.add_light(light)

# Add frame number
TEXT_POSITION = (WINDOW_SIZE[0]-70, WINDOW_SIZE[1]-70)
frame_text_actor = plotter.add_text("0", TEXT_POSITION, font_size=18)

# ---------------------------------------------------------------------------- 
# Add mesh actors
# ----------------------------------------------------------------------------

if RENDER_SKEL:    
    skel_mesh = add_skeleton_from_Skeleton(plotter, skeleton, helper_idxs, alt_bone_color=SPRING_BONE_COLOR)

if RENDER_MESH: 
    
    mesh, mesh_actor = add_mesh(plotter, V_rest, F, opacity=OPACITY, return_actor=True,
                                pbr=RENDER_PHYS_BASED, metallic=MATERIAL_METALLIC, roughness=MATERIAL_ROUGHNESS)
    
    #texture = pv.read_texture(DATA_PATH + "texture/1.png") 
    #mesh.active_texture_coordinates = tc
    #mesh.texture_map_to_plane(inplace=True)
    #mesh_actor.texture = texture

plotter.open_movie(RESULT_PATH + f"/{FNAME}-{ALGO}-{MODE}.mp4")
n_poses = keyframe_poses.shape[0]
trans = np.zeros((len(skeleton.rest_bones), 3)) # TODO: remove +1 when you remove root bone issue

# ---------------------------------------------------------------------------------
# Simulate and save data
# ---------------------------------------------------------------------------------
V_anim = []
J_anim = []
assert n_poses > 1, f"Expected keyframe poses to be at least 2. Got {n_poses} poses."
for rep in range(N_REPEAT + N_REST):         # This can be refactored too as it's not related to render
    for pose_idx in range(n_poses-1): # Loop keyframes, this could be refactored.
        for frame_idx in range(FRAME_RATE):
            
            if rep < N_REPEAT:
                # Lerp with next frame
                theta = lerp(keyframe_poses[pose_idx], keyframe_poses[pose_idx+1], frame_idx/FRAME_RATE)
              
            posed_locations = skeleton.pose_bones(theta, trans, degrees=DEGREES)
            abs_rot_quat, abs_trans = skeleton.get_absolute_transformations(theta, trans, degrees=DEGREES)
            M_rigid = skinning.get_transform_mats_from_quat_rots(abs_trans, abs_rot_quat)[1:] # TODO...
                
            if MODE=="Rigid":
                skel_mesh_points = posed_locations[2:] # TODO: get rid of root bone convention
                if RENDER_MESH: mesh_points = skinning.LBS_from_quat(V_rest, W, abs_rot_quat[1:], abs_trans[1:], use_normalized_weights=NORMALIZE_WEIGHTS) # TODO: get rid of root
            else:
                posed_locations = helper_rig.update_bones(posed_locations) # Update the rigidly posed locations
                skel_mesh_points = posed_locations[2:] # TODO: get rid of root bone convention
                   
                rest_bone_locations = skeleton.get_rest_bone_locations(exclude_root=False) # TODO: Remove this line from here
                M = inverse_kinematics.get_absolute_transformations(rest_bone_locations, posed_locations, return_mat=True, algorithm=ALGO)[1:]  # TODO: get rid of root
                    
                M_hybrid = M_rigid
                M_hybrid[helper_idxs] = M[helper_idxs]
                if RENDER_MESH: mesh_points = skinning.LBS_from_mat(V_rest, W, M_hybrid, use_normalized_weights=NORMALIZE_WEIGHTS)
                       
            
            if RENDER_MESH: V_anim.append(mesh_points)
            if RENDER_SKEL: J_anim.append(skel_mesh_points)

    

# ---------------------------------------------------------------------------------
# Show computed results
# ---------------------------------------------------------------------------------

n_frames = max(len(V_anim), len(J_anim))
for frame in range(n_frames):
    # Set data for renderer
    if RENDER_MESH: mesh.points = V_anim[frame]
    if RENDER_SKEL: skel_mesh.points = J_anim[frame] 
    
    frame_text_actor.input = str(frame+1)
    plotter.write_frame()   # Write a frame. This triggers a render.
    

plotter.close()
plotter.deep_clean()



