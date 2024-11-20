#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is created to demonstrate dynamic animation of a plant pot.

Created on Thu Oct 10 14:34:34 2024
@author: bartu
"""

import os
import igl
import numpy as np
import pyvista as pv

import __init__
from src.data import model_data
from src import skinning
from src.utils.linalg_utils import lerp
from src.global_vars import RESULT_PATH
from src.kinematics import inverse_kinematics
from src.helper_handler import HelperBonesHandler
from src.utils.linalg_utils import normalize_arr_np
from src.skeleton import create_skeleton_from
from src.render.pyvista_render_tools import (add_mesh, 
                                             add_texture,
                                             add_skeleton_from_Skeleton, 
                                             set_mesh_color_scalars,
                                             set_mesh_color)

# ----------------------------------------------------------------------------
# Declare parameters
# ----------------------------------------------------------------------------
MODEL_NAME = "monstera" # Available options: "duck", "blob", "cloth", "monstera"

RENDER_AS_GIF = False # If set to False, render as .mp4 (WARNING: GIF export is buggy, you might need to render several times, or turn off ADD_LIGHT)
RENDER_TEXTURE = False
COLOR_CODE = False # True if you want to visualize the distances between rigid and dynamic
WIREFRAME = False
RENDER_MESH = True
RENDER_SKEL = False
SMOOTH_SHADING = True # Automatically set True if RENDER_PHYS_BASED = True
RENDER_PHYS_BASED = False
OPACITY = 1.0
MATERIAL_METALLIC = 0.0
MATERIAL_ROUGHNESS = 0.2
WINDOW_SIZE = (1920, 1080)

ADD_LIGHT = True
LIGHT_INTENSITY = 0.3 # Between [0, 1]
LIGHT_POS = (10.5, 3.5, 3.5)
SPRING_BONE_COLOR =  "blue"

INTEGRATION = "PBD" # PBD or Euler
ALGO = "RST" # RST, SVD, T
NORMALIZE_WEIGHTS = True

FIXED_SCALE = True # Set true if you want the jiggle bone to preserve its length
EDGE_CONSTRAINT = False # Recommended to set either FIXED_SCALE or EDGE_CONSTRAINT True
POINT_SPRING = False # Set true for less jiggling (point spring at the tip), set False to jiggle the whole bone as a spring.
EXCLUDE_ROOT = True # Set true in order not to render the invisible root bone (it's attached to origin)
DEGREES = True # Set true if pose is represented with degrees as Euler angles.
N_REPEAT = 1
N_REST = 0

FRAME_RATE = 24 # 24, 30, 60
TIME_STEP = 1./FRAME_RATE  
MASS = 1.5
STIFFNESS = 100.
DAMPING = 20.            
MASS_DSCALE = 0.5       # Scales mass velocity (Use [0.0, 1.0] range to slow down)
SPRING_DSCALE = 1.0     # Scales spring forces (increase for more jiggling)

model_dict = model_data.model_dict[MODEL_NAME]
OBJ_PATH = model_dict["OBJ_PATH"]
RIG_PATH = model_dict["RIG_PATH"]
TEXTURE_PATH = model_dict["TEXTURE_PATH"]
helper_idxs = model_dict["helper_idxs"]
keyframe_poses = model_dict["keyframe_poses"]

# ---------------------------------------------------------------------------- 
# Set skeletal animation data 
# ---------------------------------------------------------------------------- 
V_rest, _, _, F, _, _ =  igl.read_obj(OBJ_PATH)

with np.load(RIG_PATH) as data: 
     W = data["weights"] # (n_verts, n_bones)
     B = data["joints"] # (n_bones, 2, 3)
     kintree = data["kintree"] # (n_bones, 2)

# Rotate, translate, scale the rig data if needed
B = model_data.adjust_rig(B, MODEL_NAME)

# ---------------------------------------------------------------------------- 
# Define helper spring bones on the existing skeleton
# ---------------------------------------------------------------------------- 
skeleton = create_skeleton_from(B, kintree)

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
                                edge_constraint = EDGE_CONSTRAINT,
                                simulation_mode = INTEGRATION) 

# ---------------------------------------------------------------------------- 
# Create plotter 
# ---------------------------------------------------------------------------- 
RENDER = True
plotter = pv.Plotter(notebook=False, off_screen=not RENDER, window_size = WINDOW_SIZE, border=False, shape = (1,2))

# Add light
if ADD_LIGHT:
    light = pv.Light(position=LIGHT_POS, light_type='headlight', intensity=LIGHT_INTENSITY)
    plotter.add_light(light)
# ---------------------------------------------------------------------------- 
# Add mesh actors
# ----------------------------------------------------------------------------
    
# ---------- First Plot ----------------
plotter.subplot(0, 0)
model_data.adjust_camera(plotter, MODEL_NAME)

plotter.add_text("Rigid Skinning (LBS)", "lower_left", font_size=18)
frame_text_actor = plotter.add_text("0", (600,0), font_size=18) # Add frame number


if RENDER_MESH: 
    mesh_rigid, mesh_rigid_actor = add_mesh(plotter, V_rest, F, opacity=OPACITY, return_actor=True, 
                                            show_edges=WIREFRAME,
                                            smooth_shading=SMOOTH_SHADING,
                                            pbr=RENDER_PHYS_BASED, metallic=MATERIAL_METALLIC, roughness=MATERIAL_ROUGHNESS)
  
    if RENDER_TEXTURE:
        add_texture(mesh_rigid, mesh_rigid_actor, TEXTURE_PATH)
        
if RENDER_SKEL: skel_mesh_rigid = add_skeleton_from_Skeleton(plotter, skeleton)

# ---------- Second Plot ---------------
plotter.subplot(0, 1)
model_data.adjust_camera(plotter, MODEL_NAME)

plotter.add_text("Dynamic Skinning (Ours)", "lower_left", font_size=18)

if RENDER_SKEL: skel_mesh_dyn = add_skeleton_from_Skeleton(plotter, skeleton, helper_idxs, alt_bone_color=SPRING_BONE_COLOR)

if RENDER_MESH: 
    mesh_dyn, mesh_dyn_actor = add_mesh(plotter, V_rest, F, opacity=OPACITY, return_actor=True, 
                                        show_edges=WIREFRAME, smooth_shading=SMOOTH_SHADING,
                                        pbr=RENDER_PHYS_BASED, metallic=MATERIAL_METALLIC, roughness=MATERIAL_ROUGHNESS)
    
    if RENDER_TEXTURE:
        add_texture(mesh_dyn, mesh_dyn_actor, TEXTURE_PATH)
        


n_poses = keyframe_poses.shape[0]
n_bones = len(skeleton.rest_bones)
trans = np.zeros((n_bones, 3)) # TODO: remove +1 when you remove root bone issue

# ---------------------------------------------------------------------------------
# Simulate and save data
# ---------------------------------------------------------------------------------
V_anim_rigid, V_anim_dyn = [], []
J_anim_rigid, J_anim_dyn = [], []
rest_bone_locations = skeleton.get_rest_bone_locations(exclude_root=False) 
assert n_poses > 1, f"Expected keyframe poses to be at least 2. Got {n_poses} poses."
for rep in range(N_REPEAT + N_REST):         # This can be refactored too as it's not related to render
    for pose_idx in range(n_poses-1): # Loop keyframes, this could be refactored.
        for frame_idx in range(FRAME_RATE):
            
            if rep < N_REPEAT:
                # Lerp with next frame
                theta = lerp(keyframe_poses[pose_idx], keyframe_poses[pose_idx+1], frame_idx/FRAME_RATE)
              
            rigidly_posed_locations = skeleton.pose_bones(theta, trans, degrees=DEGREES)
            abs_rot_quat, abs_trans = skeleton.get_absolute_transformations(theta, trans, degrees=DEGREES)
            M_rigid = skinning.get_transform_mats_from_quat_rots(abs_trans, abs_rot_quat)[1:] # TODO...
                
          
            skel_mesh_points_rigid = rigidly_posed_locations[2:] # TODO: get rid of root bone convention
            if RENDER_MESH: mesh_points_rigid = skinning.LBS_from_quat(V_rest, W, abs_rot_quat[1:], abs_trans[1:], use_normalized_weights=NORMALIZE_WEIGHTS) # TODO: get rid of root
           
            dyn_posed_locations = helper_rig.update_bones(rigidly_posed_locations) # Update the rigidly posed locations
            skel_mesh_points_dyn = dyn_posed_locations[2:] # TODO: get rid of root bone convention
                   
            M = inverse_kinematics.get_absolute_transformations(rest_bone_locations, dyn_posed_locations, return_mat=True, algorithm=ALGO)[1:]  # TODO: get rid of root
                    
            M_hybrid = M_rigid
            M_hybrid[helper_idxs] = M[helper_idxs]
            if RENDER_MESH: mesh_points_dyn = skinning.LBS_from_mat(V_rest, W, M_hybrid, use_normalized_weights=NORMALIZE_WEIGHTS)
                       
            
            if RENDER_MESH: 
                V_anim_rigid.append(mesh_points_rigid)
                V_anim_dyn.append(mesh_points_dyn)
            if RENDER_SKEL: 
                J_anim_rigid.append(skel_mesh_points_rigid)
                J_anim_dyn.append(skel_mesh_points_dyn)

# Compute differences between rigid and jiggling
V_dyn = np.array(V_anim_dyn)
V_rigid = np.array(V_anim_rigid)
n_frames = max(len(V_anim_rigid), len(J_anim_rigid))

distance_err_dyn = np.linalg.norm(V_rigid - V_dyn, axis=-1)  # (n_frames, n_verts)
tot_err_dyn =  np.sum(distance_err_dyn)
avg_err_dyn = tot_err_dyn / n_frames
normalized_dists = normalize_arr_np(distance_err_dyn) 
print(">> Total vertex distance difference: ", np.round(tot_err_dyn,4))
print(">> Average per-vertex difference: ", np.round(avg_err_dyn, 4))


# ---------------------------------------------------------------------------------
# Show computed results
# ---------------------------------------------------------------------------------
if RENDER_AS_GIF:
    plotter.open_gif(os.path.join(RESULT_PATH, f"{MODEL_NAME}_{INTEGRATION}.gif"))
else:
    plotter.open_movie(os.path.join(RESULT_PATH, f"{MODEL_NAME}-{ALGO}.mp4"))
    
for frame in range(n_frames):
    # Set data for renderer
    if RENDER_MESH: 
        mesh_rigid.points = V_anim_rigid[frame]
        mesh_dyn.points = V_anim_dyn[frame]
        if COLOR_CODE:  # Color code jigglings 
            set_mesh_color_scalars(mesh_dyn, normalized_dists[frame])  
        
    if RENDER_SKEL: 
        skel_mesh_rigid.points = J_anim_rigid[frame] 
        skel_mesh_dyn.points = J_anim_dyn[frame]
    
    frame_text_actor.input = str(frame+1)
    plotter.write_frame()   # Write a frame. This triggers a render.
    

plotter.close()
plotter.deep_clean()



