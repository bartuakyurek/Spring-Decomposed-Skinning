#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is created to compare our Spring Decomposed Skinning (SDS) results with the paper
"Two-Way Coupling of Skinning Transformations and Position Based Dynamics"

The input of this file is extracted by editing the original source code of Controllable PBD,
and saved it as a .npz file. 

Please also refer to their implementation presented at 
https://yoharol.github.io/pages/control_pbd/

Created on Thu Nov 12, 2024
@author: bartu
"""

import os
import numpy as np
import pyvista as pv

import __init__
from src import skinning
from src.data import model_data
from src.kinematics import inverse_kinematics
from src.helper_handler import HelperBonesHandler
from src.global_vars import DATA_PATH, RESULT_PATH
from src.utils.linalg_utils import normalize_arr_np
from src.skeleton import Skeleton, create_skeleton_from
from src.utils.linalg_utils import translation_vector_to_matrix
from src.render.pyvista_render_tools import (add_mesh, 
                                             add_skeleton_from_Skeleton, 
                                             set_mesh_color_scalars)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Main variables
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# faces : (n_faces, 3) dtype=int, list of indices for face connectivity in the mesh
# original_weights : (n_handles, n_verts) binding weights between vertices and the handles originally given by the existing data (we'll also introduce weights for our additional helper rig)

# verts_rest : (n_verts, 3) dtype=float, vertex locations in the mesh at the rest pose
# verts_cpbd : (n_frames, n_verts, 3)

# handle_locations_rest : (n_handles, 3) handle positions in the rest pose
# handle_locations_rigid : (n_frames, n_handles, 3) handle positions at every frame (WARNING: We assume handles are translated for this demo)
# handle_locations_cpbd : (n_frames, n_handles, 3) handle positions according to Controllable PBD output (see source code: https://github.com/yoharol/PBD_Taichi)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
MODEL_NAME = "spot"
AVAILABLE_MODES = ["point springs", "helper rig"]
MAKE_ALL_SPRING = True # Set true to turn all bones spring bones
SKELETON_MODE = AVAILABLE_MODES[1] # "point springs" or "helper rig" 

# RENDER PARAMETERS
RENDER_MESH = True
RENDER_SKEL = False
WIREFRAME = False

ADD_LIGHT = True
LIGHT_INTENSITY = 0.6 # Between [0, 1]
LIGHT_POS = (10.5, 3.5, 3.5)
                       
SMOOTH_SHADING = True # Automatically set True if RENDER_PHYS_BASED = True
RENDER_PHYS_BASED = False
OPACITY = 1.0
MATERIAL_METALLIC = 0.2
MATERIAL_ROUGHNESS = 0.3
BASE_COLOR = [0.8,0.7,1.0] # RGB

DEFAULT_BONE_COLOR = "white"
CPBD_BONE_COLOR ="green" # CPBD stands for Controllable PBD (the paper we compare against)
SPRING_BONE_COLOR = "blue"

COLOR_CODE = True # True if you want to visualize the distances between rigid and dynamic
WINDOW_SIZE = (1200, 1600)

# SIMULATION PARAMETERS
ALGO = "T" # ["T", "RST", "SVD"] RST doesn't work good with this demo, SVD never works good either
INTEGRATION = "Euler" # PBD or Euler

AUTO_NORMALIZE_WEIGHTS = True # Using unnomalized weights can cause problems
COMPLIANCE = 0.0 # Set between [0.0, inf], if 0.0 hard constraints are applied, only available if EDGE_CONSTRAINT=True    
EDGE_CONSTRAINT = True # Setting it True can stabilize springs but it'll kill the motion after the first iteration 
POINT_SPRING = False # Currently it doesn't move at all if EDGE_CONSTRAINT=True
FRAME_RATE = 24 # 24, 30, 60
TIME_STEP = 1./FRAME_RATE  
MASS = 1.
STIFFNESS = 150.
DAMPING = 10.  
MASS_DSCALE = 0.2       # Mass velocity damping (Use [0.0, 1.0] range to slow down)
SPRING_DSCALE = 1.0     # Scales spring forces (increase for more jiggling)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# READ DATA
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
SPOT_DATA_PATH = os.path.join(DATA_PATH, MODEL_NAME) 
OBJ_PATH =  os.path.join(SPOT_DATA_PATH, f"{MODEL_NAME}.obj")
TGF_PATH =  os.path.join(SPOT_DATA_PATH, f"{MODEL_NAME}.tgf")
TEXTURE_PATH = os.path.join(SPOT_DATA_PATH, f"{MODEL_NAME}_texture.png")
HELPER_RIG_PATH = os.path.join(SPOT_DATA_PATH, f"{MODEL_NAME}_rig_data.npz")
SPOT_EXTRACTED_DATA_PATH = os.path.join(SPOT_DATA_PATH, f"{MODEL_NAME}_extracted.npz")

# Read animation data
with np.load(SPOT_EXTRACTED_DATA_PATH) as data:
    
    verts_cpbd = data["verts_yoharol"]
    faces = data["faces"]
    
    handle_locations_cpbd = data["handles_yoharol"]
    handle_locations_rigid = data["handles_rigid"]

    original_weights = data["weights"]
    
verts_rest = verts_cpbd[0]
handle_locations_rest = handle_locations_rigid[0] #cpbd[0]

assert len(verts_cpbd) == len(handle_locations_cpbd), f"Expected verts and handles to have same length at dim 0. Got shapes {verts_cpbd.shape}, {handle_locations_cpbd.shape}."
assert verts_cpbd.shape[1] == len(verts_rest), "Expected the loaded data vertices to match with the loaded .obj rest vertices."

# Sanity check the loaded data
print("> Verts anim shape", verts_cpbd.shape)
print("> Faces shape", faces.shape)
print("> Handles (controllable pbd) shape", handle_locations_cpbd.shape)
print("> Handles (rigid) shape", handle_locations_rigid.shape)
print("> Handle locations at rest :\n", handle_locations_rest)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SETUP RIGS
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
W_rigid = np.array(original_weights)

print(">> WARNING: Assuming the provided handle locations are sparse point handles ")

skeleton_rigid = Skeleton(root_vec = [0.,0.,0.]) # pseudo root bone
for point_location in handle_locations_rest:
     skeleton_rigid.insert_bone(endpoint = point_location, 
                          startpoint = point_location,
                          parent_idx = 0) # pseudo root bone


n_rigid_bones = len(skeleton_rigid.rest_bones) 
# Add helper bones according to mode
if SKELETON_MODE == "point springs": # Make all bones in the existing rig spring bones
    print(">> INFO: Skeleton is taken as point springs...")
    skeleton_dyn = skeleton_rigid
    W_dyn = W_rigid
    helper_idxs = [i+1 for i in range(len(skeleton_dyn.rest_bones)-1)]
    original_bones = helper_idxs
else: # Load helper rig as an addition to rigid rig
    print(">> INFO: Loading helper rig...")
    # Load joint locations, kintree and weights
    with np.load(HELPER_RIG_PATH) as data:
         W_dyn = data["weights"] #[:,1:] # Excluding dummy root bone I put in blender
         blender_joints = data["joints"]#[1:]
         blender_kintree = data["kintree"]#[1:] - 1# Excluding dummy root bone I put in blender
         rigid_bones_blender = data["rigid_idxs"] 
        
    # Adjust weights 
    original_bones = rigid_bones_blender + 1 # [ 1,  2,  3,  4,  5,  13, 14, 18] TODO: root...
    W_dyn[:,original_bones] = W_rigid # Set rigid bone weights to original, #[1:] excluding dummy root bone I put in blender
        
    # Adjust helper bone indices
    helper_idxs = np.array([i for i in range(1, len(blender_kintree)+1)])
    if not MAKE_ALL_SPRING:
        for rigid_bone in original_bones:
                idx = np.argwhere(helper_idxs == rigid_bone)
                helper_idxs = np.delete(helper_idxs, idx)
            
    # Adjust the imported rig such that it aligns with the mesh (Blender rig export is weird, I couldn't solve it yet)
    B =  model_data.adjust_rig(blender_joints, MODEL_NAME)
    
    # Adjust rigid bone locations to the original locations (Bleder impored locations differ a bit)
    for i,rigid_idx in enumerate(rigid_bones_blender): # assumes the rigid bones are aligned with handle locations! 
        # Adjust endpoint
        B[rigid_idx, 1] = handle_locations_rest[i]
        
        # Adjust startpoint
        kintree_children = blender_kintree[:,1]
        kintree_idx = kintree_children[kintree_children == rigid_idx]
        selected_kintree = blender_kintree[kintree_idx]
        for parent_child in selected_kintree:
            parent_idx, child_idx = parent_child
            assert child_idx == rigid_idx
            if parent_idx != -1: 
                B[rigid_idx, 0] = B[parent_idx, 1]     
        
    # Create a skeleton instance
    skeleton_dyn = create_skeleton_from(B, blender_kintree)
    

helper_rig = HelperBonesHandler(skeleton_dyn, 
                                helper_idxs,
                                mass          = MASS, 
                                stiffness     = STIFFNESS,
                                damping       = DAMPING,
                                mass_dscale   = MASS_DSCALE,
                                spring_dscale = SPRING_DSCALE,
                                dt            = TIME_STEP,
                                point_spring  = POINT_SPRING,
                                edge_constraint   = EDGE_CONSTRAINT,
                                compliance    = COMPLIANCE,
                                simulation_mode = INTEGRATION) 

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SETUP PLOTS
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
plotter = pv.Plotter(notebook=False, off_screen=False,
                     window_size = WINDOW_SIZE, border=False, shape = (3,1))

def set_lights(plotter):
    if ADD_LIGHT:
        light = pv.Light(position=LIGHT_POS, light_type='headlight', intensity=LIGHT_INTENSITY)
        plotter.add_light(light)

def adjust_camera_spot(plotter):
    plotter.camera.tight(padding=0.5, view="zy", adjust_render_window=False)
    plotter.camera.azimuth = 210

#def add_texture(polydata, actor, img_path):
#    tex = pv.read_texture(img_path)
#    polydata.texture_map_to_plane(inplace=True)
#    actor.texture = tex

set_lights(plotter)

# ---------- First Plot (LBS) ----------------
plotter.subplot(0, 0)

if RENDER_MESH: 
    mesh_rigid, mesh_rigid_actor = add_mesh(plotter, verts_rest, faces,
                                            color = BASE_COLOR,
                                            return_actor=True, 
                                            opacity=OPACITY, 
                                            show_edges=WIREFRAME,
                                            pbr=RENDER_PHYS_BASED, 
                                            metallic=MATERIAL_METALLIC, 
                                            roughness=MATERIAL_ROUGHNESS,
                                            smooth_shading=SMOOTH_SHADING)
    #if RENDER_TEXTURE:
    #    add_texture(mesh_rigid, mesh_rigid_actor, TEXTURE_PATH)
  
if RENDER_SKEL: 
    skel_mesh_rigid = add_skeleton_from_Skeleton(plotter, skeleton_rigid, default_bone_color=DEFAULT_BONE_COLOR)

adjust_camera_spot(plotter)
frame_text_actor = plotter.add_text("0", (30,0), font_size=18) # Add frame number

# ---------- Second Plot (CPBD) ----------------
plotter.subplot(1, 0)
if RENDER_MESH: 
    mesh_cpbd, mesh_cpbd_actor = add_mesh(plotter, verts_rest, faces, 
                                            color = BASE_COLOR,
                                            return_actor=True, 
                                            opacity=OPACITY, 
                                            show_edges=WIREFRAME,
                                            pbr=RENDER_PHYS_BASED, 
                                            metallic=MATERIAL_METALLIC, 
                                            roughness=MATERIAL_ROUGHNESS,
                                            smooth_shading=SMOOTH_SHADING)
  
if RENDER_SKEL: 
    skel_mesh_cpbd = add_skeleton_from_Skeleton(plotter, skeleton_rigid, default_bone_color=CPBD_BONE_COLOR)

#set_lights(plotter)
adjust_camera_spot(plotter)

# ---------- Third Plot (Ours) ----------------
plotter.subplot(2, 0)
if RENDER_MESH: 
    mesh_dyn, mesh_dyn_actor = add_mesh(plotter, verts_rest, faces, 
                                            color = BASE_COLOR,
                                            return_actor=True, 
                                            opacity=OPACITY, 
                                            show_edges=WIREFRAME,
                                            pbr=RENDER_PHYS_BASED, 
                                            metallic=MATERIAL_METALLIC, 
                                            roughness=MATERIAL_ROUGHNESS,
                                            smooth_shading=SMOOTH_SHADING)

if RENDER_SKEL: 
    skel_mesh_dyn = add_skeleton_from_Skeleton(plotter, skeleton_dyn, 
                                               helper_idxs=helper_idxs, 
                                               is_smpl=True, # TODO: This is ridiculous, but I have to update the data cause I want to omit the root bone...
                                               default_bone_color=DEFAULT_BONE_COLOR, 
                                               spring_bone_color=SPRING_BONE_COLOR)

adjust_camera_spot(plotter)
#set_lights(plotter)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# COMPUTE DEFORMATION
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
print(">> WARNING: This demo assumes the handles are only translated.")
def get_LBS_spot(cur_handles, prev_handles):
    diff = cur_handles - prev_handles
    M = np.array([translation_vector_to_matrix(t) for t in diff])
    V_lbs = skinning.LBS_from_mat(verts_rest, W_rigid, M, use_normalized_weights=AUTO_NORMALIZE_WEIGHTS)

    return V_lbs # Note: I didn't compute LBS joints via FK since we are given the positions

def convert_points_to_bones(handles, flatten=True):
    point_bones =[[p,p] for p in handles]
    if flatten : point_bones =  np.reshape(point_bones, (-1,3))
    return point_bones

n_frames = len(handle_locations_rigid)
n_bones_rigid = len(skeleton_rigid.rest_bones)
V_anim_dyn, J_anim_dyn = [], []
n_bones_dyn = len(skeleton_dyn.rest_bones)
n_additional_bones = n_bones_dyn - n_bones_rigid
V_anim_rigid = []
J_anim_rigid = []
rest_bone_locations = skeleton_dyn.get_rest_bone_locations(exclude_root=False)
for i in range(n_frames):
    
    cur_handles, rest_handles = handle_locations_rigid[i], handle_locations_rigid[0] #[i-1]
    diff = cur_handles - rest_handles
    
    # --------- LBS -----------------------------------------------------------
    V_lbs = get_LBS_spot(cur_handles, rest_handles)
    V_anim_rigid.append(V_lbs)
    J_anim_rigid.append(convert_points_to_bones(cur_handles))
    
    # --------- Ours -----------------------------------------------------------
    # Prepare translation and rotations
    t = np.zeros((n_bones_dyn,3))
    t[original_bones,:] = diff
    pose = np.zeros((n_bones_dyn, 3))
     
    # Pose with FK 
    rigidly_posed_handles = skeleton_dyn.pose_bones(pose, t, degrees=True)    
    dyn_posed_handles = helper_rig.update_bones(rigidly_posed_handles)
    
    M = inverse_kinematics.get_absolute_transformations(rest_bone_locations, 
                                                        dyn_posed_handles, 
                                                        return_mat=True, 
                                                        algorithm=ALGO)[1:]  # TODO: get rid of root
    
    M_hybrid = M # TODO -> _, q,t = pose_bones(get_transformas=True) and M_rigid = compose_mat(q,t)
    J_dyn = dyn_posed_handles[2:] # TODO: remove root...
    V_dyn = skinning.LBS_from_mat(verts_rest, W_dyn, M_hybrid, 
                                  use_normalized_weights=AUTO_NORMALIZE_WEIGHTS)
               
    V_anim_dyn.append(V_dyn)
    J_anim_dyn.append(J_dyn)
    
# =============================================================================
#  Compute differences between LBS and Dynamic results   
# =============================================================================
V_anim_rigid = np.array(V_anim_rigid)
V_anim_dyn = np.array(V_anim_dyn)
distance_err_cpbd = np.linalg.norm(V_anim_rigid - verts_cpbd, axis=-1)  # (n_frames, n_verts)
distance_err_dyn = np.linalg.norm(V_anim_rigid - V_anim_dyn, axis=-1)  # (n_frames, n_verts)

normalized_dists_cpbd = normalize_arr_np(distance_err_cpbd)
normalized_dists_dyn = normalize_arr_np(distance_err_dyn)

# =============================================================================
# Display animation
# =============================================================================
plotter.open_movie(os.path.join(RESULT_PATH, f"{MODEL_NAME}_PBD_Complience_{COMPLIANCE}.mp4"))
for frame in range(n_frames):
    # Set data for renderer
    if RENDER_MESH: 
        mesh_rigid.points = V_anim_rigid[frame]
        mesh_cpbd.points = verts_cpbd[frame]
        mesh_dyn.points = V_anim_dyn[frame]
        
    if RENDER_SKEL: 
        skel_mesh_rigid.points = J_anim_rigid[frame] 
        skel_mesh_cpbd.points = convert_points_to_bones(handle_locations_cpbd[frame])
        skel_mesh_dyn.points = J_anim_dyn[frame]
    
    # Color code jigglings 
    if COLOR_CODE:
        set_mesh_color_scalars(mesh_cpbd, normalized_dists_cpbd[frame])  
        set_mesh_color_scalars(mesh_dyn, normalized_dists_dyn[frame])  
        
    frame_text_actor.input = str(frame+1)
    plotter.write_frame()   # Write a frame. This triggers a render.

plotter.close()
plotter.deep_clean()
