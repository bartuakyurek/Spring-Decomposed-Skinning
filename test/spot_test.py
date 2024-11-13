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
import igl
import numpy as np
import pyvista as pv

import __init__
from src.global_vars import DATA_PATH, RESULT_PATH
from src.skeleton import Skeleton, add_helper_bones
from src.data import model_data
from src import skinning
from src.utils.linalg_utils import lerp, translation_vector_to_matrix
from src.global_vars import RESULT_PATH
from src.kinematics import inverse_kinematics
from src.helper_handler import HelperBonesHandler
from src.utils.linalg_utils import normalize_arr_np
from src.skeleton import create_skeleton_from
from src.render.pyvista_render_tools import (add_mesh, 
                                             add_skeleton_from_Skeleton, 
                                             set_mesh_color_scalars,
                                             set_mesh_color)

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

RENDER_MESH = True
RENDER_SKEL = True
WIREFRAME = False
RENDER_PHYS_BASED = False
AUTO_NORMALIZE_WEIGHTS = False

OPACITY = 0.8
MATERIAL_METALLIC = 0.0
MATERIAL_ROUGHNESS = 0.2

DEFAULT_BONE_COLOR = "white"
CPBD_BONE_COLOR ="green" # CPBD stands for Controllable PBD (the paper we compare against)
#SPRING_BONE_COLOR = "blue"

#COLOR_CODE = True # True if you want to visualize the distances between rigid and dynamic

#EYEDOME_LIGHT = False
WINDOW_SIZE = (1500 * 2, 1200)



# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# READ DATA
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
SPOT_DATA_PATH = os.path.join(DATA_PATH, MODEL_NAME) 
OBJ_PATH =  os.path.join(SPOT_DATA_PATH, f"{MODEL_NAME}.obj")
TGF_PATH =  os.path.join(SPOT_DATA_PATH, f"{MODEL_NAME}.tgf")
SPOT_HELPER_RIG_PATH = os.path.join(DATA_PATH, f"{MODEL_NAME}_helper_rig_data.npz")
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
W = np.array(original_weights)

print(">> WARNING: Assuming the provided handle locations are sparse point handles ")

skeleton = Skeleton(root_vec = [0.,0.,0.]) # pseudo root bone
for point_location in handle_locations_rest:
     skeleton.insert_bone(endpoint = point_location, 
                          startpoint = point_location,
                          parent_idx = 0) # pseudo root bone
    
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# SETUP PLOTS
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
plotter = pv.Plotter(notebook=False, off_screen=False) 
                     #window_size = WINDOW_SIZE, border=False, shape = (1,1))

def adjust_camera_spot(plotter):
    plotter.camera.tight(padding=1, view="zy")
    plotter.camera.azimuth = 180

# ---------- First Plot ----------------
plotter.subplot(0, 0)
if RENDER_MESH: 
    mesh_rigid, mesh_rigid_actor = add_mesh(plotter, verts_rest, faces, 
                                            return_actor=True, 
                                            opacity=OPACITY, 
                                            show_edges=WIREFRAME,
                                            pbr=RENDER_PHYS_BASED, 
                                            metallic=MATERIAL_METALLIC, 
                                            roughness=MATERIAL_ROUGHNESS)
  
if RENDER_SKEL: 
    skel_mesh_rigid = add_skeleton_from_Skeleton(plotter, skeleton, default_bone_color=DEFAULT_BONE_COLOR)
adjust_camera_spot(plotter)
frame_text_actor = plotter.add_text("0", (600,0), font_size=18) # Add frame number

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# COMPUTE DEFORMATION
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
print(">> WARNING: This demo assumes the handles are only translated.")

n_frames = len(handle_locations_rigid)
n_bones = len(skeleton.rest_bones)
V_anim_rigid = [verts_rest]
J_anim_rigid = [skeleton.get_rest_bone_locations(exclude_root=True)]
for i in range(1, n_frames):
    diff = handle_locations_rigid[i] - handle_locations_rigid[i-1]
    
    M = np.array([translation_vector_to_matrix(t) for t in diff])
    V_lbs = skinning.LBS_from_mat(verts_rest, W, M, use_normalized_weights=AUTO_NORMALIZE_WEIGHTS)
    
    V_anim_rigid.append(V_lbs)
    
    point_bones = np.reshape([[p,p] for p in handle_locations_rigid[i]],(-1,3))
    J_anim_rigid.append(point_bones)
    #t = np.append(np.zeros((1,3)), diff, axis=0) # TODO: remove pseudo root
    #pose = np.zeros((n_bones, 3)
    #posed_handles = skeleton.pose_bones(pose, t, degrees=True)
    #tmp = posed_handles[2:]
    #idxs = [j for j in range(1,len(tmp),2)]
    #assert np.sum(tmp[idxs] - handle_locations_rigid[i]) < 1e-10
    

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DISPLAY ANIMATION
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------
plotter.open_movie(RESULT_PATH + f"/{MODEL_NAME}_comparison.mp4")
for frame in range(n_frames):
    # Set data for renderer
    if RENDER_MESH: 
        mesh_rigid.points = V_anim_rigid[frame]
        #mesh_dyn.points = V_anim_dyn[frame]
        
    if RENDER_SKEL: 
        skel_mesh_rigid.points = J_anim_rigid[frame] 
        #skel_mesh_dyn.points = J_anim_dyn[frame]
    
    # Color code jigglings 
    #if COLOR_CODE:
    #    set_mesh_color_scalars(mesh_dyn, normalized_dists[frame])  
        
    frame_text_actor.input = str(frame+1)
    plotter.write_frame()   # Write a frame. This triggers a render.
    

plotter.close()
plotter.deep_clean()