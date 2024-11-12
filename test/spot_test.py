#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is created to compare our Spring Decomposed Skinning results with the paper
"Two-Way Coupling of Skinning Transformations and Position Based Dynamics"

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

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Main variables
# --------------------------------------------------------------------------------------------------------------------------------------------------------
# faces : (n_faces, 3) dtype=int, list of indices for face connectivity in the mesh

# verts_rest : (n_verts, 3) dtype=float, vertex locations in the mesh at the rest pose
# verts_controllable_pbd : (n_frames, n_verts, 3)

# handle_locations_rest : (n_handles, 3) handle positions in the rest pose
# handle_locations_rigid : (n_frames, n_handles, 3) handle positions at every frame (WARNING: We assume handles are translated for this demo)
# handle_locations_controllable_pbd : (n_frames, n_handles, 3) handle positions according to Controllable PBD output (see source code: https://github.com/yoharol/PBD_Taichi)
# --------------------------------------------------------------------------------------------------------------------------------------------------------

model_name = "spot"
SPOT_DATA_PATH = os.path.join(DATA_PATH, model_name) 
SPOT_EXTRACTED_DATA_PATH = os.path.join(SPOT_DATA_PATH, f"{model_name}_extracted.npz")

# Read animation data
with np.load(SPOT_EXTRACTED_DATA_PATH) as data:
    
    verts_controllable_pbd = data["verts_yoharol"]
    faces = data["faces"]
    
    handle_locations_controllable_pbd = data["handles_yoharol"]
    handle_locations_rigid = data["handles_rigidl"]

    rigid_handle_weights = data["weights"]
    
# Sanity check the loaded data
print("> Verts anim shape", verts_controllable_pbd.shape)
print("> Faces shape", faces.shape)
print("> Handles shape", handle_locations_controllable_pbd.shape)
print("> Handle locations anim [0] :\n", handle_locations_controllable_pbd[0])

verts_rest = verts_controllable_pbd[0]
handle_locations_rest = handle_locations_controllable_pbd[0]
assert len(verts_controllable_pbd) == len(handle_locations_controllable_pbd), f"Expected verts and handles to have same length at dim 0. Got shapes {verts_controllable_pbd.shape}, {handle_locations_controllable_pbd.shape}."
assert verts_controllable_pbd.shape[1] == len(verts_rest), "Expected the loaded data vertices to match with the loaded .obj rest vertices."






