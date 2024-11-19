#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file is created to bundle the data I add to the pipeline. 

Created on Tue Oct  1 07:31:12 2024
@author: bartu
"""
import os
import numpy as np
from scipy.spatial.transform import Rotation

from . import poses
from ..global_vars import DATA_PATH


def get_obj_path(name): return os.path.join(DATA_PATH, name, f"{name}.obj")
def get_rig_path(name): return os.path.join(DATA_PATH, name, f"{name}_rig_data.npz")
def get_texture_path(name): return os.path.join(DATA_PATH, name, f"{name}_texture.png") # If set to None, default texture will be used

duck = "duck"
blob = "blob"
monstera = "monstera"
cloth = "cloth"

model_dict = {
  duck: {"OBJ_PATH": get_obj_path(duck),
         "RIG_PATH":get_rig_path(duck),
         "TEXTURE_PATH": get_texture_path(duck),
         "keyframe_poses": poses.duck_rig_pose,
         "helper_idxs": np.array([i for i in range(1, 6)])
         },
  
  blob : {"OBJ_PATH": get_obj_path(blob),
          "RIG_PATH": get_rig_path(blob),
          "TEXTURE_PATH": get_texture_path(blob),
          "keyframe_poses": poses.blob_rig_pose,
          "helper_idxs": np.array([i for i in range(1, 35)]) #[5,6,21,22,25,26,27,33],
      },
  monstera : {"OBJ_PATH": get_obj_path(monstera),
          "RIG_PATH": get_rig_path(monstera),
          "TEXTURE_PATH": get_texture_path(monstera),
          "keyframe_poses": poses.monstera_rig_pose,
          "helper_idxs":  np.array([i for i in range(1, 24)]),
      },
  cloth : {"OBJ_PATH": get_obj_path(cloth),
          "RIG_PATH": get_rig_path(cloth),
          "TEXTURE_PATH": get_texture_path(cloth),
          "keyframe_poses": poses.cloth_rig_pose,
          "helper_idxs":  np.array([i for i in range(1, 23)]),
      },
  
  "sample" : {"OBJ_PATH": None,
          "RIG_PATH": None,
          "TEXTURE_PATH":None,
          "keyframe_poses": None,
          "helper_idxs": None,
      },
}

def adjust_rig(B, MODEL_NAME):
    
    if MODEL_NAME == "spot" or MODEL_NAME == "spot_high":
        r = Rotation.from_euler('x', -90, degrees=True)
        for i in range(len(B)): B[i] = r.apply(B[i])
        
        B = B - np.array([0, 0, -0.8])  # Translate to origin
        B = B * 0.35 # Scale
    
    if MODEL_NAME == "duck":
        B = B * 2.8 # Scale
    
    elif MODEL_NAME == "blob":
        r = Rotation.from_euler('x', -90, degrees=True)
        for i in range(len(B)): B[i] = r.apply(B[i])
        B = B * 0.6
        
    elif MODEL_NAME == "cloth":
        pass
    
    elif MODEL_NAME == "monstera":
        r = Rotation.from_euler('x', -90, degrees=True)
        for i in range(len(B)):  B[i] = r.apply(B[i])
            
        B = B - np.array([0, 0, 0.6])  # Translate to origin
        B = B * 0.37 # Scale
    else:
        print(">> WARNING: Model {MODEL_NAME} has no defined adjust_rig() yet.")
    
    return B

def adjust_camera(plotter, MODEL_NAME, resize_window=False):
    
    if MODEL_NAME == "duck":
        plotter.camera.tight(padding=3, view="yz", adjust_render_window=resize_window)
        plotter.camera.position = [0.0, 5.0, 4.0]
        plotter.camera.focal_point = (0.0, 0.5, 3.5)
        plotter.camera.roll = 180
        
    elif MODEL_NAME == "blob":
        plotter.camera_position = 'zy'
        plotter.camera.position = [-15.0, 0.0, 0]
        plotter.camera.view_angle = 20 # This works like zoom actually
        plotter.camera.focal_point = (0.0, 0.3, 0.0)
        
    elif MODEL_NAME == "cloth":
        plotter.camera_position = 'zy'
        plotter.camera.position = [-15.0, 0.0, 0]
        plotter.camera.view_angle = 20 # This works like zoom actually
        plotter.camera.focal_point = (0.0, 0.0, 0.0)
        plotter.roll = 90
    elif MODEL_NAME == "monstera":
        plotter.camera_position = 'zy'
        plotter.camera.position = [-8.0, 2.0, 0]
        plotter.camera.view_angle = 20 # This works like zoom actually
        plotter.camera.focal_point = (0.0, 0.65, 0.0)
    else:
        plotter.camera.tight(padding=3, view="yz")
        plotter.camera.position = [0.0, 5.0, 4.0]
        plotter.camera.focal_point = (0.0, 0.5, 3.5)
        plotter.camera.roll = 180