#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file is created to bundle the data I add to the pipeline. 

Created on Tue Oct  1 07:31:12 2024
@author: bartu
"""
import os
import numpy as np

from . import poses
from ..global_vars import DATA_PATH


duck = "duck"


model_dict = {
  duck: {"OBJ_PATH": os.path.join(DATA_PATH, duck, duck+".obj"),
         "RIG_PATH": os.path.join(DATA_PATH, duck, duck+"_rig_data.npz"),
         "keyframe_poses": poses.duck_rig_pose,
         "helper_idxs": np.array([i for i in range(1, 6)])
         }
}