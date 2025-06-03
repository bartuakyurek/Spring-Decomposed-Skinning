#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:36:50 2024

# TODO: use yaml

@author: bartu
"""

_SPACE_DIMS_ = 3    # Space dimensions either 2 or 3
HOMO_COORD = False  # Adds 1 to the transformation dimensions if True

# WARNING: This is not used in every function yet
VERBOSE = True        

ABS_PATH = "/Users/USER/Documents/Github/Spring-Decomp/"
RESULT_PATH = ABS_PATH + "results/"
DATA_PATH = ABS_PATH + "data/"
DFAUST_PATH = DATA_PATH + "DFaust_67"
MODEL_REGIS_PATH = DATA_PATH + "dyna/dyna_dataset_f.h5" # Registrations of SMPL model
MODEL_PATH = ABS_PATH + "src/models/body_models/"
IGL_DATA_PATH = DATA_PATH +"igl_data/"


# Available DFAUST ids
subject_ids = ['50004', '50020', '50021', '50022', '50025',
               '50002', '50007', '50009', '50026', '50027']

pose_ids = ['hips', 'knees', 'light_hopping_stiff', 'light_hopping_loose',
            'jiggle_on_toes', 'one_leg_loose', 'shake_arms', 'chicken_wings',
            'punching', 'shake_shoulders', 'shake_hips', 'jumping_jacks',
            'one_leg_jump', 'running_on_spot']
