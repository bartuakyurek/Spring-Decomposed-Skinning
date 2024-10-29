#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:36:50 2024

@author: bartu
"""
# TODO: Why space dims are _underscored_ but not homo_coords?
#       keep the naming convention consistent by changing to either one.
_SPACE_DIMS_ = 3    # Space dimensions either 2 or 3
HOMO_COORD = False  # Adds 1 to the transformation dimensions if True

# WARNING: This is not used in every function yet
VERBOSE = True        

# TODO: why don't you use os.path...? 
ABS_PATH = "/Users/bartu/Documents/Github/Spring-Decomp/"
RESULT_PATH = ABS_PATH + "results/"
DATA_PATH = ABS_PATH + "data/"
DFAUST_PATH = DATA_PATH + "DFaust_67"
MODEL_PATH = ABS_PATH + "src/models/body_models/"
IGL_DATA_PATH = DATA_PATH +"igl_data/"

# TODO: rename this file as config, and maybe use a yaml file.