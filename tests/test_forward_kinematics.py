#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file is created to test the Skeleton class and it's forward kinematics
to see if the kinematics are implemented correctly.


Created on Thu Oct 10 14:34:34 2024
@author: bartu
"""
import igl
import __init__
import numpy as np
import pyvista as pv
from pyvista_render_tools import add_skeleton
from skeleton import Skeleton

joint_locations = np.array([
                            [1., 1., 0.],
                            [1., 2., 0.],
                            [1., 3., 0.]
                            ])
kintree = np.array([
                    [0, 1],
                    [1, 2]
                    ])

EXCLUDE_ROOT = False
# ---------------------------------------------------------------------------- 
# Create skeleton based on data
# ---------------------------------------------------------------------------- 
# TODO: if you always create root_vec based on your first data, why don't you 
#       include it in the insert_bone phase to make it easier to read?
test_skeleton = Skeleton(root_vec = joint_locations[0])
for edge in kintree:
     parent_idx, bone_idx = edge
     test_skeleton.insert_bone(endpoint_location = joint_locations[bone_idx], 
                               parent_node_idx = parent_idx)
     
# ---------------------------------------------------------------------------- 
# Add skeleton mesh based on T-pose locations
# ---------------------------------------------------------------------------- 
n_bones = len(test_skeleton.bones)
rest_bone_locations = test_skeleton.get_rest_bone_locations(exclude_root=EXCLUDE_ROOT)
line_segments = np.reshape(np.arange(0, 2*(n_bones-1)), (n_bones-1, 2))

print("Rest bone locations:")
for i in range(n_bones):
    print(rest_bone_locations[2*i], "-->",rest_bone_locations[2*i+1] )

pose = np.array(
                [
                 [0.,0. ,0.],
                 [30.,0.,0.],
                 [0.,0. ,0.],
                ]
                )

posed_bones = test_skeleton.pose_bones(pose, degrees=True, exclude_root=EXCLUDE_ROOT)
          
print("Posed bone locations:")
for i in range(n_bones):
    print(posed_bones[2*i], "-->",posed_bones[2*i+1] )
    
    