#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:19:53 2024

@author: bartu
"""

import matplotlib.pyplot as plt


def matplot_skeleton(joint_locations, kintree):
    """
        joints: array object including joint locations of a skeleton
        kintree: 2D array object of kinematic tree of skeleton, 
                    every instance has the joint indices of a bone's 
                    beginning joint and ending joint
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # XYZ coordinate locations
    x = 0   
    y = 1
    z = 2
    for bone in kintree:
        bone_begin_idx, bone_end_idx = bone
        bone_begin_coord = joint_locations[bone_begin_idx]
        bone_end_coord = joint_locations[bone_end_idx]
        
        ax.plot([bone_begin_coord[x], bone_end_coord[x]], 
                [bone_begin_coord[y],bone_end_coord[y]],
                zs=[bone_begin_coord[z],bone_end_coord[z]])
        
        ax.scatter(joint_locations[:,x], joint_locations[:,y], joint_locations[:,z], zdir='z', s=20)