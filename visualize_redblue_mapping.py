#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:54:33 2024

@author: bartu
"""

import numpy as np

   
IS_RIGID = True
anim_mode = "rigid" if IS_RIGID else "jiggle"

with np.load("./results/anim_{}.npz".format(anim_mode)) as data:
    v = data['arr_0']
    
with np.load("./results/faces_{}.npz".format(anim_mode)) as data:
    f = data['arr_0']
   

## Create viewer canvas and add a mesh (only single mesh is allowed rn)
single_mesh_viewer = Viewer()
single_mesh_viewer.set_mesh_animation(v, f)
#single_mesh_viewer.set_mesh_opacity(0.6)

if not IS_RIGID:
    # Add color distances to jiggling animation
    with np.load("./results/rb_color_mapping.npz") as data:
        c = data['arr_0']
    single_mesh_viewer.set_mesh_colors(c)

## Run animation
single_mesh_viewer.run_animation() #, jpg_dir=jpg_path+"{}.jpg")


     
