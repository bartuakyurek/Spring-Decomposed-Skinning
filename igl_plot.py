#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 17:04:14 2024

@author: bartu
"""

import igl
import scipy as sp
import numpy as np
import meshplot
from meshplot import plot, subplot, interact
from meshplot import Viewer
import meshplot as mp
import time


IS_RIGID = True
anim_mode = "rigid" if IS_RIGID else "jiggle"

with np.load("./results/anim_{}.npz".format(anim_mode)) as data:
    V = data['arr_0']
    
with np.load("./results/faces_{}.npz".format(anim_mode)) as data:
    f = data['arr_0']
   
if not IS_RIGID:
    # Add color distances to jiggling animation
    with np.load("./results/rb_color_mapping.npz") as data:
        c = data['arr_0']


meshplot.offline()
"""
#plot(v[0], f)
viewer_settings = {"width": 600, "height": 600, "antialias": True, "scale": 1.5, "background": "#ffffff",
                "fov": 30}
viewer = Viewer(viewer_settings)
viewer.add_mesh(v[0], f)

viewer.update_object(vertices=v[0])
#plot(v[0], plot=viewer)
viewer.save()
"""

v = V[0]
p = mp.plot(v, f)

for i in range(50):
    v_new = V[i]
    p.update_object(vertices=v_new)
    v = v_new
    time.sleep(0.1) # depending on how long your simulation step takes you want to wait for a little while