#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:38:13 2024

@author: bartu
"""

import igl
import numpy as np
import meshplot
from meshplot import plot


if __name__ == "__main__":
    OBJ_PATH = "./results/SMPL-T.obj"
    V, _, _, F, _, _ = igl.read_obj(OBJ_PATH)
    
    with np.load("./results/smpl_weight_vertex_colors.npz") as data:
        colors =  data['arr_0']
    
    meshplot.offline()
    plot(V, F, colors)
    
    