#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:19:34 2024

@author: bartu
"""

import igl
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import meshplot
from meshplot import plot, subplot, interact

from get_color_dists import get_color_by_val

def cotan_laplacian(V, F):
    pass

if __name__ == "__main__":
    
    OBJ_PATH = "/Users/Bartu/Documents/Datasets/DFAUST/results/50004_jiggle_on_toes/00010.obj"
    V, _, _, F, _, _ = igl.read_obj(OBJ_PATH)  
    
    L = igl.cotmatrix(V, F)
    num_eigvecs = 20
    eigvals, eigvecs = eigsh(L, k=num_eigvecs, which="SM")
    
    i = selected_eigen_function = 19
    eigvec = eigvecs[:,i]
    
    normalized_eigvec = eigvec - np.min(eigvec)
    normalized_eigvec /=   np.max(normalized_eigvec) 
    
    colors = []  
    for val in normalized_eigvec:
        colors.append(get_color_by_val(val))
    
    colors = np.array(colors)
    
    meshplot.offline()
    plot(V, F, colors)
    
    #plt.plot(eigvecs[:, i])
    #plt.show()
