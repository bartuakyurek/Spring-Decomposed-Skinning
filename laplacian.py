#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Computes cot Laplacian and its smallest 20 eigenfunctions.
Selected eigenfunction is visually rendered in output .html file.

"""

import igl 
import numpy as np
from scipy.sparse.linalg import eigsh
import meshplot
from meshplot import plot

from get_color_dists import get_color_by_val, my_cmap
from matplotlib import cm
 
COLOR_MAP_CHOICE = "jet"
#"hot" # "rainbow", # "default" # "jet"
                    
if __name__ == "__main__":
    
    ######### PARAMETERS TO BE SET ####################
    num_eigvecs = 20
    selected_eigen_function = [0, 1, 2, 3, 4, 10, 15, 19]
    OBJ_PATH = "./results/SMPL-T.obj" #homer.obj"
    #OBJ_PATH = "/Users/Bartu/Documents/Datasets/DFAUST/results/50004_jiggle_on_toes/00010.obj"
    ###################################################
    
    # Load object    
    V, _, _, F, _, _ = igl.read_obj(OBJ_PATH)    
    # _, V, F, _, _ = igl.decimate(V, F, 512)
    
    L = igl.cotmatrix(V, F)
    eigvals, eigvecs = eigsh(L, k=num_eigvecs, which="SM")
    
    for i in selected_eigen_function:
        eigvec = eigvecs[:,i]
        
        normalized_eigvec = eigvec - np.min(eigvec)
        normalized_eigvec /=   np.max(normalized_eigvec) 
        
        
        colors = []  
        for val in normalized_eigvec:

            if COLOR_MAP_CHOICE == "hot":
                colors.append(cm.hot(val)[0:3])
                
            elif COLOR_MAP_CHOICE == "rainbow":
                colors.append(my_cmap(val)[0:3])
                
            elif COLOR_MAP_CHOICE == "jet":
                colors.append(cm.jet(val)[0:3])
                
            else: #if COLOR_MAP_CHOICE == "default":
                colors.append(get_color_by_val(val))
        
        colors = np.array(colors)
        
        meshplot.offline()
        plot(V, F, colors, filename="Tpose-mode-{}.html".format(i+1))
    