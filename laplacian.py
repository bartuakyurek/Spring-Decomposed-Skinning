#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Computes cot Laplacian and its smallest 20 eigenfunctions.
Selected eigenfunction is visually rendered in output .html file.

"""

import igl 
import numpy as np
import scipy
from scipy.sparse.linalg import eigsh
import meshplot
from meshplot import plot

from get_color_dists import get_color_by_val
from colors import my_cmap_RGB
from matplotlib import cm
import matplotlib.pyplot as plt


# TODO: i took it online but it is too slow compared to igl's implementation!
def cotangent_laplacian(vertices, faces, epsilon: float=1e-12):
    vertices_count, faces_count = vertices.shape[0], faces.shape[0]
    vertices_grouped_by_faces = vertices[faces]
    v0, v1, v2 = vertices_grouped_by_faces[:, 0], vertices_grouped_by_faces[:, 1], vertices_grouped_by_faces[:, 2]
    # use vector norm to find length of face edges
    A = np.linalg.norm((v1 - v2), axis=1)
    B = np.linalg.norm((v0 - v2), axis=1)
    C = np.linalg.norm((v0 - v1), axis=1)
    '''Heron's formula'''
    s = 0.5 * (A + B + C)
    area = np.sqrt((s * (s - A) * (s - B) * (s - C)).clip(min=epsilon))
    '''Law of cosines, in cotangnet'''
    A_squared, B_squared, C_squared = A * A, B * B, C * C
    cotangent_a = (B_squared + C_squared - A_squared) / area
    cotangent_b = (A_squared + C_squared - B_squared) / area
    cotangent_c = (A_squared + B_squared - C_squared) / area
    cot = np.stack([cotangent_a, cotangent_b, cotangent_c], axis=1)
    cot /= 4.0
    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = np.stack([ii, jj], axis=0).reshape((2, faces_count * 3))
    m = np.zeros(shape=(vertices_count, vertices_count))
    m[idx[0], idx[1]] = cot.reshape(-1)
    m += m.T
    return m


def is_symmetric(A, tol=1e-8):
    try: 
        return scipy.sparse.linalg.norm(A-A.T, scipy.Inf) < tol;
    except:
        return np.all(np.abs(A-A.T) < tol)


COLOR_MAP_CHOICE = "hot"
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
    num_verts = V.shape[0]
    # _, V, F, _, _ = igl.decimate(V, F, 512)
    
    L = igl.cotmatrix(V, F)
    #L = cotangent_laplacian(V, F)
    #L = scipy.sparse.csc_matrix(L)
    
    M = igl.massmatrix(V, F, type=igl.MASSMATRIX_TYPE_BARYCENTRIC)
    M_inv = M
    M_inv.data = 1. / M.data
    Laplace_Beltrami = M_inv * L
    L = Laplace_Beltrami
    
    if is_symmetric(L):
        print(">> Found symmetric Laplacian.")
        eigvals, eigvecs = eigsh(L, k=num_eigvecs, which="SM")
    else:
        print(">> Found asymmetric Laplacian.")
        eigvals, eigvecs = scipy.sparse.linalg.eigs(L, k=num_eigvecs, which="SM")
        
    # eigvals, eigvecs = np.linalg.eig(L.todense()) --> computationally expensive
    for i in selected_eigen_function:
    
        eigvec = np.real(eigvecs[:,i])
        
        normalized_eigvec = eigvec - np.min(eigvec)
        normalized_eigvec /=   np.max(normalized_eigvec) 
        
        colors = []
        for val in normalized_eigvec:

            if COLOR_MAP_CHOICE == "hot":
                colors.append(cm.hot(val)[0:3])
                
            elif COLOR_MAP_CHOICE == "rainbow":
                colors.append(my_cmap_RGB(val)[0:3])
                
            elif COLOR_MAP_CHOICE == "jet":
                colors.append(cm.jet(val)[0:3])
                
            else: #if COLOR_MAP_CHOICE == "default":
                colors.append(get_color_by_val(val))
                
        #plt.plot(eigvec)
        #plt.ylabel(f'eigen_vector_{i}')
        #plt.show()
        
        colors = np.array(colors)
        meshplot.offline()
        plot(V, F, colors, filename="./results/eigenfunc/Tpose-mode-{}.html".format(i+1))
    
    eigfunc_path = "./results/laplace-beltrami-eigs.npz"
    print(">> Saving Laplace-Beltrami operator's eigen values and functions to ", eigfunc_path)
    np.savez(eigfunc_path, eigvals, eigvecs)