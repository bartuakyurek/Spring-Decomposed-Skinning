#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Computes cot Laplacian and its smallest K eigenfunctions for selected animation frames
The resulting eigen decomposition is saved in the end.

"""

import igl 
import numpy as np
import scipy
from scipy.sparse.linalg import eigsh


def is_symmetric(A, tol=1e-8):
    try: 
        return scipy.sparse.linalg.norm(A-A.T, scipy.Inf) < tol;
    except:
        return np.all(np.abs(A-A.T) < tol)
    
def get_laplacian(V, F):
    L = igl.cotmatrix(V, F)
    
    M = igl.massmatrix(V, F, type=igl.MASSMATRIX_TYPE_BARYCENTRIC)
    M_inv = M
    M_inv.data = 1. / M.data
    Laplace_Beltrami = M_inv * L
    
    return Laplace_Beltrami

def get_laplacian_batch(V, F):
    assert len(V.shape) == 3, "Expected V.shape to have length 3, (#frames, #verts, #dims)"
    
    L_batch = []
    for frame_verts in V:
        Laplace_Beltrami = get_laplacian(frame_verts, F)
        L_batch.append(Laplace_Beltrami)
        
    return L_batch


def get_eigen_of_laplacian(L):
    assert len(L.shape) == 2, "Expected L.shape to have length 2"

    if is_symmetric(L):
        print(">> Found symmetric Laplacian.")
        eigvals, eigvecs = eigsh(L, k=num_eigvecs, which="SM")
    else:
        print(">> Found asymmetric Laplacian.")
        eigvals, eigvecs = scipy.sparse.linalg.eigs(L, k=num_eigvecs, which="SM")
        
    return eigvals, eigvecs
    
def get_eigen_of_laplacian_batch(L):

    eigvals_batch, eigvecs_batch = [], []
    for laplacian in L:
       eigvals, eigvecs = get_eigen_of_laplacian(laplacian)
       
       eigvals_batch.append(eigvals)
       eigvecs_batch.append(eigvecs)
     
    return eigvals_batch, eigvecs_batch


def normalize_eigvec(eigvec):
    eigvec = np.real(eigvec)
    
    normalized_eigvec = eigvec - np.min(eigvec)
    normalized_eigvec /=   np.max(normalized_eigvec) 
    
    return normalized_eigvec
    

if __name__ == "__main__":
    
    ######### PARAMETERS TO BE SET ####################
    num_eigvecs = 10
    selected_frames = np.arange(10, 110, 5)
    ###################################################
    
    # Load object   
    with np.load("./data/dfaust_sample_data.npz") as file:
        # file contents:
        #   betas, pose, trans, target_verts, smpl_verts, faces, joints
        V = file['arr_3']
        F = file['arr_5']
        
        V = np.array(V, dtype=float)
        F = np.array(F, dtype=int)
    
    V_selected = V[selected_frames]
    L = get_laplacian_batch(V_selected, F)
    eigvals_batch, eigvecs_batch = get_eigen_of_laplacian_batch(L)
    
    np.savez("./results/eigdecomp_batch_{}_frames.npz".format(len(selected_frames)), V_selected, selected_frames, L, eigvals_batch, eigvecs_batch)
    
    """
    for i in range(num_eigvecs):
        eigvec = np.real(eigvecs[:,i])
        normalized_eigvec = normalize_eigvec(eigvec)
        
        for val in normalized_eigvec:
            pass
    """
       
    #eigfunc_path = "./results/laplace-beltrami-eigs.npz"
    #print(">> Saving Laplace-Beltrami operator's eigen values and functions to ", eigfunc_path)
    #np.savez(eigfunc_path, eigvals, eigvecs)