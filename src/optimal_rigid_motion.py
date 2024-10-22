#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 06:36:20 2024

This file implements finding the optimal rigid motion (rotation and translation)
between two point sets P1 and P2. When the optimal rigid motion, 
i.e. a spatial transformation is applied to P1, the transformed P1' is a better 
fit for the P2 than original P1. If P2 is the rigidly transformed version of
P1, this transformation directly maps P1 --> P2.

Note that it doesn't necessarily mean the source will be snapped on the target.
However, if the target set is indeed a rigidly transformed version of the
source set, this file should return one to one correspondence.

[!] WARNING: optimal rigid motion works best when the point sets have at 
least 3 points. It may fail with sets with 2 points.

It's based on the As Rigid As Possible deformation's SVD based implementation.
See https://igl.ethz.ch/projects/ARAP/svd_rot.pdf for the implementation details.

@author: bartu
"""

# ================================================================================================================
#                                           IMPORTS AND GLOBALS
# ================================================================================================================

import numpy as np
from .utils.sanity_check import _assert_equality
from .global_vars import _SPACE_DIMS_

# ================================================================================================================
#                            SANITY CHECK FUNCTIONS FOR OPTIMAL RIGID MOTION ALGORITHM
# ================================================================================================================

def __check_icp_set_shapes(P, Q, W):
    # Sanity checks for inputs of ICP algorithm
    # P is the original point set
    # Q is the target point set
    # W is the weight matrix corresponding for each point pair P->Q
    # See https://igl.ethz.ch/projects/ARAP/svd_rot.pdf for details
    assert len(P.shape) == 2, f"Point sets must have shape (num_points, dims) shape.Provided P has {P.shape}."
    assert len(Q.shape) == 2, f"Point sets must have shape (num_points, dims) shape.Provided Q has {Q.shape}."
    assert P.shape == Q.shape, f"Point sets must have the same shape. Provided are {P.shape} and {Q.shape}."
    
    n_points, n_dims = P.shape
    assert W.shape == (n_points, ) or W.shape == (n_points, 1), f"Weights matrix must have shape  ({n_points}, ) or ({n_points}, 1). Provided has shape {W.shape}"
    return (n_points, n_dims)

def __naive_weighted_sum(values, weights):
    # This function is used as a sanity check
    # for matrix mulptiplications
    assert len(values) == len(weights), f"Values and weights must have the same length. Provided has shape {values.shape}, and {weights.shape}"
    assert len(values.shape) == 2 or len(values.shape) == 1, f"Batched inputs are not supported yet."
    
    weighted_sum = 0.0
    if  len(values.shape) == 2:
        weighted_sum = np.zeros(values.shape[1])
    
    for i in range(len(values)):
        weighted_sum += weights[i] * values[i]
        
    return weighted_sum

# ================================================================================================================
#                               CORE FUNCTIONS OF FINDING THE OPTIMAL RIGID MOTION
# ================================================================================================================


def get_centroid(point_coords, weights):
    # Returns the coordinate of weighted centroid point 
    # Given a set of 3D or 2D points and their corresponding weight
    
    # Note: set all weights to 1 if you want a uniform average over points
    # Note: returned array has shape (_SPACE_DIMS_, )
    n_points, _ = point_coords.shape
    if weights.shape == (n_points, 1):
        weights = weights.squeeze()
        
    total_weights = np.sum(weights)
    weighted_point_sum = point_coords.T @ weights
    
    sanity_check = __naive_weighted_sum(point_coords, weights)
    _assert_equality(sanity_check, weighted_point_sum)
    
    centroid = weighted_point_sum / total_weights
    assert centroid.shape == (_SPACE_DIMS_, )
    
    return centroid

def get_optimal_rigid_motion(P, Q, W=None):
    """
    Get optimal rotation and translation from P to Q. 
    That is, when P is rotated and translated, it'll fit better to Q.
    
    Parameters
    ----------
    P : np.ndarray
        Source point set has shape (n_points, n_dims) for P->Q mapping.
    Q : np.ndarray
        Target point set has shape (n_points, n_dims) for P->Q mapping.
    W : np.ndarray (optional)
        Weight matrix of shape (n_points) that is the weighting between P->Q
        mapping.

    Returns
    -------
    Rot : np.ndarray
        Rotation matrix has shape (3, 3)
    trans : np.ndarray
        Translation vector has shape (3,)

    """
    if W is None: W=np.ones(P.shape[0])
    
    n_points, n_dims = __check_icp_set_shapes(P, Q, W)
    assert n_dims == _SPACE_DIMS_, f"Number of dimensions in given set must be equal to {_SPACE_DIMS_} in {_SPACE_DIMS_}D setting. (See _SPACE_DIMS_ declaration)"  
    if W.shape == (n_points, 1): W = W.squeeze()
   
    # Step 1: Compute weighted centroids of P and Q
    P_centroid = get_centroid(P, W)
    Q_centroid = get_centroid(Q, W)
    
    # Step 2: Subtract of centroid coordinate from all points in the set
    X = P_centered = P - P_centroid 
    Y = Q_centered = Q - Q_centroid
    
    # Step 3: Compute the covariance matrix
    W_diag = np.diag(W)    # n x n matrix
    X_col_major = X.T      # d x n matrix
    Y_col_major = Y.T      # d x n matrix
    
    assert W_diag.shape == (n_points, n_points), f"Diagonal weight matrix has to be ({n_points}, {n_points}), found {W_diag.shape} "
    assert X_col_major.shape == (n_dims, n_points)
    assert X_col_major.shape == Y_col_major.shape
    
    # S is the coveriance matrix
    S = X.T @ (W_diag @ Y) 
    sanity_check = X_col_major @ W_diag @ Y_col_major.T # Corresponds to matrix dimensions on notes 
    _assert_equality(S, sanity_check)
    
    # Step 4: Compute the SVD and the optimal rotation
    U, Sigma, Vh = np.linalg.svd(S, full_matrices=True)
    
    assert U.shape == (n_dims, n_dims)
    assert Sigma.shape == (n_dims, )
    assert Vh.shape == (n_dims, n_dims)
    
    S_sanity = U @ (np.diag(Sigma) @ Vh)
    assert S_sanity.shape == S.shape, f"Sanity Check Failed! SVD Reconstructed matrix has shape {S_sanity.shape}, expected {S.shape}"
    _assert_equality(S[0], S_sanity[0])
    _assert_equality(S[1], S_sanity[1])
    _assert_equality(S[2], S_sanity[2])
    
    # What np.linalg.svd returns, is the transposed of what we need in step 4 in the notes.
    V = Vh.T 
    det_vu = np.linalg.det(V @ U.T)
    I = np.eye(n_dims)
    I[-1, -1] = det_vu
    Rot = (V @ I) @ U.T
   
    # Step 5: Compute the optimal translation
    trans = Q_centroid - (Rot @ P_centroid)

    return Rot, trans

# ================================================================================================================
#                                           M A I N 
# ================================================================================================================

if __name__ == "__main__":
    print(">> Testing ", __file__)
    
    # ================================================================================================================
    #        Testing a small set of points
    #        WARNING: points sets with 2 elements may not return optimal R and t
    # ================================================================================================================
    
    # Random rotation and translation
    R = np.random.rand(3,3)
    t = np.random.rand(3,1)
    
    # make R a proper rotation matrix, force orthonormal
    U, S, Vt = np.linalg.svd(R)
    R = U@Vt
    
    # remove reflection
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1
       R = U@Vt 
       
    # number of points
    n = 10
    
    A = np.random.rand(3, n)
    B = R@A + t
    
    # Recover R and t
    ret_R, ret_t = get_optimal_rigid_motion(A.T, B.T, W=np.ones(A.shape[1]))

    # Compare the recovered R and t with the original
    B2 = (ret_R @ A).T + ret_t
    B2 = B2.T
    
    # Find the root mean squared error
    err = B2 - B
    err = err * err
    err = np.sum(err)
    rmse = np.sqrt(err/n)
    
    print("Points A")
    print(A)
    print("")
    
    print("Points B")
    print(B)
    print("")
    
    print("Ground truth rotation")
    print(R)
    
    print("Recovered rotation")
    print(ret_R)
    print("")
    
    print("Ground truth translation")
    print(t)
    
    print("Recovered translation")
    print(ret_t)
    print("")
    
    print("RMSE:", rmse)
    
    if rmse < 1e-5:
        print("Everything looks good!")
    else:
        print("Hmm something doesn't look right ...")
    
    