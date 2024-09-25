#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 06:36:20 2024

@author: bartu
"""

# ================================================================================================================
#                                           IMPORTS AND GLOBALS
# ================================================================================================================

import numpy as np
from sanity_check import __check_equality, __equate_shapes

_SPACE_DIMS_ = 3

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
    __check_equality(sanity_check, weighted_point_sum)
    
    centroid = weighted_point_sum / total_weights
    assert centroid.shape == (_SPACE_DIMS_, )
    
    return centroid

def get_optimal_rigid_motion(P, Q, W):
    
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
    __check_equality(S, sanity_check)
    
    # Step 4: Compute the SVD and the optimal rotation
    U, Sigma, V = np.linalg.svd(S, full_matrices=True)
    
    assert U.shape == (n_dims, n_dims)
    assert Sigma.shape == (n_dims, )
    assert V.shape == (n_dims, n_dims)
    
    S_sanity = U @ (np.diag(Sigma) @ V)
    assert S_sanity.shape == S.shape, f"Sanity Check Failed! SVD Reconstructed matrix has shape {S_sanity.shape}, expected {S.shape}"
    __check_equality(S[0], S_sanity[0])
    __check_equality(S[1], S_sanity[1])
    __check_equality(S[2], S_sanity[2])
    
    # What np.linalg.svd returns, is the transposed of what we need in step 4 in the notes.
    tmp_V, tmp_U = V, U
    U = tmp_V
    V = tmp_U.T
    
    det_vu = np.linalg.det(V @ U.T)
    I = np.eye(n_dims)
    I[-1, -1] = det_vu
    Rot = V @ I @ U.T
    
    # Step 5: Compute the optimal translation
    trans = Q_centroid - (Rot @ P_centroid)

    return Rot, trans

# ================================================================================================================
#                                           M A I N 
# ================================================================================================================

if __name__ == "__main__":
    print(">> Testing ", __file__)
    from matplot_viewer import Matplot_Viewer
    
    # ================================================================================================================
    #        Testing trivial cases: rigid motion between two line segments
    # ================================================================================================================

    # Case #1: Same line segments, see if you transform you'll still get the same line segment.

    line_segment = np.array([
                    [0.5, 3.0, 0.5],
                    [2.0, 3.0, 0.0]
                ])
    weights_segment = np.array([1.0, 1.0])
    #weights_segment /= np.sum(weights_segment)
    
    print(">> Trivial case 1: rigid motion between two identical line segments")
    Rot, trans = get_optimal_rigid_motion(line_segment, line_segment, weights_segment)
    print("R\n", Rot)
    print("t ", trans)
    transformed_line_segment = (line_segment @ Rot) + trans
    print(__check_equality(line_segment, transformed_line_segment))
    

    # Case #2: Two different line segments, see if you transform first segment, you'll get the second segment 
    print(">> Trivial case 2: rigid motion from one segment to another line segment")
    another_line_segment = np.array([
                                        [0.5, 0.5, 3.0],
                                        [0.0, 3.0, 2.0]
                                    ])
    Rot, trans = get_optimal_rigid_motion(line_segment, another_line_segment, weights_segment)
    transformed_line_segment = (line_segment @ Rot) + trans
    print(__check_equality(line_segment, transformed_line_segment))

    # ================================================================================================================
    #        Testing a small set of points
    # ================================================================================================================

    P = np.array([
                    [0.5, 3.0, 0.5],
                    [2.0, 3.0, 0.0],
                    [1.0, 2.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ])
    
    
    Q = np.array([
                    [0.67, 2.0, 0.5],
                    [2.0, 3.0, 0.12],
                    [1.0, 1.5, 1.0],
                    [1.1, 1.3, 0.0],
                    [0.0, 1.0, 0.23],
                ])
    
    W = np.array([
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.0],
                ]) 
    
    
    
    