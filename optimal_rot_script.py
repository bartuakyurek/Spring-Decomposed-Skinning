#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 06:36:20 2024

@author: bartu
"""
import numpy as np

_DEBUG_ = True
_TOLERANCE_ = 1e-8

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
    assert W.shape == (n_points, 1), f"Weights matrix must have shape ({n_points}, 1). Provided has shape {W.shape}"

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

def __check_1D(arr):
    # Sanity check for an input array of either shape (N, ), (N, 1), or (1, N)
    arr_dims = len(arr.shape)
    is_2d = arr_dims == 2
    is_1d = arr_dims == 1
    
    assert is_2d or is_1d, "Batched input is not supported yet."
    
    if is_2d:
        assert arr.shape[0] == 1 or arr.shape[1] == 1, f"Array must be 1D. Provided has shape {arr.shape} "
    return arr_dims

def __equate_shapes(first_arr : np.ndarray, second_arr : np.ndarray, verbose=_DEBUG_):
    # Given two arrays that are supposedly in the same shape but
    # due to matrix manipulation on of their shape has gotten different,
    # this function aims to make them in the same shape.
    
    # Note: it only supports 1D shapes of (N, ) or (N, 1) and will raise error if not.
    # Note 2: this function could have been implemented with less repetition. 
   
    # WARNING: If the given arrays has shapes (1, N) and (1, N), or 
    # they have (N, ) and (N, ) the returned arrays will still be the same. 
    # For the rest of the code the returned arrays are (N, 1).
    
    if first_arr.shape == second_arr.shape:
        if verbose : print("INFO: No modification on array shapes had done.")  
        return first_arr, second_arr
    
    first_arr_dims = __check_1D(first_arr)
    second_arr_dims = __check_1D(second_arr)
    
    if first_arr_dims == 1 and second_arr_dims == 2:
        # Case 1: (N, ) vs. (N, 1)
        if second_arr.shape[1] == 1:
            return np.expand_dims(first_arr, axis=1), second_arr
        # Case 2: (N, ) vs. (1, N)
        if second_arr.shape[0] == 1:
            return np.expand_dims(first_arr, axis=1), second_arr.T
        
    elif first_arr_dims == 2 and second_arr_dims == 1:
        # Case 3: (N, 1) vs. (N,)
        if first_arr.shape[1] == 1:
            return first_arr, np.expand_dims(second_arr, axis=1)
        # Case 4:  (1, N) vs. (N,)
        if first_arr.shape[0] == 1:
            return first_arr.T, np.expand_dims(second_arr, axis=1)
    
    elif first_arr_dims == 2 and second_arr_dims == 2:
        # Case 5: (N, 1) vs. (1, N)
        if first_arr.shape[1] == 1 and second_arr.shape[0] == 1:
            return first_arr, second_arr.T
        # Case 6: (1, N) vs. (N, 1)
        if first_arr.shape[0] == 1 and second_arr.shape[1] == 1:
            return first_arr.T, second_arr
        
        else:
            raise Exception(f"Unexpected array shapes. Got {first_arr.shape} and {second_arr.shape}. Make sure to check the arrays are shape of (N,) (N,1) or (1,N).")
    
    else:
        raise Exception("Unexpected error occured. Are you sure your dimensionality assertions are correct?")

def __check_set_equality(first_set, second_set, tolerance=_TOLERANCE_):
    # Used as a sanity check for naive vs. fast implementations of the same linear algebra operations.
    first_set, second_set = __equate_shapes(first_set, second_set)
    assert first_set.shape == second_set.shape, f"Two sets must have equal shapes. Provided are {first_set.shape} and {second_set.shape}"
    
    diff = first_set - second_set
    total_diff = np.sum(np.abs(diff))
    
    assert total_diff < tolerance, f"Set equality check failed. Difference between sets are {total_diff} > tolerance"
    return total_diff

def compute_centroid(point_coords, weights):
    total_weights = np.sum(weights)
    weighted_point_sum = point_coords.T @ weights
    
    
    sanity_check = __naive_weighted_sum(point_coords, weights) + 1e-9
    __check_set_equality(sanity_check, weighted_point_sum)
    
    centroid = weighted_point_sum / total_weights
    return centroid

if __name__ == "__main__":
    # ============================ INPUTS =========================================
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
                    [0.5],
                    [0.5],
                    [1],
                    [1],
                    [1],
                ])
    # =============================================================================
    
    n_points, n_dims = __check_icp_set_shapes(P, Q, W)
    
    P_centroid = compute_centroid(P, W)
    Q_centroid = compute_centroid(Q, W)
    
    print(f"P has a centroid at \n{P_centroid}")
    print(f"Q has a centroid at \n{Q_centroid}")
    
    
