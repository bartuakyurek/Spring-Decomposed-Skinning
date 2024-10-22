#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:03:04 2024

@author: bartu
"""

import numpy as np

_DEBUG_ = True
_TOLERANCE_ = 1e-8


def _assert_normalized_weights(weights):
    weights_sum = np.sum(weights, axis=0) # For each vertex sum weights of all bones
    assert not np.any(weights_sum < 1-1e-12), "Expected weights for a vertex to sum up 1.0"
    assert not np.any(weights_sum > 1+1e-12), "Expected weights for a vertex to sum up 1.0"
   

def _check_or_convert_numpy(arr):
    if type(arr) is list:
        arr = np.array(arr)
    assert type(arr) == np.ndarray, f"Expected type np.ndarray, got {type(arr)}"
    return arr

def _assert_vec3(arr):
    # Verify that a given array has (3,1) or (3,) shape as a 3D vector.
    shape_len = _assert_unbatched(arr)
    if shape_len == 1:
        assert arr.shape[0] == 3, f"Expected shape (3, ) got {arr.shape}"
    elif shape_len == 2:
        assert arr.shape[0] == 3, f"Expected shape (3, 1) got {arr.shape}"
    else:
        assert False, ">> Unexpected error occured."
        
    return shape_len
    
def _assert_unbatched(arr):
    # Sanity check for an input array of either shape (N, ), (N, 1), or (1, N)
    arr_dims = len(arr.shape)
    is_2d = arr_dims == 2
    is_1d = arr_dims == 1
    
    assert is_2d or is_1d, "Batched input is not supported yet."
    
    if is_2d:
        assert arr.shape[0] == 1 or arr.shape[1] == 1, f"Array must be 1D. Provided has shape {arr.shape} "
    return arr_dims

def _is_equal(first_vec, second_vec, tolerance=_TOLERANCE_):
    # Robust implementation of X == Y check that could fail due to numerical
    # reasons even if X and Y are virtually the same. 
    assert first_vec.shape == second_vec.shape, f"Two sets must have equal shapes. Provided are {first_vec.shape} and {second_vec.shape}"
    
    diff = first_vec - second_vec
    total_diff = np.sum(np.abs(diff))
    
    return total_diff < tolerance

def _assert_equality(first_set, second_set, tolerance=_TOLERANCE_):
    # Used as a sanity check for naive vs. fast implementations of the same linear algebra operations.
    # The line above introduces too much hassle 
    #first_set, second_set = _equate_shapes(first_set, second_set)
    assert first_set.shape == second_set.shape, f"Two sets must have equal shapes. Provided are {first_set.shape} and {second_set.shape}"
    
    diff = first_set - second_set
    total_diff = np.sum(np.abs(diff))
    
    assert total_diff < tolerance, f"Set equality check failed. Difference between sets are {total_diff} > tolerance"
    return total_diff

def _equate_shapes(first_arr : np.ndarray, second_arr : np.ndarray, verbose=_DEBUG_):
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
    
    first_arr_dims = _assert_unbatched(first_arr)
    second_arr_dims = _assert_unbatched(second_arr)
    
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
        
if __name__ == '__main__':
    print(f">> Running debug tests for {__file__}...\n")
    
    def test_assert_vec3():
        shape_3_vec = np.array([2,3,4])
        _assert_vec3(shape_3_vec)
        
        shape_1_3_vec = np.array([shape_3_vec])
        shape_3_1_vec = shape_1_3_vec.T
        _assert_vec3(shape_3_1_vec)
        
        try:
            _assert_vec3(shape_1_3_vec)
            print(">> ERROR: Expected failure case failed to fail!")
        except:
            print(">> Caught expected failure at (1,3) shape vector.")
            
            
        non_vec3 = np.empty((np.random.randint(50), np.random.randint(50)))
        try:
            _assert_vec3(non_vec3)
            print(">> ERROR: Expected failure case failed to fail!")
        except:
            print(f">> Caught expected failure at {non_vec3.shape} shape vector.")
    
    def test_is_equal():
        a = np.random.rand(30)
        b = np.random.rand(30)
        c = b
        
        if _is_equal(a, b):
            print(">> ERROR: Expected failure case failed to fail!")
        else:
            print(">> Caught expected failure when a != b")
         
        if not _is_equal(b, c):
            print(">> ERROR: Expected b=c to be True, got False")
            
    n_tests = 10
    for i in range(n_tests):
        test_assert_vec3()
        test_is_equal()

