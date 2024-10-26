#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 11:18:55 2024

Compute the transformation matrix, that Rotates, Scales, and Transforms (RST)
the given source line segment to the target line segment.

@author: bartu
"""
import numpy as np
from numpy import linalg as LA


def get_aligning_rotation(src_vec, target_vec):
    """
    Align the source vector to the target vector via rotation.
    
    WARNING: If the given vectors does share the same norm, then the
    rotation will cause scaling as well. Also note that this function
    fails if the two vectors are opposite of each other.
    
    DISCLAIMER: This function is adopted from
    https://gist.github.com/kevinmoran/b45980723e53edeb8a5a43c49f134724

    Parameters
    ----------
    src_vec : np.ndarray
        Source vector to be transformed, has shape (3,).
    target_vec : np.ndarray
        Target vector to align the source vector, has shape (3,).

    Returns
    -------
    result : np.ndarray
        Rotation matrix has shape (3,3).

    """
    assert src_vec.shape == (3,) and target_vec.shape == (3,)
    
    if np.abs(LA.norm(v1) - LA.norm(target_vec)) > 1e-18:
        print("WARNING: Given vectors do not share the same norm. The rotation \
               may cause scaling in the transformed vector. Please normalize   \
               the vectors first.")
              
    cosA = np.dot(src_vec, target_vec)
    assert not cosA == -1, "Cannot evaluate vectors whose cosine is -1."
    
    k = 1. / (1. + cosA)
    axis = np.cross(src_vec, target_vec)

    row1 = (axis[0] * axis[0] * k) + cosA, (axis[1] * axis[0] * k) - axis[2], (axis[2] * axis[0] * k) + axis[1],
    row2 = (axis[0] * axis[1] * k) + axis[2], (axis[1] * axis[1] * k) + cosA, (axis[2] * axis[1] * k) - axis[0]
    row3 = (axis[0] * axis[2] * k) - axis[1], (axis[1] * axis[2] * k) + axis[0], (axis[2] * axis[2] * k) + cosA
    
    result = np.vstack((row1, row2, row3), dtype=float)
    assert result.shape == (3,3)
    return result

if __name__ == "__main__":

    v1 = np.array([1., 3., 0.])
    v2 = np.array([0., 1., 0.])
    R = get_aligning_rotation(v1, v2)
    rotated_v1 =  R @ v1
    print("v1: ", v1)
    print("v2: ", v2)
    print("R:\n", R)
    print("R @ v1 = ", rotated_v1)
    
    
    
    
    