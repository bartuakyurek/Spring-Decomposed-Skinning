#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:01:59 2024

@author: bartu
"""
import numpy as np

def perpendicular_vector(v, is_normalized=True):
    r""" Finds an arbitrary perpendicular vector to *v*."""
    # for two vectors (x, y, z) and (a, b, c) to be perpendicular,
    # the following equation has to be fulfilled
    #     0 = ax + by + cz

    # x = y = z = 0 is not an acceptable solution
    if v[0] == v[1] == v[2] == 0:
        raise ValueError('zero-vector')

    # If one dimension is zero, this can be solved by setting that to
    # non-zero and the others to zero. Example: (4, 2, 0) lies in the
    # x-y-Plane, so (0, 0, 1) is orthogonal to the plane.
    if v[0] == 0:
        return np.array([1, 0, 0])
    if v[1] == 0:
        return np.array([0, 1, 0])
    if v[2] == 0:
        return np.array([0, 0, 1])

    # arbitrarily set a = b = 1
    # then the equation simplifies to
    #     c = -(x + y)/z
    perp_vector = np.array([1, 1, -1.0 * (v[0] + v[1]) / v[2]]) 
    
    if is_normalized:
        return perp_vector / np.linalg.norm(perp_vector) 
    else:
        return perp_vector
    
