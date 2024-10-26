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

try:
    from .utils.linalg_utils import get_aligning_rotation
except:
    from utils.linalg_utils import get_aligning_rotation


if __name__ == "__main__":

    v1 = np.array([1., 3., 0.])
    v2 = np.array([0., 1., 0.])
    R = get_aligning_rotation(v1, v2)
    rotated_v1 =  R @ v1
    print("v1: ", v1)
    print("v2: ", v2)
    print("R:\n", R)
    print("R @ v1 = ", rotated_v1)
    
    
    
    
    