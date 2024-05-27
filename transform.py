#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:33:47 2024

@author: bartu
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class Transform:
    rotation_quat: np.ndarray
    rotation_mat: np.ndarray
    translation: np.ndarray
    
    

def to_quaternion(matrix: np.ndarray):
    pass


def to_matrix(quat: np.ndarray):
    pass