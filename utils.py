#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:15:07 2024

@author: bartu
"""
import torch
import numpy as np

def batch_axsang_to_quats(rot):
    """
    Convert axis-angle rotation representations to quaternions.
    
    Parameters
    ----------
    rot : np.ndarray
        axis-angle rotation vector of shape (batch, 3).

    Returns
    -------
    np.ndarray
        quaternions representing the provided axis-angle rotations.

    """
    assert rot.shape[1] == 3
    assert type(rot) == np.ndarray
    
    roll = rot[:, 0] / 2.
    pitch = rot[:, 1] / 2.
    yaw = rot[:, 2] / 2.
    
    sin = np.sin
    cos = np.cos
    stack = np.stack
   
    qx = sin(roll) * cos(pitch) * cos(yaw)
    qy = cos(roll) * sin(pitch) * cos(yaw)
    qz = cos(roll) * cos(pitch) * sin(yaw)
    qw = cos(roll) * cos(pitch) * cos(yaw)
    
    return stack((qx, qy, qz, qw)).transpose()

