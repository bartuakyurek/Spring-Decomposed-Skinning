#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:15:07 2024

@author: bartu
"""
import torch
import numpy as np

def lerp(arr1, arr2, ratio):
    # TODO: Please make it more robust? Like asserting array shapes etc...
    return ((1.0 - ratio) * arr1) + (ratio * arr2)


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
    assert len(rot.shape) <= 2, f"Expected rotation vector to have (3, ) or (batch, 3) shape, got {rot.shape}."
    
    if len(rot.shape) == 1:
        assert rot.shape == (3, ), f"Expected rotation vector to have (3, ) or (batch, 3) shape, got {rot.shape}."
        rot = np.expand_dims(rot, 0)
        
    else:
        assert rot.shape[1] == 3, f"Expected rotation vector to have (3, ) or (batch, 3) shape, got {rot.shape}."
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
  