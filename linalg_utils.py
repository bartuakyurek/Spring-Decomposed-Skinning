#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:15:07 2024

@author: bartu
"""
import torch
import numpy as np
from scipy.spatial.transform import Rotation


def lerp(arr1, arr2, ratio):
    # TODO: Please make it more robust? Like asserting array shapes etc...
    return ((1.0 - ratio) * arr1) + (ratio * arr2)

def get_rotation_mats(rot_quats):
    
    assert len(rot_quats.shape) == 2 and rot_quats.shape[1]==4, f"Expected to rotation quaternions to have shape (n_rotations, 4), got {rot_quats.shape}."
    R_mats = []
    for i in range(rot_quats.shape[0]):
        rot = Rotation.from_quat(rot_quats[i])
        R_mats.append(rot.as_matrix())
    return np.array(R_mats)

def compose_transform_matrix(trans_vec, rot : Rotation ):
    """
    Compose a transformation matrix given the translation vector
    and Rotation object.

    Parameters
    ----------
    trans_vec : np.ndarray or list
        3D translation vector to be inserted at the last column of 4x4 matrix.
    rot : scipy.spatial.transform.Rotation
        Rotation object of scipy.spatial.transform. This is internally
        converted to 3x3 matrix to place in the 4x4 transformation matrix.

    Returns
    -------
    M : np.ndarray
        Transformation matrix composed by 3x3 rotation matrix and 3x1 translation
        vector, with the last row being [0,0,0,1].
    """
    
    if type(trans_vec) is list:
        trans_vec = np.array(trans_vec)
    
    if trans_vec.shape == (3,1):
        trans_vec = trans_vec[:,0]
        
    assert type(trans_vec) == np.ndarray, f"Expected translation vector to have type np.ndarray, got {trans_vec.shape}."
    assert trans_vec.shape == (3, ), f"Expected translation vector to have shape (3,) got {trans_vec.shape}"
    
    rot_mat = rot.as_matrix()
    
    # Convert absolute rotations and translations into a single transformation matrix
    M = np.zeros((4,4))
    M[:3, :3] = rot_mat     # Place rotation matrix
    M[:3, -1] = trans_vec # Place translation vector
    M[-1, -1] = 1.0  
    # Sanity check that M transformation matrix last row must be [0 0 0 1] 
    assert np.all(M[-1] == np.array([0.,0.,0.,1.])), f"Unexpected error occured at {M}."
    return M

"""
def batch_axsang_to_quats(rot):
   
    Convert axis-angle rotation representations to quaternions.
    
    Parameters
    ----------
    rot : np.ndarray
        axis-angle rotation vector of shape (batch, 3).

    Returns
    -------
    np.ndarray
        quaternions representing the provided axis-angle rotations.

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
"""
  