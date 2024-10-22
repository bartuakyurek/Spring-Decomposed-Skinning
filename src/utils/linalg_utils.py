#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:15:07 2024

@author: bartu
"""
import torch
import numpy as np
from numpy import linalg as LA
from scipy.spatial.transform import Rotation


def min_distance(point, line_segment):
    """
    Get the shortest distance from a point to a line segment.

    Parameters
    ----------
    point : np.ndarray
        Vector that corresponds to the 3D location of the point, has shape (3,)
        or (3,1).
    line_segment : np.ndarray
        Array that holds to endpoints of a line segment, has shape (2,3)

    Returns
    -------
    shortest_distance : float
        Shortest distance from given point to given line segment.

    """
    assert type(point) is np.ndarray, f"Expected point type to be numpy ndarray, got {type(point)}."
    assert type(line_segment) is np.ndarray, f"Expected line_segment type to be numpy ndarray, got {type(line_segment)}."
    assert point.shape == (3,) or point.shape == (3,1), f"Expected point to have shape (3,) or (3,1). Got {point.shape}."
    assert line_segment.shape == (2,3), f"Expected line segment to have shape (2,3) got {line_segment.shape}."
    
    head, tail = line_segment
    shortest_distance = -999

    AB = head - tail   # Vector of the line segment
    BE = point - head  
    AE = point - tail
    
    AB_BE = np.dot(AB, BE) 
    AB_AE = np.dot(AB, AE)
    
    # Case 1, if closer to head
    if AB_BE > 0: 
        shortest_distance = LA.norm(point - head)

    # Case 2, if closer to tail
    elif AB_AE < 0: 
        shortest_distance = LA.norm(point - tail)
    
    # Case 3, if in between, find the perpendicular distance
    else:
        AB_norm = np.dot(AB, AB)        
        perp = np.cross(AB, AE)
        perp = perp / AB_norm  
        shortest_distance = LA.norm(perp)
    
    return shortest_distance

def get_midpoint(vec1, vec2):
    #return (vec2 - vec1) * 0.5 + vec1
    return lerp(vec1, vec2, 0.5) 
    
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

def get_transform_mats(trans, rotations):
    assert len(trans) == len(rotations), f"Given lists must have same lengths. Got {len(trans)} and {len(rotations)}"

    K = len(trans)
    M = np.empty((K, 4, 4))
    for i in range(K):
        rot = Rotation.from_quat(rotations[i])
        M[i] = compose_transform_matrix(trans[i], rot)
        
    return M
    
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
    
    assert type(rot) is Rotation, f"Expected Rotation class instance for rot variable, got {type(rot)}."
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


if __name__ == "__main__":
    print(">> Testing min_distance()...")
    
    # -------------------------------------------------------------------------
    point = np.array([2., 1.5, 0.])
    line_segment = np.array([[2., 1., 0.],
                             [2., 2., 0.]])
    
    dist = min_distance(point, line_segment)
    print("Min distance found: ", dist, " expected: 0.0")
    
    # -------------------------------------------------------------------------
    point = np.array([2., 0.5, 0.])
    line_segment = np.array([[2., 1., 0.],
                             [2., 2., 0.]])
    
    dist = min_distance(point, line_segment)
    print("Min distance found: ", dist, " expected: 0.5")
    # -------------------------------------------------------------------------
    
    point = np.array([1., 0.0, 0.])
    line_segment = np.array([[2., 1., 0.],
                             [2., 2., 0.]])
    
    dist = min_distance(point, line_segment)
    print("Min distance found: ", dist, f" expected: {np.sqrt(2)}")
    # -------------------------------------------------------------------------
    
    
    