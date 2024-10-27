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

from .sanity_check import _assert_normalized_weights

def normalize_weights(weights):
    """
    Make sure the weights per row (i.e. vertex) sums up to 1.0.
    WARNING: This is NOT the standard way of data normalization (i.e. shifting
                                                                 mean and scaling
                                                                 std.)
    Parameters
    ----------
    weights : np.ndarray
        Vertex-bone binding weights have shape (n_verts, n_bones).

    Returns
    -------
    normalized_weights : np.ndarray
        Has shape (n_verts, n_bones), where every row sums up to 1.0

    """
    assert len(weights.shape) == 2, f"Expected weights to have shape length of 2, got shape {weights.shape}."
    
    row_sum = np.sum(weights, axis=1, keepdims=True) # For each vertex sum weights of all bones
    assert len(row_sum) == len(weights), f"Expected to sum over all vertices, got shape mismatch with sum {row_sum.shape} and weights {weights.shape} at dimension 0."
    
    normalized_weights = weights / (row_sum + 1e-30)
    _assert_normalized_weights(normalized_weights)
    return normalized_weights

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

def translation_vector_to_matrix(trans):
    """
    Given the translation vector [x, y, z], 
    retrieve the translation matrix:
                                    [1, 0, 0, x]
                                    [0, 1, 0, y]
                                    [0, 0, 1, z]
                                    [0, 0, 0, 1]
    Parameters
    ----------
    trans : np.ndarray
        A 3D vector to represent translation, has shape (3,1) or (3,)

    Returns
    -------
    mat : np.ndarray
        A homogeneous matrix that translates a vector when applied, has shape (4,4)
    """
    if trans.shape == (3,1): trans = trans[:,0]
    elif trans.shape == (1,3): trans = trans[0,:]
    assert trans.shape == (3,), f"Expected translation vector to be 3D vector, got shape {trans.shape}."
    
    mat = np.eye(4)
    mat[0:3,-1] = trans
    return mat

def get_transform_mats_from_quat_rots(trans, rotations):
    assert len(trans) == len(rotations), f"Given lists must have same lengths. Got {len(trans)} and {len(rotations)}"

    K = len(trans)
    M = np.empty((K, 4, 4))
    for i in range(K):
        rot = Rotation.from_quat(rotations[i])
        M[i] = compose_transform_matrix(trans[i], rot)
        
    return M
    
def scale_this_matrix(mat, scale):
    """
    Scale a square matrix with given 3x3 or 4x4 homogeneous matrix 
    and the scaling vector or scalar.

    Parameters
    ----------
    mat : np.ndarray
        The matrix to be scaled, has shape (3,3) or (4,4).
        Note that in (4,4) case it's expected to be a homogenous matrix
        so that the last entry, i.e. 1.0 at the end, won't be scaled.
        
    scale : np.ndarray or float
        Scales the matrix by multiplying the diagonal entries.

    Returns
    -------
    mat : np.ndarray
        Scaled version of the given matrix, has the same shape.

    """
    assert mat.shape == (3,3) or mat.shape == (4,4), f"Expected 3x3 or 4x4 matrix to be scaled. Got {mat.shape}."
    
    if type(scale) is float or type(scale) is np.float64:
        scale = np.repeat(scale, 3, axis=0)
    else:  
        assert len(scale) == 3, f"Expected scale to be 3D vector or a scalar, got {scale.shape}."
        
    mat[0, 0] *= scale[0]
    mat[1, 1] *= scale[1]
    mat[2, 2] *= scale[2]
    return mat

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
    
    if LA.norm(src_vec) > 1. + 1e-18 or LA.norm(target_vec) > 1. + 1e-18:
        print(">> WARNING: Given vectors aren't normalized. The rotation \
               may cause scaling in the transformed vector. Please normalize \
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

def compose_transform_matrix(trans, rot, scale=None, rot_is_mat=False):
    """
    Compose a transformation matrix given the translation vector
    and Rotation object.

    Parameters
    ----------
    trans_vec : np.ndarray or list
        3D translation vector to be inserted at the last column of 4x4 matrix.
    rot : scipy.spatial.transform.Rotation or np.nd.array
        Rotation object of scipy.spatial.transform. This is internally
        converted to 3x3 matrix to place in the 4x4 transformation matrix.
    
    scale : np.ndarray or float
        Scales the matrix by multiplying the diagonal entries.
        
    rot_is_mat : bool
        Indicates if the provided rotation is a matrix if set True. If set
        False, it'll be expected to be a Rotation class instance.
    Returns
    -------
    M : np.ndarray
        Transformation matrix composed by 3x3 rotation matrix and 3x1 translation
        vector, with the last row being [0,0,0,1].
    """
   
    if type(trans) is list:
        trans = np.array(trans)
    
    if trans.shape == (3,1):
        trans = trans[:,0]
    
    assert type(trans) == np.ndarray, f"Expected translation vector to have type np.ndarray, got {trans.shape}."
    assert trans.shape == (3, ), f"Expected translation vector to have shape (3,) got {trans.shape}"
    
    if not rot_is_mat:
        assert type(rot) is Rotation, f"Expected Rotation class instance for rot variable, got {type(rot)}."
        rot_mat = rot.as_matrix()
    else:
        assert rot.shape == (3,3), f"Expected rotation matrix to have shape (3,3), got {rot.shape}"
        rot_mat = rot
        
    # Convert absolute rotations and translations into a single transformation matrix
    M = np.zeros((4,4))
    M[:3, :3] = rot_mat     # Place rotation matrix
    M[:3, -1] = trans # Place translation vector
    M[-1, -1] = 1.0  
    
    # Apply scale
    if scale is not None:
        M = scale_this_matrix(M, scale)
        
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
    
    
    