#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file is created to infer transformations, given the bone locations at the
rest pose and the updated bone locations at the current frame. Note that this
is not a typical Inverse Kinematics module, the name might be misleading. 

The transformations are inferred via mapping line segments from one frame 
to another. The available options are:
                                        - SVD based optimal rigid motion
                                        - RST based affine transformations

"""
import numpy as np
from scipy.spatial.transform import Rotation

from ..utils.linalg_utils import get_midpoint, compose_rigid_transform_matrix
from .optimal_rigid_motion import get_optimal_rigid_motion
from .rst_map import get_RST

def _get_bone_SVD_optimal_rigid(bone_rest_tuple, bone_cur_tuple):
    """
    Get the optimal rigid motion (rotation and translation) of a single
    bone given the rest locations and current locations of the endpoints.
    
    WARNING: It doesn't give good results when directly used in the skinning
    matrices. It can map the bones from rest pose to current pose however
    when used on a volume, it doesn't produce consistent results.
    
    Parameters
    ----------
    bone_rest_tuple : tuple or np.ndarray
        Holds the endpoint locations of the bones in a (bone_start, bone_end)
        manner at the rest pose. This will be used as a source points
        that should be mapped to the target points.
    bone_cur_tuple : tuple or np.ndarray
        Holds the endpoint locations of the bones in a (bone_start, bone_end)
        manner at the current frame. These are the target points where we
        like to get when we transform the rest pose bone.

    Returns
    -------
    R_mat : np.ndarray
        Rotation matrix has shape (3,3).
    t : np.ndarray
        Translation vector has shape (3,).
    
    """
    source_points = np.empty((3,3))
    target_points = np.empty((3,3))
    
    source_points[0] = bone_rest_tuple[0] # head
    source_points[2] = bone_rest_tuple[1] # tail
    source_points[1] = get_midpoint(source_points[2], source_points[0])
    
    target_points[0] = bone_cur_tuple[0] # head
    target_points[2] = bone_cur_tuple[1] # tail
    target_points[1] = get_midpoint(target_points[0], target_points[2])
    
    R_mat, t = get_optimal_rigid_motion(source_points, target_points)
    return R_mat, t

def _get_bone_RST(bone_rest, bone_cur):
    if type(bone_rest) is list: bone_rest = np.array(bone_rest)
    if type(bone_cur) is list: bone_cur = np.array(bone_cur)
    M = get_RST(bone_rest, bone_cur)
    return M

def _get_bone_trans_mat(bone_rest, bone_cur):
    assert bone_rest.shape == (2,3)
    assert bone_cur.shape == (2,3)
    
    M = np.eye(4)
    
    endtip_diff = bone_cur[-1] - bone_rest[-1]
    
    M[0:3,-1] = endtip_diff
    return M
    

def get_absolute_transformations(rest_locations, 
                                 posed_locations, 
                                 return_mat=False, algorithm="RST"):
    """
    Parameters
    ----------
    # TOOD: please make your data structure consistent. posed_locations
    has shape (n_bones * 2)... instead it can have (n_bones, 2) shape
    for readability.
    
    return_mat : bool
        If True, return the rotation and translation as a whole 4x4 matrix
        that has rotation and translation inside. If False, return a quaternion
        for absolute rotation and a 3D vector for absolute translation.
    
    algorithm : str
        Choice of algorithm to compute transformations. 
        Available options are:
            - "RST" : Computes Rotation-Scale-Translation matrix
            - "SVD" : Computes SVD-based optimal rigid motion (see related .py script)
                      WARNING: Do not use it to directly feed to the skinning algorithm
                               because it causes volume collapse.
            - "T" : Computes plain 4x4 translation matrices at the tip of the bones
            
    Returns
    -------
    abs_rot_quats: np.ndarray
        Absoulute rotation as quaternions 
    abs_trans: np.ndarray
        3D vectors for absolute translation
    """        
    
    n_bones = int(len(posed_locations)/2) 
    
    abs_rot_quats = np.empty((n_bones, 4))
    abs_trans =  np.empty((n_bones, 3))
    abs_M = np.empty((n_bones, 4, 4))
    
    # Select the algorithm to compute bone matrices
    get_bone_mats = None
    if algorithm == "RST": assert return_mat, "Expected return_mat=True if the algorithm is set to RST as RST returns a single transformation matrix."
   
    if algorithm == "RST": get_bone_mats = _get_bone_RST
    elif algorithm == "SVD": get_bone_mats = _get_bone_SVD_optimal_rigid
    elif algorithm == "T" : get_bone_mats = _get_bone_trans_mat
    else: raise ValueError(f"Unexpected algorithm type: {algorithm}.")
    
    # Loop over rest bones
    for i in range(n_bones):
        # Get bone matrices
        bone_rest = np.array([rest_locations[2*i], rest_locations[2*i+1]]) #
        bone_cur = np.array([posed_locations[2*i], posed_locations[2*i+1]]) # 
        
        if algorithm == "SVD": 
            R_mat, t = get_bone_mats(bone_rest, bone_cur)
            if return_mat:  # Save transforms as a 4x4 matrix
                abs_M[i] = compose_rigid_transform_matrix(t, R_mat, rot_is_mat=True)
            else:     
                # Convert matrices to quaternions
                abs_trans[i] = t
                rot = Rotation.from_matrix(R_mat)
                abs_rot_quats[i] = rot.as_quat()     
        
        else: # RST or T
            abs_M[i] = get_bone_mats(bone_rest, bone_cur)
            
    if return_mat: 
        return abs_M
    else:
        return abs_rot_quats, abs_trans