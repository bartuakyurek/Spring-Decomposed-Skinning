#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

IMPORTANT NOTES:
- Q: are SMPL's rotations in degrees or in radians?


"""
import igl
import numpy as np
from scipy.spatial.transform import Rotation

from skeleton import Skeleton
from sanity_check import _assert_normalized_weights
from linalg_utils import get_transform_mats, compose_transform_matrix

# ---------------------------------------------------------------------------------
# Helper routine to obtain posed mesh vertices
# ---------------------------------------------------------------------------------
def get_skel_points(skeleton, theta, trans, degrees, exclude_root, combine_points=True):
   
    bone_locations = skeleton.pose_bones(theta, trans, degrees=degrees, exclude_root=exclude_root)
    
    skel_mesh_points = bone_locations
    if combine_points:
        skel_mesh_points = np.reshape(bone_locations, (-1,3)) # Combine all the 3D points into one dimension
   
    return skel_mesh_points

def _get_mesh_points(mode):
    
    if mode == "Rigid":
        posed_mesh_points = None
    else:
        posed_mesh_points = None
        print("Warning: skinning not implemented yet...")
    return posed_mesh_points

def bind_weights(mesh_verts, skel_verts, method="Euclidean"): 
    """
    Find the set of weights that will be the
    skinning weights for the provided mesh vertices. Usually this method
    is called at the T-pose, such that skinning function can take these 
    T-pose binding weights to map the mesh from T-pose to the desired 
    deformation. 

    Parameters
    ----------
    mesh_verts : np.ndarray
        Vertices of the mesh, has shape (n_verts, 3).
    skel_verts : np.ndarray
        Endpoint locations of the skeleton that has shape (n_bones * 2, 3).
        It's assumed every bone has 2 endpoints such that every consecutive
        [i][i+1] where i is an even number, represents a bone.
        
        Note that you need to convert SMPL joints to Skeleton class in your
        pipeline before using this method; otherwise the input data will not
        match with the implementation.

    Returns
    -------
    weights : np.ndarray.
        Binding weights has shape (n_verts, n_bones). Every row belongs to
        a vertex in the mesh, and every column is dedicated to the bones, 
        (be careful, bones not joints, those are used interchangibly in skeletal 
        animation but in this pipeline every bone has 2 joints). 
        
        Every entry w=(v, b) is the weight w that is bound to vertex v and bone b. 
        Meaning that vertex v, will inherit the transformation of bone b at
        w amount. Usually w is in range [0.0, 1.0] for smooth vertex blending.
        However setting it outside of this range is still theoretically possible.
    """
    assert type(mesh_verts) == np.ndarray
    assert type(skel_verts) == np.ndarray
    # TODO change shapes for these asserts when we hold skeleton data as (n_bones, 2, 3)
    assert len(skel_verts.shape) == 2, f"Expected skeleton vertices to have shape length 2 for (n_bones * 2, 3). Got shape {skel_verts.shape}." 
    assert skel_verts.shape[0] % 2 == 0, f"First dimension of skeleton vertices is expected to be an even number, as every bone has 2 joints. Got {skel_verts.shape[0]} \at dim 0."
    
    n_bones = int(skel_verts.shape[0] / 2) # TODO: should be changed if we change to shape (n_bones, 2, 3) 
    n_verts = mesh_verts.shape[0]
    
    weights = None
    if method == "Euclidean":
        
        weights = np.zeros((n_verts, n_bones))
    
        # Sanity check for to see if every vertex has total weights of 1.0
        _assert_normalized_weights(weights)
    else:
        print(f">> WARNING: Bind_weigths() for {method} is not implemented yet. Returning None.")
    return weights


def LBS(V, W, abs_rot, abs_trans):
    assert W.shape[0] == V.shape[0], f"Expected weights and verts to have same length at dimension 0, i.e. weights has shape (n_verts, n_bones)\
                                                 and verts has shape (n_verts, 3), got shape {W.shape} and {V.shape}."
    n_verts, n_bones = W.shape
    assert abs_rot.shape == (n_bones, 4), f"Expected absolute rotations in quaternions to have shape ({n_bones}, 4), got {abs_rot.shape}."
    assert abs_trans.shape == (n_bones, 3), f"Expected absolute translations to have shape ({n_bones}, 3), got {abs_trans.shape}."
        
    Ms = get_transform_mats(abs_trans, abs_rot)
    V_homo = np.append(V, np.ones((n_verts,1)), axis=-1)

    # Pose vertices via matrix multiplications 
    V_homo = np.expand_dims(V_homo, axis=-1) # shape (n_verts, 4, 1) for broadcasting 
    weighted_transforms = np.tensordot(W, Ms, axes=(-1,0)) # shape (n_verts, 4, 4)
    V_posed_homo = np.matmul(weighted_transforms, V_homo)  # shape (n_verts, 4, 1)
    V_posed = V_posed_homo[:, :3, 0]
   
    assert V_posed.shape == V.shape
    return V_posed
    


def skinning(verts, abs_rot, abs_trans, weights, skinning_type="LBS"):
    """
    Deform the vertices by provided transformations and selected skinning
    method.

    Parameters
    ----------
    verts : np.ndarray
        Vertices to be deformed by provided transformations.
    abs_rot : np.ndarray
        Absolute rotation transformation quaternions of shape (n_bones, 4)
    abs_trans : np.ndarray
        Absolute translation vec3 of shape (n_bones, 3)
    weights : np.ndarray
        Binding weights between vertices and bone transformation of shape
        (n_verts, n_bones). Note that to deform the skeleton, set weights of 
        shape (n_bones * 2, n_bones) with 1.0 weights for every couple rows.
        e.g. for a two bones it is 
                                    [[1.0, 0.0],
                                     [1.0, 0.0],
                                     [0.0, 1.0],
                                     [0.0, 1.0]] 
    skinning_type : str, optional
        DESCRIPTION. The default is "LBS".

    Returns
    -------
    V_deformed : np.ndarray
        Deformed vertices of shape (n_verts, 3)
    """
    
   
    if skinning_type == "LBS" or skinning_type == "lbs":
        # Deform vertices based on Linear Blend Skinning
        return LBS(V         = verts, 
                   W         = weights,
                   abs_rot   = abs_rot, 
                   abs_trans = abs_trans)
    else:
        raise ValueError(f">> ERROR: This skinning type \"{skinning_type}\" \
                         is not supported yet.")

if __name__ == "__main__":
    print(">> Testing skinning.py...")
    
   
