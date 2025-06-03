#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

IMPORTANT NOTES:
- Q: are SMPL's rotations in degrees or in radians?


"""

import numpy as np
from scipy.spatial.transform import Rotation

from .utils.sanity_check import _assert_normalized_weights
from .utils.linalg_utils import get_transform_mats_from_quat_rots, min_distance, normalize_weights

# ---------------------------------------------------------------------------------
# Helper routine to obtain posed mesh vertices
# ---------------------------------------------------------------------------------

def bind_weights(mesh_verts, skel_verts, method="Envelope", envelope=10.0): 
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
        
    # TODO: update parameter descriptions here
    
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
    
    assert len(skel_verts.shape) == 2, f"Expected skeleton vertices to have shape length 2 for (n_bones * 2, 3). Got shape {skel_verts.shape}." 
    assert skel_verts.shape[0] % 2 == 0, f"First dimension of skeleton vertices is expected to be an even number, as every bone has 2 joints. Got {skel_verts.shape[0]} \at dim 0."
    
    n_bones = int(skel_verts.shape[0] / 2) 
    n_verts = mesh_verts.shape[0]
    
    weights = None
    
    if method == "Envelope":
        weights = np.zeros((n_verts, n_bones))
        line_segment = np.empty((2,3))
        for i in range(n_verts):
            for j in range(n_bones):
                vert = mesh_verts[i]
                line_segment[0] = skel_verts[2*j]     # 
                line_segment[1] = skel_verts[2*j + 1] # 
                    
                dist = min_distance(vert, line_segment)
                
                if dist < envelope:
                    weights[i,j] = 1 / (dist + 1e-12) # Avoid divide by zero
                else:
                    weights[i,j] = 0.0                # For sanity
                    
        weight_sum = np.sum(weights, axis=0)
        weights = weights / weight_sum                # Normalize weights
        
        # Sanity check for to see if every vertex has total weights of 1.0
        _assert_normalized_weights(weights)
    else:
        print(f">> WARNING: Bind_weigths() for {method} is not implemented yet. Returning None.")
    return weights


def LBS_from_quat(V, W, abs_rot, abs_trans, use_normalized_weights=True):
    assert W.shape[0] == V.shape[0], f"Expected weights and verts to have same length at dimension 0, i.e. weights has shape (n_verts, n_bones)\
                                                 and verts has shape (n_verts, 3), got shape {W.shape} and {V.shape}."
    
    if use_normalized_weights:
        try: _assert_normalized_weights(W)
        except: W = normalize_weights(W)
    
    n_verts, n_bones = W.shape
    assert abs_rot.shape == (n_bones, 4), f"Expected absolute rotations in quaternions to have shape ({n_bones}, 4), got {abs_rot.shape}."
    assert abs_trans.shape == (n_bones, 3), f"Expected absolute translations to have shape ({n_bones}, 3), got {abs_trans.shape}."
        
    Ms = get_transform_mats_from_quat_rots(abs_trans, abs_rot)
    return LBS_from_mat(V, W, Ms)
    
def LBS_from_mat(V, W, M, use_normalized_weights=True):
    assert W.shape[0] == V.shape[0], f"Expected weights and verts to have same length at dimension 0, i.e. weights has shape (n_verts, n_bones)\
                                                 and verts has shape (n_verts, 3), got shape {W.shape} and {V.shape}."
    assert W.shape[1] == M.shape[0], f"Expected weights matrix columns dimension 1, to match with transformation matrix dimension 0. Got shapes {W.shape} and {M.shape}."
    if use_normalized_weights:
        try: _assert_normalized_weights(W)
        except: W = normalize_weights(W)
    
    n_verts, n_bones = W.shape
    V_homo = np.append(V, np.ones((n_verts,1)), axis=-1)

    # Pose vertices via matrix multiplications 
    V_homo = np.expand_dims(V_homo, axis=-1) # shape (n_verts, 4, 1) for broadcasting 
    weighted_transforms = np.tensordot(W, M, axes=(-1,0))  # shape (n_verts, 4, 4)
    V_posed_homo = np.matmul(weighted_transforms, V_homo)  # shape (n_verts, 4, 1)
    V_posed = V_posed_homo[:, :3, 0]
   
    assert V_posed.shape == V.shape
    return V_posed

def LBS_joints(J, M):
    assert J.shape[1] == 3
    assert len(J.shape) == 2
    n_verts = len(J)
    V_homo = np.append(J, np.ones((n_verts,1)), axis=-1)

    # Pose vertices via matrix multiplications 
    V_homo = np.expand_dims(V_homo, axis=-1) # shape (n_verts, 4, 1) for broadcasting 
    V_posed_homo = np.matmul(M, V_homo)  # shape (n_verts, 4, 1)
    V_posed = V_posed_homo[:, :3, 0]
   
    assert V_posed.shape == J.shape
    return V_posed

if __name__ == "__main__":
    print(">> Testing skinning.py...")
    
   
