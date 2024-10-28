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


from ..utils.linalg_utils import(get_aligning_rotation, 
                                    translation_vector_to_matrix,
                                    angle_between_vectors_np,
                                    get_3d_scale)

def get_RST(src_segment, target_segment):
    # Step 0 - Declare source and target points
    assert src_segment.shape == (2,3) and target_segment.shape ==  (2,3)
    
    # Step 1 - Translate source to origin, apply the same translation to target
    offset = np.zeros((1, 3))
    offset[0,:] = src_segment[0]
    
    src_bone_space = src_segment - offset
    tgt_translated = target_segment - offset
    
    # Step 2 - Translate the translated target to origin, and save the translation 
    t = tgt_translated[0]
    tgt_bone_space = tgt_translated - t
    T = translation_vector_to_matrix(t)
    
    assert LA.norm(src_bone_space[0]) < 1e-20, "Expected bone space translations to land on origin."
    assert LA.norm(tgt_bone_space[0]) < 1e-20, "Expected bone space translations to land on origin."

    # Step 3 - Compute the rotation between source and target vectors 
    R = get_aligning_rotation(src_bone_space[1], tgt_bone_space[1], homogeneous=True)
    src_bs_rotated = R[:3,:3] @ src_bone_space[1]
    # Check if the angle between is practically zero (note that lower than 1e-6 can fail)
    angle = angle_between_vectors_np(src_bs_rotated, tgt_bone_space[1])
    assert angle < 1e-5, f"Expected the rotated bone space vector to be aligned with target at bone space. Got angle {angle} > 1e-5."
    
    # Step 4 - Compute the scaling of source by the norms ratio 
    # Note that we compute that after rotation, because rotation also scales when the vectors aren't normalized
    S = get_3d_scale(src_bs_rotated, tgt_bone_space[1], return_mat=True, homogeneous=True)
    
    # Step 5 - Combine all translation, rotation and scale in a single matrix
    # Note that compose_transform_matrix() wil not work as the same.
    M = T @ S @ R # (R)otate, (S)cale, (T)ranslate
    
    offs = translation_vector_to_matrix(offset)
    inv_offs = translation_vector_to_matrix(-offset)
    return offs @ M @ inv_offs
    
if __name__ == "__main__":
    print("Testing RST...")
    src_segment = np.array([
                            [1., 1., 0],
                            [2., 2., 0.1]
                            ])
    
    target_segment = np.array([
                            [3., 2., 0.1],
                            [5., 10.4, 0.02]
                            ])
    
    # Obtain transformations ----------------------------------------------------------------
    M = get_RST(src_segment, target_segment)
    
    # Convert homogeneous coordinates
    src_homo = np.append(src_segment, np.ones((2,1)),axis=-1) # (2,3) append (2,1) -> (2,4)
    
    # Get result
    src_transformed = M @ src_homo.T        # (4,4) @ (2,4).T -> (4,2)
    src_transformed = src_transformed.T[:,:3]  # (4,2).T -> (2,3)
    
    # Print Results -------------------------------------------------------------------------
    # Check if the obtained matrix can result:  M @ src = target
    print("Source: ", src_segment[0], src_segment[1])
    print("Target: ", target_segment[0], target_segment[1])
    
    begin = src_transformed[0]
    end = src_transformed[1]
    print("Result: ", np.round(begin, 4), 
                     np.round(end, 4))
    print("The result must match with the target.")

    
    
    
    