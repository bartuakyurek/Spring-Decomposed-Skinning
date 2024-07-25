#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:01:08 2024

@author: bartu
"""
import torch
import igl
import numpy as np
from scipy.spatial.transform import Rotation as R

def translate_affine(T, t):
    assert T.shape[-1] == 4
    assert T.shape[-2] == 4
    assert t.shape[-1] == 3
    assert len(T.shape) == 3 or len(T.shape) == 2
    assert len(T.shape) == len(t.shape) + 1 
    
    T_new = T
    if len(T.shape) == 3:
        assert T.shape[0] == t.shape[0]
        T_new[:, :3,-1] += t[:]
    else:
        T_new[:3, -1] += t
    
    return T_new
    

def _get_affine(R, t):
  
    affine = np.zeros((4,4))
    affine[-1,-1] = 1.
    affine[:3, :3] = R
    affine[:3, -1] = t
    
    return affine

def forward_kin(joint_pos, 
                joint_edges, 
                joint_parents, 
                relative_rot, # axis angle representation
                relative_trans=None):
    
    #relative_rot_q = batch_axsang_to_quats(relative_rot)
    r = R.from_euler('xyz', relative_rot, degrees=False)
    relative_rot_q = r.as_quat()
    relative_rot_q = np.array(relative_rot_q, order='F') # order is for libigl's implementation
    
    if relative_trans is None:
        relative_trans = np.zeros_like(relative_rot_q)[:,:3]
        
    absolute_rot, absolute_trans = igl.forward_kinematics(joint_pos, 
                                                          joint_edges, 
                                                          joint_parents, 
                                                          relative_rot_q,
                                                          relative_trans)
    
    return absolute_rot, absolute_trans

def LBS(V, W, J, JE, theta):
    
    P = igl.directed_edge_parents(JE)
    abs_rot, abs_t = forward_kin(J, JE, P, theta)
    
    r = R.from_quat(abs_rot)
    R_mat = r.as_matrix()
    #R_mat = np.array(quat2mat(torch.from_numpy(abs_rot)))
   
    num_bones = abs_t.shape[0]
    V_posed = np.zeros_like(V)
    for vertex in range(V_posed.shape[0]):
        for bone in range(num_bones):
           
            # TODO: remove for loop!
            if W[vertex, bone] < 1e-15:
                continue # Weight is zero, don't add any contribution
            
            
            ####
            """
            aff = np.eye(3)
            q = np.array([0.053427851090398007,
                          -0.012677099504295151,
                          -0.28733881120315019,
                          0.95625371290906935])
            
            r_tmp = R.from_quat(q)
            aff2 = r_tmp.apply(aff)
            """
            ####
            
            affine = np.eye(4)
            affine[0:3, -1] += abs_t[bone]
            affine[0:3,0:3] = R_mat[bone] #.transpose() 
            
            V_homo = np.zeros((4))
            V_homo[:3] = V[vertex]
            V_homo[-1] = 1.
            
            v_tmp = affine @ V_homo # .transpose()
            v_tmp *= W[vertex, bone]
            
            V_posed[vertex] +=  v_tmp[:3]
            
            """
            # TODO: J[bone] means the bone_tail, but you need bone_head location!
            # TODO: maybe learn how to rotate a matrix with a quat as in Libigl's.
            V_bone_space = V[vertex] - J[bone] #abs_t[bone]
            V_bone_space_rotated = np.matmul(R_mat[bone].transpose(), V_bone_space)
            V_world_space = V_bone_space_rotated + J[bone] #abs_t[bone]
            
            # TODO: WHICH ONE? TRANSPOSED OR UNTRANSPOSED?????????????????????????
            V_posed[vertex] += W[vertex, bone] * V_world_space
            
            """
            
            """
            transform_mat = _get_affine(R_mat[bone], abs_t[bone])
            V_homo = np.insert(V[vertex], 3, 1.0)
            
            # TODO: WHICH ONE? TRANSPOSED OR UNTRANSPOSED?????????????????????????
            V_new_homo = np.matmul(transform_mat, V_homo) #.transpose() @ V_homo
            V_new_homo *=  W[vertex, bone]
            
            V_posed[vertex] += V_new_homo[:3]
            """
            # TODO: put the rotation and translation into affine matrix
            # and then apply affine transformation!
    return V_posed

def inverse_LBS(V_posed, W, J, JE, theta):
    return LBS(V_posed, W, J, JE, -theta)

if __name__ == "__main__":
    print(">> Testing transformations.py...")
    

    # Load the mesh data
    with np.load("./results/skinning_T_pose_data.npz") as data:
        V = data['arr_0'] # Vertices, V x 3
        F = data['arr_1'] # Faces, F x 3
        J = data['arr_2'] # Joint locations J x 3
        theta = data['arr_3'] # Joint relative rotations J x 3
        kintree = data['arr_4'] # Skeleton joint hierarchy (J-1) x 2
        W = data['arr_5']
    
    
    # Modify the kintree by adding a ghost joint location to origin
    # to be compatible with libigl's forward kinematics
    J_modified = np.insert(J, J.shape[0], [0,0,0], axis=0)
    kintree_modified = np.insert(kintree, kintree.shape[0], [24, 0], axis=0)
    
    theta = np.array(theta) # TODO: stick to either torch or numpy, dont juggle two
    
    V_posed = LBS(V, W, J_modified, kintree_modified, theta)
    V_cycle = LBS(V_posed, W, J_modified, kintree_modified, -theta)
    
    #V_unposed = inverse_LBS(V, W, J_modified, kintree_modified, theta)
    #V_cycle = LBS(V_unposed, W, J_modified, kintree_modified, theta)
    
    print(np.sum(V - V_cycle))
    F = np.array(F, dtype=int)
    
    random_str = str(np.random.rand())[3:9]
    igl.write_obj("./results/pose-unpose/V_posed_SMPL"+random_str+".obj", V, F)
    igl.write_obj("./results/pose-unpose/V_cycle_SMPL"+random_str+".obj", V, F)
    
    print(">> End of test ", random_str)

