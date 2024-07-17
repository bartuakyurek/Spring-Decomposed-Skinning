#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:01:08 2024

@author: bartu
"""
import torch
import igl
import numpy as np

def quat2mat(quat):
    """
    DISCLAIMER: Taken from
    https://github.com/Dou-Yiming/Pose_to_SMPL/blob/main/smplpytorch/pytorch/rodrigues_layer.py
    
    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]
    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat

def batch_rodrigues(axisang):
    """
    DISCLAIMER: Taken from 
    https://github.com/Dou-Yiming/Pose_to_SMPL/blob/main/smplpytorch/pytorch/rodrigues_layer.py

    Parameters
    ----------
    axisang : TYPE
        DESCRIPTION.

    Returns
    -------
    rot_mat : TYPE
        DESCRIPTION.

    """
    #axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat

###############################################################################

def _axisang_to_matrix(axisang):
    return batch_rodrigues(axisang)
    

def _get_affine(R, t):
  
    affine = np.zeros((4,4))
    affine[-1,-1] = 1.
    affine[:3, :3] = R
    affine[:3, -1] = t
    
    return affine

def _axisang_to_affine(axisang, t):
    rot_matrix = _axisang_to_matrix(axisang)
    return _get_affine(rot_matrix, t)

def batch_axsang_to_quats(axisang):
    # TODO: HEPSINI TORCH'DA YAP YAPACAKSAN....
    axisang = torch.from_numpy(axisang)
    
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)

    quat = np.array(quat, order='F')
    return quat

###############################################################################

def get_unpose_matrix(theta, t):
    return _axisang_to_affine(-theta, -t)

def get_pose_matrix(theta, t):
    return _axisang_to_affine(theta, t)
    
def forward_kin(joint_pos, 
                joint_edges, 
                joint_parents, 
                relative_rot, # axis angle representation
                relative_trans=None):
    
    relative_rot_q = batch_axsang_to_quats(relative_rot)
    
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
    
    R_mat = np.array(quat2mat(torch.from_numpy(abs_rot)))
   
    num_bones = abs_t.shape[0]
    
    V_posed = np.zeros_like(V)
    for vertex in range(V_posed.shape[0]):
        for bone in range(num_bones):
           
            # TODO: remove for loop!
            if W[vertex, bone] < 1e-15:
                continue # Weight is zero, don't add any contribution
            
            transform_mat = _get_affine(R_mat[bone], abs_t[bone])
            V_homo = np.insert(V[vertex], 3, 1.0)
            V_new_homo = W[vertex, bone] * transform_mat @ V_homo
            
            V_posed[vertex] += V_new_homo[:3]
            # TODO: put the rotation and translation into affine matrix
            # and then apply affine transformation!
    
    return V_posed

def inverse_LBS(V_posed, W, J, JE, theta):
    return LBS(V_posed, W, J, JE, -theta)

if __name__ == "__main__":
    print(">> Testing transformations.py...")
    
    
    # TODO: add root bone's head location as origin so that we can have FK properly
    
    # Load the mesh data
    with np.load("./results/skinning_sample_data.npz") as data:
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
    V_unposed = inverse_LBS(V, W, J_modified, kintree_modified, theta)
    V_cycle = LBS(V_unposed, W, J_modified, kintree_modified, theta)
    
    print(np.sum(V - V_cycle))
    #np.savez("./results/V_unposed.npz", V_unposed)
    F = np.array(F, dtype=int)
    igl.write_obj("V_unposed_SMPL.obj", V_unposed, F)
    igl.write_obj("V_cycle_SMPL.obj", V_cycle, F)
    
    
    print(">> End of test.")

