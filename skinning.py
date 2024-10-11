#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

IMPORTANT NOTES:
- Q: are SMPL's rotations in degrees or in radians?


"""
import igl
import torch
import numpy as np
from scipy.spatial.transformation import Rotation

def compose_transform_matrix(trans_vec, rot : Rotation ):
    """
    Compose a transformation matrix given the translation vector
    and Rotation object.

    Parameters
    ----------
    trans_vec : np.ndarray or list
        3D translation vector to be inserted at the last column of 4x4 matrix.
    rot : Rotation
        Rotation object of scipy.spatial.transformation. This is internally
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
    
    V_deformed = None
    if skinning_type == "LBS" or skinning_type == "lbs":
        # Deform vertices based on Linear Blend Skinning
        pass
    else:
        print(f">> ERROR: This skinning type \"{skinning_type}\" is not supported yet.")
    
    return V_deformed


# todo:  taken https://github.com/Dou-Yiming/Pose_to_SMPL/blob/main/smplpytorch/pytorch/rodrigues_layer.py
def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
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


def batch_axsang_to_quats(rot):
    assert rot.shape[1] == 3

    roll = rot[:, 0] / 2.
    pitch = rot[:, 1] / 2.
    yaw = rot[:, 2] / 2.
    
    if type(rot) == np.ndarray:
        sin = np.sin
        cos = np.cos
        stack = np.stack
    else: 
        sin = torch.sin
        cos = torch.cos
        stack = torch.stack
        
    qx = sin(roll) * cos(pitch) * cos(yaw)
    qy = cos(roll) * sin(pitch) * cos(yaw)
    qz = cos(roll) * cos(pitch) * sin(yaw)
    qw = cos(roll) * cos(pitch) * cos(yaw)
    
    return stack((qx, qy, qz, qw)).transpose()

def batch_axsang_to_quats2(axisang):
    # TODO: assert axisang is type torch array
   
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    
    return quat

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
    """
    WARNING: Relative translation is not implemented yet.

    Parameters
    ----------
    V : TYPE
        Vertex locations in T-Pose.
    W : TYPE
        Vertex-bone binding weights V x J.
    J : TYPE
        Joint locations, J x 3.
    JE : TYPE
        Joint edges, i.e. kintree, J-1 x 2.
    theta : TYPE
        Joint axis-angle orientations in radians.

    Returns
    -------
    T : TYPE
        Absolute transformation matrices of bones (rather than relative transformations). 
        Computed by Forward Kinematics.
    V_posed : TYPE
        Vertex world locations after posing the vertices of T-pose.

    """
    
    P = igl.directed_edge_parents(JE)
    abs_rot, abs_t = forward_kin(J, JE, P, theta)
    
    R_mat = np.array(quat2mat(torch.from_numpy(abs_rot)))
    num_bones = abs_t.shape[0]
    
    V_posed = np.zeros_like(V)
    for vertex in range(V_posed.shape[0]):
        for bone in range(num_bones):
            if bone == 0:
                pass # TODO: decide your notation! use only joints or edges! SMPL uses joint notation... but idk.
            # tmp = W[vertex, bone] * np.matmul(V[vertex], R_mat[bone]) + abs_t[bone]
            # TODO: remove for loop!
            V_posed[vertex] += W[vertex, bone] * np.matmul(V[vertex], R_mat[bone]) + abs_t[bone]
            
    # TODO: where's the T you promised in return?...
    return V_posed

def inverse_LBS(V_posed, W, J, JE, theta):
    
    unposed_V = LBS(V_posed, W, J, JE, -theta)
    return unposed_V

if __name__ == "__main__":
    print(">> Testing skinning.py...")
    """
    from smpl_torch_batch import SMPLModel
    from skeleton_data import get_smpl_skeleton

    training_data = torch.load('./data/50004_dataset.pt')
    data_loader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=False)

    device = "cpu"
    smpl_model = SMPLModel(device=device, model_path='./body_models/smpl/female/model.pkl')
    kintree = get_smpl_skeleton()
    F = smpl_model.faces

    for data in data_loader:
       beta_pose_trans_seq = data[0].squeeze().type(torch.float64)
       betas, pose, trans = beta_pose_trans_seq[:,:10], beta_pose_trans_seq[:,10:82], beta_pose_trans_seq[:,82:] 
       target_verts = data[1].squeeze()
       smpl_verts, joints = smpl_model(betas, pose, trans)
       break
       
    V = smpl_verts.detach().cpu().numpy()
    J = joints.detach().cpu().numpy()
    n_frames, n_verts, n_dims = target_verts.shape
    """
    
    
    
