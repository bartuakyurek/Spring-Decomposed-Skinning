#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:17:14 2024

@author: bartu
"""
from scipy.spatial.transform import Rotation
import numpy as np

class Bone():
    def __init__(self, endpoint_location, parent=None):
        if parent:
            assert type(parent) == Bone, f"parent parameter is expected to be type Bone, got {type(parent)}"
            
        self.end_location = np.array(endpoint_location)
        if parent is None:
            self.start_location = np.zeros(3)
            self.visible = False # Root bone is an invisible one, determining global transformation
      
        else:
            self.start_location = parent.end_location
            self.visible = True
        
        self.rotation = Rotation.from_euler('xyz', angles=[0, 0, 0])
        self.t = np.zeros(3)
        # TODO: is this t for offset vector OR is it the absolute translation of the bone?
        # I think we also need to store a boneSpaceMatrix to offset the vertices into bone space,
        # apply self.rotation and self.translation and then use the inverse boneSpaceMatrix to 
        # locate the vertices.
        # Note: it's the absolute translation that is intended to be used as in libigl's forward_kinematics()
        
        self.parent = parent
        self.children = []
        
    def set_parent(self, parent_node):
        self.parent = parent_node
    
    def add_child(self, child_node):
        self.children.append(child_node)
        
    def translate(self, offset_vec, override=True):
        """
        Translate the bone line segment given the translation vector.

        Parameters
        ----------
        offset_vec : np.ndarray
            translation  vector to be applied to the bone points, has shape (3, )
        override : bool, optional
            Override the bone locations by applying the offset_vec. When it's False,
            do not update the bone locations just update the translation information.
            The intended usage of False is for animation, where the rest pose information
            should not be updated but we need to update bone transformations for forward
            kinematics. The default is True.

        Returns
        -------
        start_translated : np.ndarray 
            Has shape (3, ), it is the translated starting point of the bone line segment.
        end_translated : np.ndarray 
            Has shape (3, ), it is the translated ending point of the bone line segment.
        """
        assert offset_vec.shape == self.t.shape, f"Expected translation vector to have shape {self.t.shape}, got {offset_vec.shape}"
        self.t += offset_vec
        
        start_translated = self.start_location + offset_vec
        end_translated = self.end_location + offset_vec
        if override:
            print(">> WARNING: You're overriding the bone rest pose locations. Turn override parameter off if you intend to use this function as pose mode.")
            self.start_location = start_translated
            self.end_location = end_translated
        
        return (start_translated, end_translated)
    
    def rotate(self, axsang, override=True):
        """
        Sets the bone rotation and adjust the endpoint location of the bone.

        Parameters
        ----------
        axsang : np.ndarray or torch.Tensor
            Axis-angle representation of shape (3,).
            
        override: bool
            If True, it will change the endpoint location of the bone. Otherwise
            the rotation will not affect the rest location of the bone, that is 
            to be used in pose mode, i.e. to retrieve bone positions with Forward
            Kinematics.

        Returns
        -------
        final_bone_pos : location of the tip of the bone that is rotated.

        """
        # Translate the bone to bone space (i.e. bone beginning is the origin now)
        bone_space_vec = (self.end_location - self.start_location)  
        
        r = Rotation.from_euler('xyz', axsang)
        self.rotation = r * self.rotation # (p * q) is q rotation followed by p rotation
        
        bone_space_rotated = r.apply(bone_space_vec)
        final_bone_pos = bone_space_rotated + self.start_location
        
        # Since this is a rotation, bone origin does not move, so only change the
        # location of the tip of the bone.
        if override:
            print(">> WARNING: You're overriding the bone rest pose locations. Turn override parameter off if you intend to use this function as pose mode.")
            self.end_location = final_bone_pos
            
        return final_bone_pos
        
class Skeleton():
    def __init__(self, root_vec=[0., 0., 1.]):
        """
            @param root_vec: list, torch or numpy.ndarray. It's a 3D coordinate vector 
            of root node laction. It will be used to create invisible root bone, 
            that is starting from the origin and ends at the provided root_vec.
        """
        self.bones = []
        
        # Initiate skeleton with a root bone
        assert len(root_vec) == 3, f"Root vector is expected to be a 3D vector, got {root_vec.shape}"
        root_bone = Bone(root_vec)
        self.bones.append(root_bone)
        
    def pose_bones(self, theta): # A.k.a apply forward kinematics given the relative rotations
        """
        Apply the given relative rotations to the bones in the skeleton.
        This is used for deforming the rest pose to the current frame.
        WARNING: YOU SHOULDN'T USE THIS FUNCTION TO DEFORM SKELETON EVERY FRAME
        It should be in between the rest pose and the desired frame.

        Parameters
        ----------
        theta : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # Get the bones as a copy from skeleton (do not directly apply these to skeleton)
        # otherwise the bone information will be distorted. We want to keep the rest pose
        # information as is, and apply forward kinematics separately.
        bones = self.bones.copy()
        
        # go to bone space
        # rotate the bone w.r.t parent's rotation first
        # then apply bone's relative rotation
        # apply translation (if any)
        # go back to world space
        
        return
        
    def insert_bone(self, endpoint_location, parent_node_idx):
        assert parent_node_idx < len(self.bones), f">> Invalid parent index {parent_node_idx}. Please select an index less than {len(self.bones)}"
        
        parent_bone = self.bones[parent_node_idx]
        new_bone = Bone(endpoint_location, parent=parent_bone)
        
        self.bones.append(new_bone)
        self.bones[parent_node_idx].add_child(new_bone)
    
    def remove_bone(self, bone_idx):
        bone_to_be_removed = self.bones[bone_idx]
        parent = bone_to_be_removed.parent
        
        if parent:
            for child in bone_to_be_removed.children:
                child.parent = parent    
        else:
            print(">> WARNING: Cannot remove root bone.")
            return
        
        # Remove the bone from the skeleton bones list
        bone_to_be_removed.children = None    # It's unnecessary probably.
        self.bones.remove(bone_to_be_removed)
        return
        
    def get_bone(self, bone_idx):
        assert bone_idx < len(self.bones), f">> Invalid bone index {bone_idx}. Please select an index less than {len(self.bones)}"
        return self.bones[bone_idx]
        
if __name__ == "__main__":
    print(">> Testing skeleton.py...")
      
    import torch
    from smpl_torch_batch import SMPLModel
    from skeleton_data import get_smpl_skeleton

    from mass_spring import MassSpringSystem
    from pyvista_render_tools import add_skeleton, add_mesh
    # ---------------------------------------------------------------------------- 
    # Load SMPL animation file and get the mesh and associated rig data
    # ---------------------------------------------------------------------------- 
    data_loader = torch.utils.data.DataLoader(torch.load('./data/50004_dataset.pt'), batch_size=1, shuffle=False)
    smpl_model = SMPLModel(device="cpu", model_path='./body_models/smpl/female/model.pkl')
    kintree = get_smpl_skeleton()
    for data in data_loader:
       beta_pose_trans_seq = data[0].squeeze().type(torch.float64)
       betas, pose, trans = beta_pose_trans_seq[:,:10], beta_pose_trans_seq[:,10:82], beta_pose_trans_seq[:,82:] 
       target_verts = data[1].squeeze()
       smpl_verts, joints = smpl_model(betas, pose, trans)
       break
    V = smpl_verts.detach().cpu().numpy()
    J = joints.detach().cpu().numpy()
    n_frames, n_verts, n_dims = target_verts.shape

    # Get rest pose SMPL data
    rest_verts, rest_joints = smpl_model(betas, torch.zeros_like(pose), trans)
    J_rest = rest_joints.numpy()[0]
    
    # Create skeleton based on rest pose SMPL data
    smpl_skeleton = Skeleton(root_vec = J_rest[0])
    for edge in kintree:
        parent_idx, bone_idx = edge
        smpl_skeleton.insert_bone(endpoint_location = J_rest[bone_idx], 
                                  parent_node_idx = parent_idx)
        
    # Print skeleton contents
    for i, bone in enumerate(smpl_skeleton.bones):
        print(f"Bone {i} at {bone.start_location} - {bone.end_location}")
        if bone.parent:
            print("Bone parent endpoint: ", bone.parent.end_location)
        print("----------------------------------")
    
        