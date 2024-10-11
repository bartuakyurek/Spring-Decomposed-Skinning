#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:17:14 2024

@author: bartu
"""
from scipy.spatial.transform import Rotation
import numpy as np
import igl

class Bone():
    def __init__(self, endpoint_location, idx, parent=None):
        assert type(idx) == int, f"Expected bone index to be type int, got {type(idx)}"
        self.idx = idx  # To couple with loaded skeleton matrix data
        
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
        
    def translate(self, offset_vec, override=True, keep_trans=True):
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
        
        start_translated = self.start_location + offset_vec
        end_translated = self.end_location + offset_vec
        if override:
            print(">> WARNING: You're overriding the bone rest pose locations. Turn override parameter off if you intend to use this function as pose mode.")
            self.start_location = start_translated
            self.end_location = end_translated
            if keep_trans:
                self.t += offset_vec
        else:
            self.t += offset_vec
        
        return (start_translated, end_translated)
    
    def rotate(self, axsang, override=True, keep_trans=True):
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
            if not keep_trans: # If we're not keeping the transformation, reset
                self.rotation = Rotation.from_euler('xyz', [0, 0, 0])
            
        return final_bone_pos
        
class Skeleton():
    def __init__(self, root_vec=[0., 0., 1.]):
        """
            @param root_vec: list, torch or numpy.ndarray. It's a 3D coordinate vector 
            of root node laction. It will be used to create invisible root bone, 
            that is starting from the origin and ends at the provided root_vec.
        """
        self.bones = []
        self.kintree = []
        
        # Initiate skeleton with a root bone
        assert len(root_vec) == 3, f"Root vector is expected to be a 3D vector, got {root_vec.shape}"
        root_bone = Bone(root_vec, idx=0)
        self.bones.append(root_bone)
        
    def get_kintree(self):
        kintree = []
        for bone in self.bones:
            if bone.parent is None:
                continue
            bone_id = bone.idx
            parent_id = bone.parent.idx
            kintree.append([parent_id, bone_id])   
            
        return kintree
    
    def get_absolute_transformations(self, theta, trans, degrees):
        
        n_bones = len(self.bones)
        assert len(theta) == n_bones, f"Expected theta to have shape (n_bones, 3), got {theta.shape}"
        assert len(trans) == n_bones, f"Expected trans to have shape (n_bones, 4), got {trans.shape}"
        assert trans.shape[1] == 4, "Please provide homogeneous coordinates."
        
        relative_trans = np.array(trans)
        relative_rot_q = np.empty((n_bones, 4))
        for i in range(n_bones):
            rot = Rotation.from_euler('xyz', theta[i], degrees=degrees)
            relative_rot_q[i] = rot.as_quat()
        
        computed = np.zeros(n_bones, dtype=bool)
        vQ = np.zeros((n_bones, 4))
        vT = np.zeros((n_bones, 4))
        # Dynamic programming
        def fk_helper(b : int): 
            if not computed[b]:
                r = np.ones(4) 
                r[:3] = self.bones[b].start_location 
                if self.bones[b].parent is None:
                    # Base case for roots
                    vQ[b] = relative_rot_q[b]
                    vT[b] = r - (relative_rot_q[b] * r) + relative_trans[b]
                    
                else:
                    # First compute parent's
                    parent_idx = self.bones[b].parent.idx
                    fk_helper(parent_idx)
            
                    vQ[b] = vQ[parent_idx] * relative_rot_q[b]
                    vT[b] = vT[parent_idx] - (vQ[b] * r) + vQ[parent_idx]*(r + relative_trans[b])
                    #vT[b] = -(vQ[b] * r) + vQ[parent_idx]*(r + relative_trans[b] - vT[parent_idx]) + vT[parent_idx]

                computed[b] = True
                
        for b in range(n_bones):
            fk_helper(b)
        
        absolute_rot, absolute_trans = vQ, vT
        return absolute_rot, absolute_trans
     
        
    def pose_bones(self, theta, trans=None, get_transforms=False, degrees=False, exclude_root=False): 
        """
        Apply the given relative rotations to the bones in the skeleton.
        This is used for deforming the rest pose to the current frame.
        WARNING: It should be in between the rest pose and the desired frame.

        Parameters
        ----------
        theta : np.ndarray 
            Relative bone rotations as given in SMPL data.
        trans : np.ndarray (optionl)
            Relative bone translations as given in SMPL data.
            If None, relative translations are set to zero vectors.
        degrees: bool
            If True, theta rotation parameter will be interpreted as in degrees.
            Else, theta is assumed to be in radians. Default is False.
            
        Returns
        -------
        final_bone_locations : list
        Joint locations of the posed skeleton, that is 2 joints per bone, for 
        the both endpoints of the bone. Has shape (n_bones * 2, 3)
        
        abs_rot_quat : np.ndarray (optional)
            Absolute quaternion rotations for each bone, has shape (n_bones, 4).
            Returned only if get_transforms=True.
            
        abs_trans : np.ndarray (optional)
            Absolute translations for each bone, has shape (n_bones, 3)
            Returned only if get_transforms=True.
        """
        assert type(theta) == np.ndarray, f"Expected pose type np.ndarray, got {type(theta)}"
        
        if trans is None:
            trans = np.zeros((len(self.bones), 4))
            trans[:, -1] = 1   # Homogeneous coordinates
        else:
            assert trans.shape == (len(self.bones), 4) or trans.shape == (len(self.bones), 3) 
            if trans.shape[1] == 3:
                trans[:, -1] = 1   # Convert to homogeneous coordinates
                
        abs_rot_quat, abs_trans_homo = self.get_absolute_transformations(theta, trans, degrees)
        final_bone_locations = np.empty((2*len(self.bones), 3))
        
        for i, bone in enumerate(self.bones):
            rot = Rotation.from_quat(abs_rot_quat[i])
            rot_mat = rot.as_matrix()
            
            # Convert absolute rotations and translations into a single transformation matrix
            M = np.zeros((4,4))
            M[:3, :3] = rot_mat 
            M[:, -1] = abs_trans_homo[i] # Homogeneous coordinates have 1.0 at the end
            
            s, e = np.ones((4,1)), np.ones((4,1))
            s[:3, 0] = bone.start_location
            e[:3, 0] = bone.end_location
            s_translated = (M @ s)[:3,0]
            e_translated = (M @ e)[:3,0]
            
            final_bone_locations[2*i] = s_translated
            final_bone_locations[2*i + 1] = e_translated
            
        if exclude_root:
            # Warning: This still assumes absolute transformations are returned for all joints
            # so it doesn't exclude root. It just excludes root bone assuming the descedents are
            # connected to them without an offset. TODO: Couldn't we design better so that we won't
            # need this option at all?
            final_bone_locations = final_bone_locations[2:] 
            
        if get_transforms:
            return final_bone_locations, abs_rot_quat, abs_trans_homo[:3]
        return final_bone_locations
        
    def insert_bone(self, endpoint_location, parent_node_idx):
        assert parent_node_idx < len(self.bones), f">> Invalid parent index {parent_node_idx}. Please select an index less than {len(self.bones)}"
        
        parent_bone = self.bones[parent_node_idx]
        new_bone = Bone(endpoint_location, idx=len(self.bones), parent=parent_bone)
        
        self.bones.append(new_bone)
        self.bones[parent_node_idx].add_child(new_bone)
        self.kintree = self.get_kintree() # Update kintree
    
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
        self.kintree = self.get_kintree()     # Update kintree
        return
        
    def get_bone(self, bone_idx):
        assert bone_idx < len(self.bones), f">> Invalid bone index {bone_idx}. Please select an index less than {len(self.bones)}"
        return self.bones[bone_idx]
    
    def get_rest_bone_locations(self, exclude_root=True):
        """
        Get the bone joint locations for the entire skeleton. Note that this is
        not the same as SMPL joint locations. In this skeleton, every bone has
        two endpoints that may or may not coincide with its children and parent
        bones endpoints (allows offset between bones).

        Parameters
        ----------
        exclude_root : bool, optional
            Exclude the root node locations from the returned list.
            This can be useful in order not to display root bone if it's 
            directly connected to the descendant bone. The default is True.
            
        Returns
        -------
        bone_endpoints : list
            list of joint locations that are both endpoints of each bone. 
            so there are #n_bones * 2 endpoints in the returned list
        """
        bone_endpoints = []
        for bone in self.bones:
            if exclude_root and bone.parent is None:
                continue # Skip the root node
                
            bone_endpoints.append(bone.start_location)
            bone_endpoints.append(bone.end_location)
            
        return bone_endpoints
    
if __name__ == "__main__":
    print(">> Testing skeleton.py...")
     
    import torch
    import pyvista as pv
    
    from smpl_torch_batch import SMPLModel
    from skeleton_data import get_smpl_skeleton
    from pyvista_render_tools import add_skeleton
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
    
    # ---------------------------------------------------------------------------- 
    # Create skeleton based on rest pose SMPL data
    # ---------------------------------------------------------------------------- 
    smpl_skeleton = Skeleton(root_vec = J_rest[0])
    for edge in kintree:
        parent_idx, bone_idx = edge
        smpl_skeleton.insert_bone(endpoint_location = J_rest[bone_idx], 
                                  parent_node_idx = parent_idx)
        
    # ---------------------------------------------------------------------------- 
    # Create plotter 
    # ---------------------------------------------------------------------------- 
    RENDER = True
    plotter = pv.Plotter(notebook=False, off_screen=not RENDER)
    plotter.camera_position = 'zy'
    plotter.camera.azimuth = -90

    # ---------------------------------------------------------------------------- 
    # Add skeleton mesh based on T-pose locations
    # ---------------------------------------------------------------------------- 
    n_bones = len(smpl_skeleton.bones)
    rest_bone_locations = smpl_skeleton.get_rest_bone_locations(exclude_root=True)
    line_segments = np.reshape(np.arange(0, 2*(n_bones-1)), (n_bones-1, 2))
    
    skel_mesh = add_skeleton(plotter, rest_bone_locations, line_segments)
    plotter.open_movie("./results/smpl-skeleton.mp4")
    
    n_repeats = 1
    n_frames = len(J)
    for _ in range(n_repeats):
        for frame in range(n_frames):
            
            # TODO: Update mesh points
            theta = np.reshape(pose[frame].numpy(), newshape=(-1, 3))
            t = trans[frame].numpy()
            # TODO: (because it's global t, should be only applied to root)
            # we're not using t, we should handle it after correcting the FK.
         
            posed_bone_locations = smpl_skeleton.pose_bones(theta, exclude_root=True)
            skel_mesh.points = posed_bone_locations
            
            # Write a frame. This triggers a render.
            plotter.write_frame()

    # Closes and finalizes movie
    plotter.close()
    plotter.deep_clean()

    
        