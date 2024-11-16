#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:17:14 2024

@author: bartu
"""
from scipy.spatial.transform import Rotation
import numpy as np

from .utils.linalg_utils import compose_rigid_transform_matrix
from .global_vars import VERBOSE
# =============================================================================
# Bone Class
# =============================================================================
class Bone():
    def __init__(self, endpoint_location, idx, parent=None):
        assert type(idx) == int, f"Expected bone index to be type int, got {type(idx)}"
        self.idx = idx  # To couple with loaded skeleton matrix data
        
        if parent:
            assert type(parent) == Bone, f"Parent parameter is expected to be type Bone, got {type(parent)}"
            
        self.end_location = np.array(endpoint_location)
        self.start_location = np.zeros(3)
        if parent is not None:
            self.start_location = parent.end_location
        
        self.t = np.zeros(3)
        self.parent = parent
        self.children = []
        
    def set_start_location(self, start_location):
        self.start_location = start_location
        
    def set_parent(self, parent_node):
        self.parent = parent_node
    
    def add_child(self, child_node):
        self.children.append(child_node)
    
    def translate(self, offset_vec, override=True, keep_trans=False):
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
            if VERBOSE:
                print(">> WARNING: You're overriding the bone rest pose locations. Turn override parameter off if you intend to use this function as pose mode.")
            self.set_start_location(start_translated)
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
            if VERBOSE:
                print(">> WARNING: You're overriding the bone rest pose locations. Turn override parameter off if you intend to use this function as pose mode.")
            self.end_location = final_bone_pos
            if not keep_trans: # If we're not keeping the transformation, reset
                self.rotation = Rotation.from_euler('xyz', [0, 0, 0])
    
        return final_bone_pos


  
# =============================================================================
#  Skeleton Class       
# =============================================================================
class Skeleton():
    def __init__(self, root_vec=[0., 0., 1.]):
        """
            @param root_vec: list, torch or numpy.ndarray. It's a 3D coordinate vector 
            of root node laction. It will be used to create invisible root bone, 
            that is starting from the origin and ends at the provided root_vec.
        """
        self.rest_bones = []
        self.kintree = []
        
        # Initiate skeleton with a root bone
        assert len(root_vec) == 3, f"Root vector is expected to be a 3D vector, got {root_vec.shape}"
        root_bone = Bone(root_vec, idx=0)
        self.rest_bones.append(root_bone)
        
    def get_kintree(self):
        kintree = []
        for bone in self.rest_bones:
            if bone.parent is None:
                continue
            bone_id = bone.idx
            parent_id = bone.parent.idx
            kintree.append([parent_id, bone_id])   
            
        return kintree
    
    def get_absolute_transformations(self, theta, trans=None, degrees=True, quats=False):
        """
        Given the rotation angles and translation parameters,
        compute the individual bone rigid motion (rotation and translation)
        through Forward Kinematics.
        
        DISCLAIMER: Forward Kinematics implementation is adopted from Libigl:
        https://github.com/libigl/libigl/blob/main/include/igl/forward_kinematics.cpp
            
        Parameters
        ----------
        theta : np.ndarray
            Axis-angle rotations for each bone. Has shape (n_bones, 3)
        trans : np.ndarray
            Relative translation vector for each bone. Has shape (n_bones, 3).
            If not provided, the relative translations will be treated as zero
            vectors. Default is None.
        degrees : bool
            Indicates if the provided angles are in degrees. Set False if
            they are in radians (note that DFAUST/SMPL dataset is in radians).
            Default is True.
        quats : bool
            Indicates if the provided rotations are in quaternions. Default is 
            False. This option overrides the parameter degrees

        Returns
        -------
        absolute_rot : np.ndarray
            Absolute rotation quaternions per bone, has shape (n_bones, 4). It can
            be used to represent rotation together with in a 4x4 transformation matrix 
            for LBS, or directly used in DQS as a quaternion.
        absolute_trans : np.ndarray
            Absolute translation as a 3D vector per bone, has shape (n_bones, 3).
            Represents the amount of total translation the bone needs in the given
            kinematic chain. Together with absoulute rotation, it can be embedded
            in a transformation matrix to be used in Linear Blend Skinning.

        """
        n_bones = len(self.rest_bones)
        
        # Check and configure translation parameters
        if trans is None: trans = np.zeros((n_bones,3))
        else: assert trans.shape == (n_bones, 3), f"Expected trans to have shape (n_bones={n_bones}, 3), got {trans.shape}"
        relative_trans = np.array(trans)
        
        # Check and configure rotation parameters
        if quats:
            assert theta.shape == (n_bones, 4), f"Expected theta as quaternions to have shape (n_bones={n_bones}, 4), got {theta.shape}"
            relative_rot_q = theta
        else:
            assert theta.shape == (n_bones, 3), f"Expected theta to have shape (n_bones={n_bones}, 3), got {theta.shape}"
            relative_rot_q = np.empty((n_bones, 4))
            for i in range(n_bones):
                rot = Rotation.from_euler('xyz', theta[i], degrees=degrees)
                relative_rot_q[i] = rot.as_quat()
        
        # Compute absolute rigid motions with Dynamic programming
        # DISCLAIMER: forward kinematics is adopted from Libigl's implementaiton
        # https://github.com/libigl/libigl/blob/main/include/igl/forward_kinematics.cpp
        computed = np.zeros(n_bones, dtype=bool)
        vQ = np.zeros((n_bones, 4))
        vT = np.zeros((n_bones, 3))
        def fk_helper(b : int): 
            if not computed[b]:
                if self.rest_bones[b].parent is None:
                    # Base case for roots
                    vQ[b] = relative_rot_q[b]
                    
                    abs_rot = Rotation.from_quat(vQ[b])
                    r = self.rest_bones[b].start_location
                    r_rotated = abs_rot.apply(r)               # (vQ[b] * r)
                    vT[b] = r - r_rotated + relative_trans[b]          
                else:
                    # First compute parent's
                    parent_idx = self.rest_bones[b].parent.idx
                    fk_helper(parent_idx)
            
                    parent_rot = Rotation.from_quat(vQ[parent_idx])
                    rel_rot = Rotation.from_quat(relative_rot_q[b])
                    vQ[b] = (parent_rot * rel_rot).as_quat()
                    
                    abs_rot = Rotation.from_quat(vQ[b])
                    r = self.rest_bones[b].start_location 
                    r_rotated = abs_rot.apply(r)  # (vQ[b] * r)
                    
                    abs_rot_parent = Rotation.from_quat(vQ[parent_idx])
                    x = r + relative_trans[b] 
                    x_rotated = abs_rot_parent.apply(x) # vQ[p]* (r + dT[b])
                   
                    vT[b] = vT[parent_idx] - r_rotated + x_rotated
                    
                computed[b] = True
                
        for b in range(n_bones):
            fk_helper(b)
        
        absolute_rot, absolute_trans = vQ, vT
        return absolute_rot, absolute_trans
     
    def compute_bone_locations(self, abs_rot_quat, abs_trans):
        """
        Given absolute rotation quaternions and translation vectors
        compute the endpoints of the bones.
        
        Parameters
        ----------
        abs_rot_quat : np.ndarray
            Has shape (n_bones, 4).
        abs_trans : np.ndarray
            Has shape (n_bones, 3).

        Returns
        -------
        final_bone_locations : np.ndarray
            Has shape (n_bones * 2). TODO: make it (n_bones, 2) and please exclude root from n_bones.
            Here n_bones include root bone but it's often confusing. 

        """
        n_bones = len(self.rest_bones)
        assert abs_rot_quat.shape == (n_bones, 4), f"Expected abs_rot_quat to have shape ({n_bones},4) got {abs_rot_quat.shape}."
        assert abs_trans.shape == (n_bones, 3), f"Expected abs_rot_quat to have shape ({n_bones},3) got {abs_trans.shape}."
        
        final_bone_locations = np.empty((2*n_bones, 3))
        for i, bone in enumerate(self.rest_bones):
            rot = Rotation.from_quat(abs_rot_quat[i])
            M = compose_rigid_transform_matrix(abs_trans[i], rot)
            
            s, e = np.ones((4,)), np.ones((4,))
            s[:3] = bone.start_location
            e[:3] = bone.end_location 
            s_translated = (M @ s)[:3]
            e_translated = (M @ e)[:3]
            
            final_bone_locations[2*i] = s_translated
            final_bone_locations[2*i + 1] = e_translated
        return final_bone_locations
        
        
    def pose_bones(self, theta, trans=None, get_transforms=False, degrees=False): 
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
            trans = np.zeros((len(self.rest_bones), 3))
        
        abs_rot_quat, abs_trans = self.get_absolute_transformations(theta, trans, degrees)
        final_bone_locations = self.compute_bone_locations(abs_rot_quat, abs_trans)
    
        #if exclude_root:
            # Warning: This still assumes absolute transformations are returned for all joints
            # so it doesn't exclude root. It just excludes root bone assuming the descedents are
            # connected to them without an offset. 
            # TODO: Couldn't we design better so that we won't need this option at all?
        #    final_bone_locations = final_bone_locations[2:] 
            
        if get_transforms:
            return final_bone_locations, abs_rot_quat, abs_trans
        return final_bone_locations
        
    def insert_bone(self, endpoint, parent_idx, 
                    offset_ratio=0.0,
                    startpoint=None):
        """
        Insert a bone providing the tip location and parent index. 

        Parameters
        ----------
        endpoint : np.ndarray
            Location of the tip of the inserted bone, has shape (3, ).
        parent_idx : int
            Index of the parent bone corresponding to the bone array.
        offset_ratio : float, optional
            Determines the starting point of the inserted bone on the parent bone.
            It must be between [0.0, 1.0], when at 1.0 the inserted bone
            starts at the starting point of the parent bone, at 0.0 it is
            at the tip of the parent bone, in between it's positioned based on 
            the parent bone's length and provided ratio. The default is 0.0.
        startpoint : np.ndarray, optional
            When provided, it sets the bone starting point. The default is None.

        Returns
        -------
        new_bone.idx : int
        Returns the index of the inserted bone in the Skeleton.bones list.
        """
        if not type(parent_idx) == int:
            assert np.issubdtype(parent_idx, np.integer), f"Expected parent index to be an integer, got {type(parent_idx)}"
       
        assert parent_idx >= 0, f"Expected parent index to be a positive number, got {parent_idx}, more than one root bone is not allowed yet."
        assert parent_idx < len(self.rest_bones), f">> Invalid parent index {parent_idx}. Please select an index less than {len(self.rest_bones)}"
        assert offset_ratio <= 1.0 and offset_ratio >= 0.0, f"Offset ratio is expected to be in range [0.0, 1.0], got {offset_ratio}."
        
        parent_bone = self.rest_bones[parent_idx]
        new_bone = Bone(endpoint, idx=len(self.rest_bones), parent=parent_bone)
        if np.linalg.norm(new_bone.end_location - new_bone.start_location) < 1e-18:
            print("WARNING: Expected inserted bone to have non-zero length. Please use a PointHandle interface for different types of handles.")
        
        self.rest_bones.append(new_bone)
        self.rest_bones[parent_idx].add_child(new_bone)
        self.kintree = self.get_kintree() # Update kintree
        
        if startpoint is None:
            if offset_ratio:
                parent_dir = parent_bone.start_location - parent_bone.end_location
                parent_dir_scaled = parent_dir * offset_ratio
                # Translate the bone along the parent bone line segment
                self.rest_bones[new_bone.idx].translate(parent_dir_scaled, override=True)
            else:
                print(">> INFO: Startpoint of the bone is taken as the tip of the parent bone")
        else:
            self.rest_bones[new_bone.idx].set_start_location(startpoint)
        
        # Sanity check the created bone.idx corresponds to its index the bones list
        assert new_bone.idx == len(self.rest_bones)-1
        return new_bone.idx
    
    def remove_bone(self, bone_idx):
        bone_to_be_removed = self.rest_bones[bone_idx]
        parent = bone_to_be_removed.parent
        
        if parent:
            for child in bone_to_be_removed.children:
                child.parent = parent    
        else:
            if len(bone_to_be_removed.children) == 1:
                child.parent = None
            print(">> WARNING: Cannot remove root bone with multiple children.")
            return
           
        # Remove the bone from the skeleton bones list
        bone_to_be_removed.children = None    # It's unnecessary probably.
        self.rest_bones.remove(bone_to_be_removed)
        self.kintree = self.get_kintree()     # Update kintree
        return
        
    def get_bone(self, bone_idx):
        assert bone_idx < len(self.rest_bones), f">> Invalid bone index {bone_idx}. Please select an index less than {len(self.rest_bones)}"
        return self.rest_bones[bone_idx]
    
    def get_rest_bone_locations(self, exclude_root, indices=None):
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
        if indices is None: indices = np.arange(0, len(self.rest_bones), dtype=int)
        bone_endpoints = []
        for i in indices:
            bone = self.rest_bones[i]
            if exclude_root and bone.parent is None:
                continue # Skip the root node
            
            bone_endpoints.append(bone.start_location)
            bone_endpoints.append(bone.end_location)
            
        return np.array(bone_endpoints)
    
# =============================================================================
# Helper Routines    
# =============================================================================
# TODO: Could we move these functions to skeleton class so that every other test
# can utilize them?
def create_skeleton(joint_locations, kintree):
    """
    WARNING: This function is suited for SMPL-like data where joints are treated
    as bones in the skeleton. If your skeleton have bone nodes instead of joint
    nodes, don't call this function, use create_skeleton_from() instead.
    
    WARNING: This function assumes there's only one root node and that is given
    in the joint_locations[0].

    Parameters
    ----------
    joint_locations : TYPE
        DESCRIPTION.
    kintree : TYPE
        DESCRIPTION.

    Returns
    -------
    test_skeleton : TYPE
        DESCRIPTION.

    """
    test_skeleton = Skeleton(root_vec = joint_locations[0])
    for edge in kintree:
         parent_idx, bone_idx = edge
         test_skeleton.insert_bone(endpoint = joint_locations[bone_idx], 
                                   parent_idx = parent_idx)   
    return test_skeleton

def create_skeleton_from(bone_locations, kintree):
    
    assert kintree.shape[1] == 2, f"Expected kintree to have length 2 in dimension 1. Found shape {kintree.shape}."
    assert bone_locations.shape[1] == 2, f"Expected bone locations to have shape (n_bones, 2, 3). Found shape {bone_locations.shape}."
    assert kintree[0][0] == -1, f"Expected the first entry of kintree to be [-1, x], i.e. root node. Found {kintree[0]}."

    n_bones = len(kintree)
    
    skeleton = Skeleton(root_vec=bone_locations[kintree[0][1]][0]) ## TODO: this would be global translation when we remove root_bone
    
    print(">> Warning: kintree is modified to fit the implementation. This should be removed once the root bone issue is resolved.")
    for i in range(n_bones):
        parent_id, bone_id = kintree[i]
        skeleton.insert_bone(
                              parent_idx = parent_id + 1 , # TODO: this is because of root bone issue
                              startpoint =  bone_locations[bone_id][0],
                              endpoint = bone_locations[bone_id][1]
                             ) 
    return skeleton
                                  
    
# TODO: this could be a general function to add multiple bones because 
# there's nothing specific about helper bones here.
def add_helper_bones(test_skeleton, 
                     helper_bone_endpoints, 
                     helper_bone_parents,
                     offset_ratio=0.0, 
                     startpoints=[]):
    """
    Add one or multiple helper bones to an existing skeleton. 
    
    Parameters
    ----------
    test_skeleton : Skeleton
        An existing skeleton to be modified via adding new bones.
    helper_bone_endpoints : np.ndarray or list
        Holds the 3D vectors that determines the location of helper bone tips.
    helper_bone_parents : list
        List of integers that indicate the index of the parent bones.
    offset_ratio : float, optional
        Determines the location of the helper bone with respect to its parent. 
        When set to 1.0, the bone starts at the start point of its parent,
        when set to 0.0 it starts at the tip of the parent. In between,
        the bone is located somewhere along the parent bone. Note that this
        option is not regarded if an explicit startpoint is given.
        The default is 0.0, i.e. the bone starts at the tip of the parent.
        
    startpoints : list, optional
        List of 3D vectors that determines the starting locations of the 
        helper bones. The default is [].

    Returns
    -------
    helper_idxs : list
        List of integers that are the indices of helper bones in the existing 
        skeleton.
    """
    assert offset_ratio <= 1.0 and offset_ratio >= 0.0, f">>  Excpected offset_ratio to be in range [0, 1], got {offset_ratio}."
    if offset_ratio:
        print("WARNING: Adding a helper bone as a child of another helper bone is assumed\
        to have no offset, i.e offset_ratio = 0.0, but provided ratio is {offset_ratio}.")
        
    n_helper = len(helper_bone_parents)
    if len(startpoints)==0: startpoints = np.repeat([None],n_helper)
    
    helper_idxs = []
    for i in range(n_helper):
        bone_idx = test_skeleton.insert_bone( endpoint = helper_bone_endpoints[i],
                                              parent_idx = helper_bone_parents[i],
                                              offset_ratio = offset_ratio,
                                              startpoint = startpoints[i]
                                              )
        helper_idxs.append(bone_idx)
    return helper_idxs


# TOOD: This has nothing to do with Skeleton class, shall we move it to smpl_utils
# kind of a module?
def extract_headtail_locations(joints, kintree, exclude_root=False, collapse=True):
    # Given the joints and their connectivity
    # Extract two endpoints for each bone
    # This function is written to deal with SMPL kintree and joints locations
    n_bones = len(kintree)
    if not exclude_root: n_bones += 1
    J = np.empty((n_bones, 2, 3))
    
    if not exclude_root: # WARNING: Assumes root node is at 0
        J[0] = np.array([[0,0,0], joints[0]])
        
    for pair in kintree:
        parent_idx, bone_idx = pair
        J[bone_idx] = np.array([joints[parent_idx], joints[bone_idx]]) 
        
    if collapse:
        J = np.reshape(J, (-1,3))
    return J

if __name__ == "__main__":
    print(">> Testing skeleton.py is NOT implemented yet...")
    
    
     
    

    
        