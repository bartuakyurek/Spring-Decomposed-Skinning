#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 10:05:05 2024

@author: bartu
"""
import numpy as np
from skeleton import Skeleton, Bone
from mass_spring import MassSpringSystem

class HelperBonesHandler:
    
    def __init__(self, skeleton, helper_indices, mass=1.0, stiffness=100,
                 mass_dscale=1.0, spring_dscale=0.01):
        """
        Create a mass-spring system provided an array of Bone objects.
        Every bone is modelled as two masses at the tip points, connected
        via a spring. The beginning mass of the bone is fixed, so that only the
        ending mass is jiggling.

        """
        self.skeleton = skeleton
        self.prev_bone_locations = skeleton.get_rest_bone_locations(exclude_root=False) 
        print("WARNING: The prev_bone_locations are set at rest locations initially. \
              However it should be set before the beginning of the animation because rest position might not be animated at all.")
        # TODO: If what you're returning is actually the JOINT locations, why are you
        # calling these variables and functions as BONE locations? What is a location of
        # a bone afterall?
        
        self.helper_indices = helper_indices
        helper_bones = np.array(skeleton.rest_bones)[helper_indices]
 
        self.dt = 1. / 24
        self.ms_system = MassSpringSystem(self.dt)
    
        print("WARNING: Masses should be initiated NOT at the rest pose but the first keyframe pose \
              to be simulated in between the frames")
        n_helper = len(helper_bones)
        for i in range(n_helper):
            helper_start = helper_bones[i].start_location
            helper_end = helper_bones[i].end_location
            
            mass1 = self.ms_system.add_mass(helper_start, mass=mass, dscale=mass_dscale)
            mass2 = self.ms_system.add_mass(helper_end, mass=mass, dscale=mass_dscale)
            
            self.ms_system.connect_masses(mass1, mass2, stiffness=stiffness, dscale=spring_dscale)
            self.ms_system.fix_mass(mass1)
    
        self.fixed_idx = self.ms_system.fixed_indices
        self.free_idx = self.ms_system.get_free_mass_indices()
    # TODO: We should be able to change the individual masses and stiffness, 
    # for optimization we should be able to provide an array of particle mass
    # that will update the individual Particle.mass in the system
    
    def update(self, theta, trans, degrees, exclude_root):
        """
        Given the relative rotations, update the skeleton joints with mass-spring
        system and return the resulting joint locations.

        Parameters
        ----------
        theta : np.ndarray
            DESCRIPTION.
        degrees : bool 
            DESCRIPTION.
        exclude_root : bool
            DESCRIPTION.

        Returns
        -------
        simulated_locations : np.ndarray
            Coordinates of the simulated bones. It has shape of (2*n_bones, 3)
            if exclude_root is set to False. If exclude_root is True, the returned
            array has shape (2*(n_bones-1), 3) where every consecutive points 
            are defining two endpoints of a bone, where bone is a line segment.
        """
        if type(theta) is list:
            theta = np.array(theta)
        elif type(theta) is not np.ndarray:
            print(f"WARNING: Theta is expected to be either list or np.ndarray type, got {type(theta)}.")
        assert type(degrees) == bool, f"Expected degrees parameter to have type bool, got {type(degrees)}"
        assert type(exclude_root) == bool, f"Expected exclude_root parameter to have type bool, got {type(exclude_root)}"

        # TODO: exclude_root=False is intentional! Do not delete it! But also it's not
        # a good design if you're not using the same parameter name, your intention is unclear
        # so maybe you could remove this exclude_root option all together and handle it elsewhere
        
        rigidly_posed_locations = self.skeleton.pose_bones(theta, trans, degrees=degrees, exclude_root=False)
        simulated_locations = rigidly_posed_locations.copy() 
        for i, helper_idx in enumerate(self.helper_indices):
            
            # TODO: why don't you rename ms_system that doesn't sound like a meaningful name?
            # even system could be more meaningful though it might be close to a keyword
            # maybe self.simulator could work.
            
            # WARNING: You're taking the difference data from the rigid skeleton, but what happens
            # if you had a chaing of helper bones that are affecting each other? i.e.
            # The start of the child helper bone would be changed in previous frame, are your posed_bpnes
            # taking this into account? No.Maybe you would change theta trans parameters before skeleton.pose_bones
            diff = rigidly_posed_locations - self.prev_bone_locations
            helper_end_idx = (2 * helper_idx) + 1 # since bone locations have 2 joints per bone, multiply helper_bone_idx by 2 
            translate_vec = diff[helper_end_idx]  # TODO: then why don't you have a better data structure? Maybe dict could 
                                                  # work to access diff[helper_idx]["end"] or diff[helper_idx].end_location to
                                                  # access the bone.
            
            # Step 1 - Translate the endpoint of the current helper bone
            # TODO: are you handling the chained helper bones? like the start of
            # the children bone should be relocated in that case, 
            # and FK needed to be  re-called?
            fixed_mass_idx = self.fixed_idx[i]
            self.ms_system.translate_mass(fixed_mass_idx, translate_vec)
            self.ms_system.simulate()

            # Step 2 - Get current mass positions
            cur_mass_locations = self.ms_system.get_mass_locations()
            free_mass_locations = cur_mass_locations[self.free_idx]
            
            simulated_locations[helper_end_idx] = free_mass_locations[i]
            self.prev_bone_locations = simulated_locations
                    
        if exclude_root:
            return simulated_locations[2:]
        return simulated_locations
        
    
    
    