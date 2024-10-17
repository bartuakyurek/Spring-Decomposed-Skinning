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
    # TODO: We should be able to change the individual masses and stiffness, 
    # for optimization we should be able to provide an array of particle mass
    # that will update the individual Particle.mass in the system
    def __init__(self, 
                 skeleton, 
                 helper_idxs, 
                 point_spring=True,
                 mass=1.0, 
                 stiffness=100, 
                 damping=1.0,
                 mass_dscale=1.0, 
                 spring_dscale=1.0, dt=1./24,
                 simulation_mode=0,
                 fixed_scale=True,
                 ):
        """
        Create a mass-spring system provided an array of Bone objects.
        Every bone is modelled as two masses at the tip points, connected
        via a spring. The beginning mass of the bone is fixed, so that only the
        ending mass is jiggling.

        Simulation mode decides which simulation implementation to be used.
        If 0, the default simulation is used. If 1, point-spring simulation
        will be used. Please do not use simulation_mode=1 if there's no point
        springs.
        """
        # ---------------------------------------------------------------------
        # Precomputation type checks
        # ---------------------------------------------------------------------
        assert type(point_spring) == bool, f"Expected point_spring parameter to \
                                             be boolean. Got {type(point_spring)}."
        if simulation_mode == 1:
            assert point_spring, "Please set point_spring mode if you want to use simulation mode 1."
        
        # TODO: If what you're returning is actually the JOINT locations, why are you
        # calling these variables and functions as BONE locations? What is a location of
        # a bone afterall?
        self.skeleton = skeleton
        self.prev_bone_locations = None
        
        self.POINT_SPRINGS = point_spring
        self.FIXED_SCALE = fixed_scale
        self.SIMULATION_MODE = simulation_mode
        
        self.helper_idxs = helper_idxs
        self.ms_system = MassSpringSystem(dt)
    
        self.helper_lengths = []
        helper_bones = np.array(skeleton.rest_bones)[helper_idxs]
        n_helper = len(helper_bones)
        for i in range(n_helper):
            helper_start = helper_bones[i].start_location
            helper_end = helper_bones[i].end_location
            self.helper_lengths.append(np.linalg.norm(helper_end - helper_start))
            
            if point_spring: # Add zero-length springs at the tip of the helper bone
                free_mass = self.ms_system.add_mass(helper_end, mass=mass, dscale=mass_dscale)
                fixed_mass = self.ms_system.add_mass(helper_end, mass=mass, dscale=mass_dscale)
            else:            # Make the helper bone itself a spring
                free_mass = self.ms_system.add_mass(helper_end, mass=mass, dscale=mass_dscale)
                fixed_mass = self.ms_system.add_mass(helper_start, mass=mass, dscale=mass_dscale)

            self.ms_system.connect_masses(free_mass, 
                                          fixed_mass, 
                                          stiffness = stiffness, 
                                          damping   = damping,
                                          dscale    = spring_dscale)
            self.ms_system.fix_mass(fixed_mass) # Fix the mass that's added to the tip
            
            # Print warnings if the settings are conflicting.
            if np.linalg.norm(helper_start - helper_end) < 1e-5:
                if point_spring is False:
                    print("> WARNING: Point springs found on the simulation.\
                             Consider setting point_spring to True.")
            else:
                if point_spring == True:
                    print(">> WARNING: point_spring setting is true but there \
                              are non-zero springs in the simulation.")
                if simulation_mode == 1:
                    print(">> WARNING: Found non-zero point springs. Reverting \
                              the simulation mode back to 0...")
                    self.SIMULATION_MODE = 0
 
        self.fixed_idx = self.ms_system.fixed_indices
        self.free_idx = self.ms_system.get_free_mass_indices()
        
        # ---------------------------------------------------------------------
        # Post-computation sanity checks 
        # ---------------------------------------------------------------------
        assert len(self.free_idx) == n_helper, f"Expected each jiggle bone to have a single \
                                                 free mass. Got {len(self.free_idx)} masses \
                                                 for {n_helper} jiggle bones."
        
    
    def init_pose(self, theta, trans, degrees):
        """
        Sets the previous bone locations to the given pose. This should be
        called before initiating animation.

        Parameters
        ----------
       theta : np.ndarray
           DESCRIPTION.
       trans : np.ndarray
           DESCRIPTION.
       degrees : bool 
           DESCRIPTION.

        Returns
        -------
        None.
        """

        initial_pose_locations = self.skeleton.pose_bones(theta, trans, degrees=degrees, exclude_root=False)
        assert self.prev_bone_locations.shape == initial_pose_locations.shape
    
        if self.prev_bone_locations is None:
            self.prev_bone_locations = initial_pose_locations
        return initial_pose_locations
    
    def reset_rig(self):
        """
        Reset the spring rig simulation.

        Returns
        -------
        None.

        """
        self.prev_bone_locations = None
        print(">> TODO: reset mass-spring forces and locations too.")
        return
    
    def _adjust_masses(self, rigid_pose_locations):
        """
        Adjust the mass locations after simulating them, such that every bone 
        will preserve its original length.

        Returns
        -------
        adjustments : list
            Depicts the amount of adjustment made on the mass locations.
            It's a list of tuples, where every tuple is (helper_idx, adjust_vector)
        """
        
        adjustments = [] # This is just an informative variable for debugging purposes
        
        if self.POINT_SPRINGS:
            # Loop over the free masses in the system
            for i, free_idx in enumerate(self.free_idx):
                
                free_mass = self.ms_system.masses[free_idx] 
                assert free_mass.mass > 1e-18, f"Expected free mass to have a weight greater than zero, got mass {free_mass.mass}."
                
                helper_idx = self.helper_idxs[i]
                original_length = self.helper_lengths[i]
                bone_start = rigid_pose_locations[2*helper_idx] # There are 2 joint locations per bone
                
                direction = bone_start - free_mass.center
                d_norm = np.linalg.norm(direction) 
                scale = d_norm - original_length
                adjust_vec = (direction/d_norm) * scale # Normalize direction and scale it
                adjustments.append((helper_idx, adjust_vec))
                   
                # Change the free mass location aligned with the bone length.
                self.ms_system.masses[free_idx].center = free_mass.center + adjust_vec
                
                # Sanity check
                new_length = np.linalg.norm(bone_start - self.ms_system.masses[free_idx].center)
                assert np.abs(new_length - original_length) < 1e-4, f"Expected the adjustment function to preserve original bone lengths got length {new_length} instead of {original_length}." 
        else:
            print(">> WARNING: Adjustment for non-point spring bones is not implemented yet.")
        
        return adjustments
    
    
    
    def update(self, theta, trans, degrees, exclude_root, dt=None):
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

    
        rigidly_posed_locations = self.init_pose(theta, trans, degrees=degrees)
        simulated_locations = rigidly_posed_locations.copy() 
        for i, helper_idx in enumerate(self.helper_idxs):
            
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

            if self.SIMULATION_MODE == 1:
                self.ms_system.simulate_zero_length(dt)
            else:
                self.ms_system.simulate(dt)
           
            # Step 1.2 - Adjust the simulation parameters such that helper bones will
            # preserve their original length
            if self.FIXED_SCALE:
                self._adjust_masses(rigidly_posed_locations)
            
            # Step 2 - Get current mass positions
            cur_mass_locations = self.ms_system.get_mass_locations()
            free_mass_locations = cur_mass_locations[self.free_idx]
            
            simulated_locations[helper_end_idx] = free_mass_locations[i]
            self.prev_bone_locations = simulated_locations
                    
        if exclude_root:
            return simulated_locations[2:]
        return simulated_locations
        
    
    
    