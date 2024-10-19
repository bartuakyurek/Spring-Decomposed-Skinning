#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 10:05:05 2024

@author: bartu
"""
import numpy as np
from numpy import linalg as LA
from skeleton import Skeleton, Bone
from mass_spring import MassSpringSystem

class HelperBonesHandler:
    
    def _set_simulator(self, mode, dt):
        self.simulator = MassSpringSystem(dt)
        
        if mode == 1:
            self.simulate_rig = self.simulator.simulate_zero_length
        elif mode == 0:
            self.simulate_rig = self.simulator.simulate
        else:
            raise ValueError("Unexpected simulation mode. Please provide either 0 or 1.")
    
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
        self.prev_sim_locations = None
        
        self.POINT_SPRINGS = point_spring
        self.FIXED_SCALE = fixed_scale
        
        self.helper_idxs = np.array(helper_idxs, dtype=int)
        self._set_simulator(simulation_mode, dt)
            
        self.helper_lengths = []
        helper_bones = np.array(skeleton.rest_bones)[helper_idxs]
        n_helper = len(helper_bones)
        for i in range(n_helper):
            helper_start = helper_bones[i].start_location
            helper_end = helper_bones[i].end_location
            self.helper_lengths.append(np.linalg.norm(helper_end - helper_start))
            
            if point_spring: # Add zero-length springs at the tip of the helper bone
                free_mass = self.simulator.add_mass(helper_end, mass=mass, dscale=mass_dscale)
                fixed_mass = self.simulator.add_mass(helper_end, mass=mass, dscale=mass_dscale)
            else:            # Make the helper bone itself a spring
                free_mass = self.simulator.add_mass(helper_end, mass=mass, dscale=mass_dscale)
                fixed_mass = self.simulator.add_mass(helper_start, mass=mass, dscale=mass_dscale)

            self.simulator.connect_masses(free_mass, 
                                          fixed_mass, 
                                          stiffness = stiffness, 
                                          damping   = damping,
                                          dscale    = spring_dscale)
            self.simulator.fix_mass(fixed_mass) # Fix the mass that's added to the tip
            
            # Print warnings if the settings are conflicting.
            if np.linalg.norm(helper_start - helper_end) < 1e-5:
                if point_spring is False:
                    print("> WARNING: Point springs found on the simulation.\
                             Consider setting point_spring to True.")
            else:
                if point_spring == True:
                    print(">> WARNING: point_spring setting is true but helper \
                              bone has a positive length. The point springs at the \
                              tip will be invisible.")
                if simulation_mode == 1:
                    print(">> WARNING: Found non-zero point springs. Reverting \
                              the simulation mode back to 0...")
                    self.SIMULATION_MODE = 0
 
        self.fixed_idxs = self.simulator.fixed_indices
        self.free_idxs = self.simulator.get_free_mass_indices()
        
        # ---------------------------------------------------------------------
        # Post-computation sanity checks 
        # ---------------------------------------------------------------------
        assert len(self.free_idxs) == n_helper, f"Expected each jiggle bone to have a single \
                                                 free mass. Got {len(self.free_idxs)} masses \
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
        
        if self.prev_sim_locations is None:
            self.prev_sim_locations = initial_pose_locations
     
        return initial_pose_locations
    
    def reset_rig(self):
        """
        Reset the spring rig simulation.

        Returns
        -------
        None.

        """
        self.prev_sim_locations = None
        print(">> TODO: reset mass-spring forces and locations too.")
        return
    
    def _preserve_bone_length(self, bone_start : np.ndarray,  
                                    free_mass_idx  : int, 
                                    original_length : float ):
        
        assert type(free_mass_idx) is int, f"Expected free mass type int, got {type(free_mass_idx)}"
        
        free_mass = self.simulator.masses[free_mass_idx] 
        assert free_mass.mass > 1e-18, f"Expected free mass to have a weight greater than zero, got mass {free_mass.mass}."
        
        direction = bone_start - free_mass.center
        d_norm = np.linalg.norm(direction) 
        scale = d_norm - original_length
        adjust_vec = (direction/d_norm) * scale # Normalize direction and scale it
        
        # Change the free mass location aligned with the bone length.
        self.simulator.masses[free_mass_idx].center = free_mass.center + adjust_vec
        
        # Sanity check
        new_length = np.linalg.norm(bone_start - self.simulator.masses[free_mass_idx].center)
        assert np.abs(new_length - original_length) < 1e-4, f"Expected the adjustment function to preserve original bone lengths got length {new_length} instead of {original_length}." 
        
        return adjust_vec
    
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
            for i, free_idx in enumerate(self.free_idxs):
                
                helper_idx = self.helper_idxs[i]
                orig_length = self.helper_lengths[i]
                bone_start = rigid_pose_locations[2*helper_idx] # There are 2 joint locations per bone
                
                adjust_vec = self._preserve_bone_length(bone_start, free_idx, orig_length)
                adjustments.append((helper_idx, adjust_vec))
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
            Axis-angle representation of relative rotations for each bone.
        degrees : bool 
            Set True if the provided theta is in degrees, False if in radians.
        exclude_root : bool
            Set True if you don't want to get the invisible root bone locations.
            Otherwise set False if you also want root bone's endpoints for rendering.
            
            TODO: it should be removed in future releases and renderer should decide
            how to render the root node.
        Returns
        -------
        simulated_locations : np.ndarray
            Coordinates of the simulated bones. It has shape of (2*n_bones, 3)
            if exclude_root is set to False. If exclude_root is True, the returned
            array has shape (2*(n_bones-1), 3) where every consecutive points 
            are defining two endpoints of a bone, where bone is a line segment.
        """
        # ---------------------------------------------------------------------
        # Precomputation checks
        # ---------------------------------------------------------------------
        if type(theta) is not np.ndarray:
            if type(theta) is list:
                theta = np.array(theta)
            else:
                print(f"WARNING: Theta is expected to be either list or np.ndarray type, got {type(theta)}.")
        assert type(degrees) == bool, f"Expected degrees parameter to have type bool, got {type(degrees)}"
        assert type(exclude_root) == bool, f"Expected exclude_root parameter to have type bool, got {type(exclude_root)}"


        # ---------------------------------------------------------------------
        # Compute the new mass positions and update helper bone locations
        # ---------------------------------------------------------------------
        # Step 0 - Get the rigidly posed locations as the target
        rigidly_posed_locations = self.init_pose(theta, trans, degrees=degrees)
        simulated_locations = rigidly_posed_locations.copy() 
        
        # WARNING: You're taking the difference data from the rigid skeleton, but what happens
        # if you had a chain of helper bones that are affecting each other? i.e.
        diff = rigidly_posed_locations - self.prev_sim_locations # rigidly_posed_locations is the target. 
        helper_end_idxs = (2 * self.helper_idxs) + 1 # bone locations have 2 joints per bone
        translate_vec = diff[helper_end_idxs]  

        # TODO: how to handle an offset? For now, we assume there's no offset between parent and this bone.
        #offset = self.skeleton.rest_bones[helper_idx].offset
        #translate_vec += offset

        # Step 1 - Translate the fixed masses at the endpoint of each helper bone
        for i, helper_idx in enumerate(self.helper_idxs):
            self.simulator.translate_mass(self.fixed_idxs[i], translate_vec[i])
        
        # Step 2 - Simulate the mass spring system with the new mass locations
        self.simulate_rig(dt)
           
        # Step 2.2 - Adjust the simulation parameters such that helper
        #  bones preserve their original length. (optional)
        # TODO: Shouldn't it be after moving the child bones?
        if self.FIXED_SCALE:
            print(">> HOLD ON: THIS IS NOT SANITY CHECKED YET.")
            self._adjust_masses(rigidly_posed_locations)
            
        # Step 3 - Get simulated mass positions
        cur_mass_locations = self.simulator.get_mass_locations()
        free_mass_locations = cur_mass_locations[self.free_idxs] 
        simulated_locations[helper_end_idxs] = free_mass_locations 
        
        # Step 4 - Adjust the bone starting points to parent's simulated end point
        for i, helper_idx in enumerate(self.helper_idxs):            
            bone = self.skeleton.rest_bones[helper_idx]
            parent_end_idx = helper_idx * 2 + 1
            for child in bone.children:
                child_start_idx = child.idx * 2
                simulated_locations[child_start_idx] = simulated_locations[parent_end_idx]
                # TODO: how about if a child has an offset? we should add it.
        
        # Step 5 - Save the simulated locations for the next iteration
        self.prev_sim_locations = simulated_locations
        # ---------------------------------------------------------------------
        # Return checks
        # ---------------------------------------------------------------------
        if exclude_root:
            return simulated_locations[2:]
        return simulated_locations
        
    
    
    