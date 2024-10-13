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
    
    def __init__(self, skeleton, helper_indices, mass=1.0, stiffness=100):
        """
        Create a mass-spring system provided an array of Bone objects.
        Every bone is modelled as two masses at the tip points, connected
        via a spring. The beginning mass of the bone is fixed, so that only the
        ending mass is jiggling.

        """
        self.skeleton = skeleton
        helper_bones = np.array(skeleton.rest_bones)[helper_indices]
 
        self.dt = 1. / 24
        self.ms_system = MassSpringSystem(self.dt)
    
        n_helper = len(helper_bones)
        for i in range(n_helper):
            helper_start = helper_bones[i].start_location
            helper_end = helper_bones[i].end_location
            
            mass1 = self.ms_system.add_mass(helper_start, mass=mass)
            mass2 = self.ms_system.add_mass(helper_end, mass=mass)
            
            self.ms_system.connect_masses(mass1, mass2, stiffness=stiffness)
            self.ms_system.fix_mass(mass1)
    
        self.fixed_idx = self.ms_system.fixed_indices
        self.free_idx = self.ms_system.get_free_mass_indices()
    # TODO: We should be able to change the individual masses and stiffness, 
    # for optimization we should be able to provide an array of particle mass
    # that will update the individual Particle.mass in the system
    
    def update(self, current_skeleton):
        
        simulated_locations = None
        
        return simulated_locations
        
    
    
    