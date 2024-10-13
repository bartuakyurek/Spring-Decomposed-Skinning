#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 10:05:05 2024

@author: bartu
"""

from skeleton import Skeleton, Bone
from mass_spring import MassSpringSystem

class HelperBonesHandler:
    
    def __init__(self, helper_bones, mass=1.0, stiffness=100):
        """
        Create a mass-spring system provided an array of Bone objects.
        Every bone is modelled as two masses at the tip points, connected
        via a spring. The beginning mass of the bone is fixed, so that only the
        ending mass is jiggling.

        Parameters
        ----------
        helper_bones : List or Array
            A list of Bone objects that will be converted to single mass-spring
            simulations.
        mass : TYPE, optional
            DESCRIPTION. The default is 1.0.

        Returns
        -------
        None.

        """
        self.dt = 1. / 24
        self.ms_system = MassSpringSystem(self.dt)
    
        n_helper = len(helper_bones)
        for i in range(n_helper):
            m1_idx = self.ms_system.add_mass(mass_coordinate=helper_bones[i].start_location, mass=mass)
            m2_idx = self.ms_system.add_mass(mass_coordinate=helper_bones[i].end_location, mass=mass)
            
            self.ms_system.connect_masses(int(m1_idx), int(m2_idx), stiffness=stiffness)
            self.ms_system.fix_mass(m1_idx)
    
        self.fixed_idx = self.ms_system.fixed_indices
        self.free_idx = self.ms_system.get_free_mass_indices()
    # TODO: We should be able to change the individual masses and stiffness, 
    # for optimization we should be able to provide an array of particle mass
    # that will update the individual Particle.mass in the system
    
    
    
    