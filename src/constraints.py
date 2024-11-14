#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:59:33 2024

@author: bartu
"""
import numpy as np

class Constraint:
    def __init__(self, n_particles, space_dims=3):
        """
        Initialize a general constraint class that is responsible for updating
        the mass positions given the constraint conditions.

        Parameters
        ----------
        n_particles : int
            Number of particles in the system.
        space_dims : int, optional
            Dimensions of space. The default is 3.

        Returns
        -------
        None.

        """
        self.C_error = 0  # Constraint error 
        self.C_grad = np.zeros((n_particles, space_dims)) # Constraint gradient per particle
       
    def solve(self):
        self.C_error = self._error()
        self.C_grad = self._gradient()
    
    def get_corrections(self, mass, masses, alpha, dt):
        lmbd = self._get_constraint_lambda(masses, alpha, dt)
        delta_x = lmbd * mass.w * self.C_grad
        return delta_x
    
    def _error(self):
        pass # Implementation is up to the specific constraint
        
    def _gradient(self):
        pass # Implementation is up to the specific constraint
    
    def _get_constraint_lambda(self, masses, alpha, dt):
        """
        Get the constraint projection scalar Lambda for General Constraint in
        PBD, (see Matthias Müller's great explanation in:
              https://youtu.be/jrociOAYqxA?t=614)

        Parameters
        ----------
        masses : array of Mass
            Masses in the system
        alpha : float
            Value for soft constraints (see https://youtu.be/jrociOAYqxA?t=730).
            Set it to 0 for hard constraints, i.e. the constraint will fully
            update the mass location.
        dt : float
            Time step.

        Returns
        -------
        lmbd : float
            Scalar to be used in projecting constraints on mass locations.

        """        
        grad_sum = 0.0
        n_masses = len(masses)
        squarednorm = lambda x: np.inner(x, x)

        for i in range(n_masses):
            grad_sum += masses[i].w * squarednorm(self.C_grad[i])
        
        complience = alpha / (dt*dt)
        lmbd = -self.C / (grad_sum + complience)
        return lmbd
    

class SpringLengthConstraint(Constraint):
    def __init__(self, n_particles, rest_length):
        self.rest_length = rest_length
    
    def _error(self, springs):
        pass
        
    def _gradient(self):
        pass
