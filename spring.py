#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:07:32 2024

Spring simulation based on Position Based Dynamics.

@author: bartu
"""
import numpy as np

class Spring:
    def __init__(self, 
                 connection_coord, 
                 rest_vector=np.array([0.1,0.,0.]), 
                 mass=1.5, 
                 stiffness=50.5, 
                 damper=10.1):
        
        self.step = 0.0
        self.dt = 1./24.
        
        self.mass = mass
        self.w = 1. / self.mass
        
        self.k = stiffness
        self.b = damper
        self.velocity = 0.0
        self.acceleration = 0.0
        
        self.rest_vector = rest_vector
        self.spring_vector = rest_vector
        self.connection_coord = connection_coord
        
        self.mass_rest_coord = self.connection_coord + self.rest_vector
        self.mass_coord = self.mass_rest_coord
        self.delta_x = self.mass_coord - self.mass_rest_coord
        
        
    def update_connection_coord(self, new_connection_coord):
        # Needs to be called every time the connected mass is changed position
        # TODO: could we use pointers for connected mass such that we automatically
        # know the coordinate, that would prevent potential misuse of this function
        # or the miss of this function call?
        self.connection_coord = new_connection_coord
        self.mass_rest_coord = self.connection_coord + self.rest_vector
        return
       
    def simulate(self):
        
        self.delta_x = self.mass_coord - self.mass_rest_coord
        
        spring_force = self.k * self.delta_x 
        damper_force = self.b * self.velocity # Fb = b * x'
        total_force = - (spring_force + damper_force)
        
        self.velocity += self.dt * self.w * total_force
        new_coord = self.mass_coord + (self.velocity * self.dt) 
        
        self.velocity = (new_coord - self.mass_coord) / self.dt
        self.mass_coord = new_coord
        
        return self.mass_coord