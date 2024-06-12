#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:07:32 2024

@author: bartu
"""
import numpy as np

class Spring:
    def __init__(self, connection_coord, rest_vector=np.array([0.2,0.,0.]), mass=5.5, stiffness=20.3, damper=0.1):
        self.step = 0.0
        self.dt = 1./24.
        
        self.mass = mass
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
        
        
    def update_connection(self, new_connection_coord):
        
        #if np.sum(np.square(new_connection_coord - self.connection_coord)) > 1e-10:
        #    self.delta_x = new_connection_coord - self.connection_coord
        #    return
        
        #self.delta_x = new_connection_coord - self.connection_coord
        
        self.connection_coord = new_connection_coord
        self.mass_rest_coord = self.connection_coord + self.rest_vector
        #self.mass_coord =  self.connection_coord +  self.spring_vector #self.mass_rest_coord #+ self.delta_x
        
        #self.delta_x = self.mass_coord - self.mass_rest_coord
    
        #print("MASS COORD ", self.mass_coord)
         
    def simulate(self):
    
        self.delta_x = self.mass_coord - self.mass_rest_coord

        spring_force = self.k * self.delta_x 
        damper_force = self.b * self.velocity # Fb = b * x'
        #print(self.delta_x)
        """
        
        print(">> 1. Velocity ", self.velocity)
        print(">> 1. Acc ", self.acceleration)
        print(">> 1. Spring and Damper F ", spring_force, damper_force)
        """
        
        
        # If we leave the acceleration alone in equation
        # acceleration = - ((b * velocity) + (k * position)) / mass
        self.acceleration = - (spring_force + damper_force) / self.mass 
        self.velocity += (self.acceleration * self.dt)  # Integral(a) = v 
        self.mass_coord += (self.velocity * self.dt) # Integral(v) = x
        
    
        #self.delta_x = self.mass_coord - self.mass_rest_coord
        #print(self.velocity)
        
        """
        print(">> 2. Velocity ", self.velocity)
        print(">> 2. Acc ", self.acceleration)
        print("Mass coord ", self.mass_coord)
        """
        return self.mass_coord