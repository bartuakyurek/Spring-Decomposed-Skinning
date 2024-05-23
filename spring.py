#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:07:32 2024

@author: bartu
"""
import numpy as np

class Spring:
    def __init__(self, connection_coord, rest_vector=np.array([1.,0.,0.]), mass=0.01):
        self.step = 0.0
        self.dt = 1./24.
        self.mass = mass
        self.rest_vector = rest_vector
        self.spring_vector = rest_vector
        self.connection_coord = connection_coord
        
    def update_connection(self, new_connection_coord):
        self.connection_coord = new_connection_coord
        #self.spring_vector = self.connection_coord 
    
    def get_mass_coord(self):
        return self.connection_coord + self.spring_vector