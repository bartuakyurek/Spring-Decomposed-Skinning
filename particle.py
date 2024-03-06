#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 13:52:11 2024

@author: bartu
"""


import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, location, scale=0.01):
        self.rest_location = location 
        self.current_location = location
        self.mesh = pv.Sphere(radius=scale, center=location)
        self.motion = Sine()
        
    def update_position(self, t):
        self.current_location = self.motion.value(t)
        return self.current_location

class Sine:
    def __init__(self, T=1, dt=0.01):
        
        self._T = T
        self.f = 1 / self.T
        self.dt = dt
        self._t = np.arange(0, self.T+self.dt, self.dt)

        self.omega = 2 * np.pi * self.f
        self.locations = np.sin(self.omega * self._t)

    @property
    def period(self):
        return self._T
    @period.setter
    def period(self, value):
        self._T = value
        
    @property
    def frequency(self):
        return self.f
    @property
    def time_step(self):
        return self.dt
    
    def value(self, t):
        return np.sin(self.omega * t)
    
    def plot_behavior(self, plot_all=True):
        if plot_all:
            print(">> WARNING: Plotting all dimensions is not implemented yet")
        else:
            plt.plot(self._t, self.locations)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title('Sine Wave Simulation')
            plt.show()
        