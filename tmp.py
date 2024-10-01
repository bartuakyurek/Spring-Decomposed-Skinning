#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 07:31:12 2024

@author: bartu
"""
import numpy as np
import pyvista as pv
from global_vars import _SPACE_DIMS_

MAX_ALLOWED_MASS = 99
class Particle:
    def __init__(self, coordinate, direction=[0., 1., 0.], mass=0.5, radius=0.05):
        
        assert np.any(direction), f"Particle direction must have nonzero length. Provided direction is {direction}."
        assert mass < MAX_ALLOWED_MASS, f"Provided mass {mass} is greater than maximum allowed mass {MAX_ALLOWED_MASS}"
        
        self.mass = mass
        self.radius = radius
        self.direction = np.array(direction)
        self.center = coordinate
        
    def relocate(self, coordinate):
        assert coordinate.shape == (_SPACE_DIMS_,) or coordinate.shape == (_SPACE_DIMS_,1), f"Mass coordinate must be in shape ({_SPACE_DIMS_},1) or ({_SPACE_DIMS_},). Got {coordinate.shape}"
        self.center = coordinate
        return


plotter = pv.Plotter(notebook=False, off_screen=False)
plotter.camera_position = 'zy'
plotter.camera.azimuth = -90

n_masses = 20
for i in range(n_masses):
    mass_particle = Particle(coordinate=np.random.rand(3))
    
    sphere = pv.Sphere(radius=mass_particle.radius, 
                       center=mass_particle.center,
                       direction=mass_particle.direction)

    plotter.add_mesh(sphere)

plotter.show()







# ---------------------------------------------------------------------------
"""
# Define pyramid centre ---> To visualize bones...
bottom_center = sphere.center + (mass_direction * mass_radius)
# Define square points given centre
def _get_square_points(center, normal, diagonal_length):
    half_diagonal = diagonal_length / 2.0
     
    pointa = center + ...
    pointb = center + ...
    pointc = center + ...
    pointd = center + ...
    
# Define pyramid end point(center + spring vector)
pointe = [0.0, 0.0, 1.608]

pyramid = pv.Pyramid([pointa, pointb, pointc, pointd, pointe])
plotter.add_mesh(pyramid)
"""
