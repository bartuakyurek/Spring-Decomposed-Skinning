#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 07:31:12 2024

@author: bartu
"""
import numpy as np
import pyvista as pv

from sanity_check import _check
from global_vars import _SPACE_DIMS_


class Particle:
    def __init__(self, coordinate, direction=[0., 1., 0.], mass=0.5, radius=0.05):
        
        MAX_ALLOWED_MASS = 99
        assert np.any(direction), f"Particle direction must have nonzero length. Provided direction is {direction}."
        assert mass < MAX_ALLOWED_MASS, f"Provided mass {mass} is greater than maximum allowed mass {MAX_ALLOWED_MASS}"
        
        self.mass = mass
        self.radius = radius
        self.center = np.array(coordinate)
        
        self.direction = np.array(direction) # Direction of mass vector, relative to its center.
        self.previous_location = self.center
        
class Spring:
    def __init__(self, stiffness, beginning_point, ending_point):
        assert beginning_point.shape == ending_point.shape
        __check_1D(beginning_point)
        self.k = stiffness
        self.rest_length = beginning_point - ending_point
        
    
class MassSpringSystem:
    def __init__(self):
        print(">> Initiated empty mass-spring system")
        self.masses = []
        self.connections = []
        self.connection_rest_lengths = []
        
    def simulate(self):
        # TODO: run spring simulation...
        # TODO: get particle velocity
        # TODO: update particle's previous location
        
        
        # Constraint: If a mass is zero, don't exert any force (f=ma=0)
        # that makes the mass fixed in space (world position still can be changed globally)
        pass
            
    def add_mass(self, mass_coordinate):
        mass = Particle(mass_coordinate)
        
        if type(mass) is Particle:
            print(f">> Added mass at {mass.center}")
            self.masses.append(mass)
        else:
            print(f"Expected Particle class, got {type(mass)}")
        
    def remove_mass(self, mass_idx):
        # TODO: remove mass dictionary entry
        pass
    
    def translate_mass(self, mass_idx, translate_vec):
        # TODO: why don't you write a typecheck function in sanity.py?
        assert type(mass_idx) is int, f"Expected mass_idx to be int, got {type(mass_idx)}"
        assert mass_idx < len(self.masses), "Provided mass index is out of bounds."
        
        self.masses[mass_idx].center += translate_vec
        return
    
    def update_mass_location(self, mass_idx, new_location):
        if type(mass_idx) is int:
            assert mass_idx < len(self.masses)
            self.masses[mass_idx].center = new_location
        else:
            print(">> Please provide a valid mass index as type int.")
    
    def connect_masses(self, first_mass_idx : int, second_mass_idx : int):
        assert type(first_mass_idx) == int and type(second_mass_idx) == int
        assert first_mass_idx != second_mass_idx, f"Cannot connect particle to itself."
        
        self.connections.append([first_mass_idx, second_mass_idx])
        
        first_mass_location = self.masses[first_mass_idx]
        second_mass_location = self.masses[second_mass_idx]
        difference_vec = first_mass_location.center - second_mass_location.center
        difference_length = np.linalg.norm(difference_vec)
        self.connection_rest_lengths.append(difference_length)
        return
    
    def disconnect_masses(self, mass_first : Particle, mass_second : Particle):
        pass
    
    def get_mass_locations(self):
        # TODO: could we store it dynamically rather than gathering them every time?
        mass_locations = np.zeros((len(self.masses),_SPACE_DIMS_))
        for i, mass_particle in enumerate(self.masses):
            mass_locations[i] = mass_particle.center
        return mass_locations
    
    def get_particle_meshes(self):
        # TODO: We can't afford to create separate mesh every time the system is updated
        # so write a function that updates system mesh coordinates as well
        # or store meshes and update the stored mesh every time something is added,
        # and update mesh in every update in particles...
        meshes = []
        for mass_particle in self.masses:
            
            sphere = pv.Sphere(radius=mass_particle.radius,
                               center=mass_particle.center, 
                               direction=mass_particle.direction)
            meshes.append(sphere)
        
        return meshes
    
    def get_spring_meshes(self):
        # TODO: return a zigzag mesh instead of a straight line
        meshes = []
        for line in self.connections:
            mass_first = self.masses[line[0]]
            mass_second = self.masses[line[1]]
            
            connection_mesh =  pv.Line(mass_first.center, mass_second.center)
            meshes.append(connection_mesh)
        return meshes
        

# -------------------------------- MAIN ---------------------------------------
# -----------------------------------------------------------------------------
plotter = pv.Plotter(notebook=False, off_screen=False)
plotter.camera_position = 'zy'
plotter.camera.azimuth = -90

mass_spring_system = MassSpringSystem()
n_masses =  10
for i in range(n_masses):
    mass_spring_system.add_mass(mass_coordinate=np.random.rand(3))
    
    if i > 0:
        mass_spring_system.connect_masses(i-1, i)
    
# Add masses withh their initial locations to PyVista Plotter
initial_mass_locations = mass_spring_system.get_mass_locations()
mass_point_cloud = pv.PolyData(initial_mass_locations)
plotter.add_mesh(mass_point_cloud)

# Add springs connections actors in between to PyVista Plotter
spring_meshes = mass_spring_system.get_spring_meshes()
for spring_mesh in spring_meshes:
    plotter.add_mesh(spring_mesh)

def callback(step):
    # Step 1 - Apply forces (if any) and simulate
    SELECTED_MASS = 0    
    mass_spring_system.translate_mass(SELECTED_MASS, np.random.rand(3) * 0.01)
    mass_spring_system.simulate()
    
    # Step 2 - Get current mass positions and update rendered particles
    cur_mass_locations = mass_spring_system.get_mass_locations()
    mass_point_cloud.points = cur_mass_locations
       
    # Step 3 - Update the renderd connections based on new locations
    for i, mass_idx_tuple in enumerate(mass_spring_system.connections):
        spring_meshes[i].points[0] = cur_mass_locations[mass_idx_tuple[0]]
        spring_meshes[i].points[1] = cur_mass_locations[mass_idx_tuple[1]]

    
plotter.add_timer_event(max_steps=200, duration=200, callback=callback)
cam_pos = [(0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]

plotter.enable_mesh_picking()
plotter.show(cpos=cam_pos)








