#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 07:31:12 2024

@author: bartu
"""
import numpy as np
import pyvista as pv
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
    
    
class MassSpringSystem:
    def __init__(self):
        print(">> Initiated empty mass-spring system")
        self.masses = []
        self.connections = []
        
    def simulate(self):
        # TODO: run spring simulation... 
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
        assert mass_idx < len(self.masses), f"Provided mass index is out of bounds."
        
        self.masses[mass_idx].center += translate_vec
        return
    
    def update_mass_location(self, mass_idx, new_location):
        if type(mass_idx) is int:
            assert mass_idx < len(self.masses)
            self.masses[mass_idx].center = new_location
        else:
            print(">> Please provide a valid mass index as type int.")
    
    def connect_masses(self, first_mass_idx : int, second_mass_idx : int):
        # TODO: check if both masses are in the system
        assert type(first_mass_idx) == int and type(second_mass_idx) == int
        assert first_mass_idx != second_mass_idx, f"Cannot connect particle to itself."
        # TODO: check if connection already exist (m1,m2) or (m2,m1)
        
        self.connections.append((first_mass_idx, second_mass_idx))
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
    
"""
spring_meshes = mass_spring_system.get_spring_meshes()

for spring_mesh in spring_meshes:
    spring_actor = plotter.add_mesh(spring_mesh)
"""

particle_meshes = mass_spring_system.get_particle_meshes()
particle_actors = []
for particle_mesh in particle_meshes:
    actor = plotter.add_mesh(particle_mesh)
    actor.position = particle_mesh.center
    particle_actors.append(actor)
   

def callback(step):
    # Apply forces (if any) and simulate
    SELECTED_MASS = 0    
    mass_spring_system.translate_mass(SELECTED_MASS, np.random.rand(3) * 0.01)
    mass_spring_system.simulate()
    
    # Get current mass positions and update rendered particles
    cur_mass_locations = mass_spring_system.get_mass_locations()
    n_masses = len(mass_spring_system.masses)
    for i in range(n_masses):
        particle_actors[i].position = cur_mass_locations[i]  

    # Update the renderd connections based on new locations
    # TODO: update lines in between
    
plotter.add_timer_event(max_steps=200, duration=500, callback=callback)
cam_pos = [(0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]

plotter.enable_mesh_picking()
plotter.show(cpos=cam_pos)








