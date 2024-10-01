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
        
        self.id_str = str(id(self))
        self.mass = mass
        self.radius = radius
        self.direction = np.array(direction)
        self.center = np.array(coordinate)
                
    def relocate(self, coordinate):
        coordinate = np.array(coordinate)
        assert coordinate.shape == (_SPACE_DIMS_,) or coordinate.shape == (_SPACE_DIMS_,1), f"Mass coordinate must be in shape ({_SPACE_DIMS_},1) or ({_SPACE_DIMS_},). Got {coordinate.shape}"
        self.center = coordinate
        return
    
    def translate(self, translate_vec):
        translate_vec  = np.array(translate_vec)
        # TODO: convert these asserts to check dims in sanity.py
        assert translate_vec.shape == (_SPACE_DIMS_,) or translate_vec.shape == (_SPACE_DIMS_,1), f"Mass coordinate must be in shape ({_SPACE_DIMS_},1) or ({_SPACE_DIMS_},). Got {translate_vec.shape}"        
        self.center += translate_vec
        
    
class MassSpringSystem:
    def __init__(self):
        print(">> Initiated empty mass-spring system")
        self.masses = []
        self.connections = []
        
    def simulate(self):
        # TODO: run spring simulation... 
        pass
    
    def add_mass(self, mass):
        if type(mass) is Particle:
            print(f">> Added mass at {mass.center}")
            self.masses.append(mass)
        else:
            print(f"Expected Particle class, got {type(mass)}")
        
    def remove_mass(self, mass):
        # TODO: remove mass dictionary entry
        pass
    
    def translate_mass(self, mass_idx, translate_vec):
        # TODO: why don't you write a typecheck function in sanity.py?
        assert type(mass_idx) is int, f"Expected mass_idx to be int, got {type(mass_idx)}"
        assert mass_idx < len(self.masses), f"Provided mass index is out of bounds."
        
        prev_location = self.masses[mass_idx].center 
        self.masses[mass_idx].translate(translate_vec)
        #self.masses[mass_idx].center += translate_vec
        print(prev_location - self.masses[mass_idx].center )
        return
    
    def update_mass_location(self, mass_idx, new_location):
        if type(mass_idx) is int:
            # TODO: assert mass exists
            self.masses[mass_idx].relocate(new_location)
    
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
        mass_locations = []
        for mass_particle in self.masses:
            mass_locations.append(mass_particle.center)   
        return mass_locations
    
    def get_particle_meshes(self):
        # TODO: We can't afford to create separate mesh every time the system is updated
        # so write a function that updates system mesh coordinates as well
        # or store meshes and update the stored mesh every time something is added,
        # and update mesh in every update in particles...
        meshes = []
        for mass_particle in self.masses:
            
            sphere = pv.Sphere(radius=mass_particle.radius,center=mass_particle.center, direction=mass_particle.direction)
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
        

# -------------------------------- MAIN --------------------------------------
# ----------------------------------------------------------------------------
plotter = pv.Plotter(notebook=False, off_screen=False)
plotter.camera_position = 'zy'
plotter.camera.azimuth = -90

mass_spring_system = MassSpringSystem()
n_masses =  5
for i in range(n_masses):
    mass_particle = Particle(coordinate=np.random.rand(3))
    mass_spring_system.add_mass(mass_particle)
    
    if i > 0:
        mass_spring_system.connect_masses(i-1, i)
    

particle_meshes = mass_spring_system.get_particle_meshes()
spring_meshes = mass_spring_system.get_spring_meshes()

for spring_mesh in spring_meshes:
    spring_actor = plotter.add_mesh(spring_mesh)

particle_actors = []
for particle_mesh in particle_meshes:
    actor = plotter.add_mesh(particle_mesh)
    particle_actors.append(actor)
 
def callback(step):
    #actor.position = [step / 100.0, step / 100.0, 0]
    SELECTED_MASS = 0
    mass_spring_system.translate_mass(SELECTED_MASS, [4.0,0,0])
    
    mass_spring_system.simulate()
    
    for i, mass in enumerate(mass_spring_system.masses):
        particle_actors[i].position = mass.center
    

plotter.add_timer_event(max_steps=100, duration=100, callback=callback)
cpos = [(0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]

plotter.enable_mesh_picking()
plotter.show(cpos=cpos)

"""
plotter.show()

RENDER=True
n_frames = 200 
for frame in range(n_frames-1):
        force = np.array([0.1, 0, 0])
        # TODO: update particle system
        for mass_id in mass_spring_system.masses:
            mass_tmp = mass_spring_system.masses[mass_id]
            mass_spring_system.update_mass_location(mass_id, mass_tmp.center + force)
            
        pts = mass_spring_system.get_mass_locations()
        for actor in system_actors:
            print(actor)
            
        break
"""
        

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
