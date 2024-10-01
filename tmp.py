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
        assert coordinate.shape == (_SPACE_DIMS_,) or coordinate.shape == (_SPACE_DIMS_,1), f"Mass coordinate must be in shape ({_SPACE_DIMS_},1) or ({_SPACE_DIMS_},). Got {coordinate.shape}"
        self.center = coordinate
        return
    
class MassSpringSystem:
    def __init__(self):
        print(">> Initiated empty mass-spring system")
        self.masses = {}
        self.connections = []

    def add_mass(self, mass):
        if type(mass) is Particle:
            print(f">> Added mass at {mass.center}")
            self.masses[mass.id_str] = mass
        else:
            print(f"Expected Particle class, got {type(mass)}")
        
    def remove_mass(self, mass):
        # TODO: remove mass dictionary entry
        pass
    
    def update_mass_location(self, mass, new_location):
        if type(mass) is Particle:
            # TODO: assert mass exists
            self.masses[mass.id_str].center = new_location
        if type(mass) is str:
            self.masses[mass].center = new_location
        else:
            print(">> Please provide a valid mass id string or a Particle object.")
    
    def connect_masses(self, mass_first : Particle, mass_second : Particle):
        # TODO: check if both masses are in the system
        assert mass_first.id_str != mass_second.id_str, f"Cannot connect particle to itself."
        assert type(mass_first) == Particle and type(mass_second) == Particle
        # TODO: check if connection already exist (m1,m2) or (m2,m1)
        
        self.connections.append((mass_first.id_str, mass_second.id_str))
        return
    
    def disconnect_masses(self, mass_first : Particle, mass_second : Particle):
        # TODO: remove dictionary entries
        assert type(mass_first) == Particle and type(mass_second) == Particle
        
        try: 
            self.connections.remove(mass_first.id_str, mass_second.id_str)
            print(f">> Connection {mass_first.center} - {mass_second.center} deleted.")
        except:
            try:
                self.connections.remove(mass_second.id_str, mass_first.id_str)
                print(f">> Connection {mass_second.center} - {mass_first.center} deleted.")
            except:
                print(">> Mass connection {mass_second.center} - {mass_first.center} not found.")
                
    def get_mass_w_index(self, idx):
        key_at_index_i = list(self.masses.keys())[idx]
        return self.masses[key_at_index_i]
    
    def get_mass_locations(self):
        # TODO: could we store it dynamically rather than gathering them every time?
        mass_locations = []
        for key in self.masses:
            mass_particle = self.masses[key]
            mass_locations.append(mass_particle.center)
            
        return mass_locations
            
    def get_system_meshes(self):
        # TODO: We can't afford to create separate mesh every time the system is updated
        # so write a function that updates system mesh coordinates as well
        # or store meshes and update the stored mesh every time something is added,
        # and update mesh in every update in particles...
        meshes = []
        for key in self.masses:
            mass_particle = self.masses[key]
            sphere = pv.Sphere(radius=mass_particle.radius,center=mass_particle.center, direction=mass_particle.direction)
            meshes.append(sphere)
             
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
n_masses = 20
for i in range(n_masses):
    mass_particle = Particle(coordinate=np.random.rand(3))
    mass_spring_system.add_mass(mass_particle)
    
    if i > 0:
        prev_particle = mass_spring_system.get_mass_w_index(i-1)
        mass_spring_system.connect_masses(prev_particle, mass_particle)


system_meshes = mass_spring_system.get_system_meshes()
system_actors = []
for mesh in system_meshes:
    actor = plotter.add_mesh(mesh)
    system_actors.append(actor)
    
plotter.enable_mesh_picking()

def callback(step):
    actor.position = [step / 100.0, step / 100.0, 0]

plotter.add_timer_event(max_steps=200, duration=500, callback=callback)
cpos = [(0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
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
