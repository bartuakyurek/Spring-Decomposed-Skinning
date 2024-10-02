#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 07:31:12 2024

@author: bartu
"""
import numpy as np
import pyvista as pv

from sanity_check import _is_equal
from global_vars import _SPACE_DIMS_

_DEFAULT_STIFFNESS = 0.1
_DEFAULT_DAMPING = 0.1

class Particle:
    def __init__(self, coordinate, orientation=[0., 1., 0.], mass=0.5, radius=0.05):
        
        MAX_ALLOWED_MASS = 99
        assert np.any(orientation), f"Particle orientation vector must have nonzero length. Provided direction is {orientation}."
        assert mass < MAX_ALLOWED_MASS, f"Provided mass {mass} is greater than maximum allowed mass {MAX_ALLOWED_MASS}"
        
        self.mass = mass
        self.radius = radius
        self.center = np.array(coordinate, dtype=float)
        self.orientation = np.array(orientation, dtype=float) # Used for rendering the mass sphere
        
        self.velocity = np.zeros_like(coordinate)
        self.springs = []
        
    def add_spring(self, s):
        self.springs.append(s)
        return
    
    def get_total_spring_forces(self):
        # No gravity or other external forces exist in the current system.
        f_spring = np.zeros_like(self.velocity, dtype=float)
        
        for spring in self.springs:
            f_spring += spring.get_force_on_mass(self)
            
        return f_spring
        
class Spring:
    def __init__(self, 
                 stiffness : float, 
                 damping : float,
                 beginning_mass : Particle, 
                 ending_mass : Particle):
        assert not _is_equal(beginning_mass.center, ending_mass.center), "Expected spring length to be nonzero, provided masses should be located on different coordinates."
        
        # TODO: Are you going to implement squared norms for optimized performance?
        self.k = stiffness
        self.kd = damping
        self.rest_length = np.linalg.norm(beginning_mass.center - ending_mass.center)
        
        self.m1 = beginning_mass
        self.m2 = ending_mass
        
    def get_force_on_mass(self, mass : Particle):
        tot_force = np.zeros_like(mass.center)
        
        distance = np.linalg.norm(self.m1.center - self.m2.center)
        if distance < 1e-16:
            distance = 1e-6 # For numerical stability
            
        scaled_distance = distance * self.k
        
        # Find speed of contraction/expansion for damping force
        normalized_dir = (self.m1.center - self.m2.center) / distance
        s1 = np.dot(self.m1.velocity, normalized_dir)
        s2 = np.dot(self.m2.velocity, normalized_dir)
        scaled_damping = -self.kd * (s1 + s2)
        
        tot_force = scaled_distance + scaled_damping
        if self.m1 == mass:
            return tot_force
        elif self.m2 == mass:
            return -tot_force
        else:
            print(">> WARNING: Unexpected case occured, given mass location does not exist for this spring. No force is exerted.")
            return None 
        
        
class MassSpringSystem:
    def __init__(self, dt):
        print(">> Initiated empty mass-spring system")
        self.masses = []
        self.connections = []
        self.dt =  dt
        
    def simulate(self, dt=None):
        if dt is None:
            dt = self.dt
            
        n_masses = len(self.masses)
        for i in range(n_masses):
            # Constraint: If a mass is zero, don't exert any force (f=ma=0)
            # that makes the mass fixed in space (world position still can be changed globally)
            # Also this allows us to not divide by zero in the acceleration computation.
            if self.masses[i].mass < 1e-12:
                continue
            
            force = self.masses[i].get_total_spring_forces()
            acc = force / self.masses[i].mass;
            
            #self.masses[i].velocity += acc * dt;
            #self.masses[i].center += self.masses[i].velocity * dt
          
            velocity = self.masses[i].velocity + acc * dt;
            previous_position = self.masses[i].center
            
            self.masses[i].center += velocity * dt
            self.masses[i].velocity = self.masses[i].center - previous_position
            
    def add_mass(self, mass_coordinate):
        mass = Particle(mass_coordinate)
        
        if type(mass) is Particle:
            print(f">> Added mass at {mass.center}")
            self.masses.append(mass)
        else:
            print(f"Expected Particle class, got {type(mass)}")
            
    def fix_mass(self, mass_idx):
        self.masses[mass_idx].mass = 0.0
        print(f">> Fixed mass at location {self.masses[mass_idx].center}")
        return
        
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
    
    def connect_masses(self, first_mass_idx : int, second_mass_idx : int, stiffness : float = _DEFAULT_STIFFNESS, damping : float = _DEFAULT_DAMPING):
        assert type(first_mass_idx) == int and type(second_mass_idx) == int
        assert first_mass_idx != second_mass_idx, "Cannot connect particle to itself."
        
        spring = Spring(stiffness, damping, self.masses[first_mass_idx], self.masses[second_mass_idx])
        self.masses[first_mass_idx].add_spring(spring)
        self.masses[second_mass_idx].add_spring(spring)
        self.connections.append([first_mass_idx, second_mass_idx])
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
        meshes = []
        for mass_particle in self.masses:
            
            sphere = pv.Sphere(radius=mass_particle.radius,
                               center=mass_particle.center, 
                               direction=mass_particle.orientation)
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

# Initiate a mass spring system container
dt = 1. / 24
mass_spring_system = MassSpringSystem(dt)

# Add masses to container and connect them
n_masses =  2
mass_spring_system.add_mass(mass_coordinate=np.array([0,0,0]))
mass_spring_system.add_mass(mass_coordinate=np.array([0,1,0]))

mass_spring_system.connect_masses(0, 1)
    
# Fix selected masses
mass_spring_system.fix_mass(0)
    
# Add masses with their initial locations to PyVista Plotter
initial_mass_locations = mass_spring_system.get_mass_locations()
mass_point_cloud = pv.PolyData(initial_mass_locations)
plotter.add_mesh(mass_point_cloud)

# Add springs connections actors in between to PyVista Plotter
spring_meshes = mass_spring_system.get_spring_meshes()
for spring_mesh in spring_meshes:
    plotter.add_mesh(spring_mesh)

def callback(step):
    
    # Step 1 - Apply forces (if any) and simulate
    if(step < 2):
        SELECTED_MASS = 1 
        mass_spring_system.translate_mass(SELECTED_MASS, np.array([0.0,0.0,0.5]))
        
    mass_spring_system.simulate()
    
    # Step 2 - Get current mass positions and update rendered particles
    cur_mass_locations = mass_spring_system.get_mass_locations()
    mass_point_cloud.points = cur_mass_locations
       
    # Step 3 - Update the renderd connections based on new locations
    for i, mass_idx_tuple in enumerate(mass_spring_system.connections):
        spring_meshes[i].points[0] = cur_mass_locations[mass_idx_tuple[0]]
        spring_meshes[i].points[1] = cur_mass_locations[mass_idx_tuple[1]]


# Note that "duration" might be misleading, it is not the duration of callback but 
# rather duration of timer that waits before calling the callback function.
dt_milliseconds = int(dt * 1000)
n_simulation_steps = 5000
plotter.add_timer_event(max_steps=n_simulation_steps, duration=dt_milliseconds, callback=callback)
plotter.enable_mesh_picking()

cam_pos = [(0.0, 0.0, 5.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.0)]
plotter.show(cpos=cam_pos)








