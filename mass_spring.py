#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 10:07:32 2024

Spring simulation based on Position Based Dynamics.


DISCLAIMER: The simulation code is heavily based on these sources:
- https://github.com/rlguy/mass_spring_system
- https://rodolphe-vaillant.fr/entry/138/introduction-jiggle-physics-mesh-deformer

@author: bartu
"""
import numpy as np
import pyvista as pv

from sanity_check import _is_equal
from global_vars import _SPACE_DIMS_, VERBOSE

_DEFAULT_STIFFNESS = 1.5
_DEFAULT_DAMPING = 0.5
_DEFAULT_MASS = 2.5
_DEFAULT_SPRING_SCALE = 1.
_DEFAULT_MASS_SCALE = 1.
class Particle:
    def __init__(self, 
                 coordinate, 
                 orientation=[0., 1., 0.], 
                 mass=_DEFAULT_MASS, 
                 dscale=_DEFAULT_MASS_SCALE,
                 radius=0.05,
                 gravity=False):
        
        MAX_ALLOWED_MASS = 99
        assert np.any(orientation), f"Particle orientation vector must have nonzero length. Provided direction is {orientation}."
        assert mass < MAX_ALLOWED_MASS, f"Provided mass {mass} is greater than maximum allowed mass {MAX_ALLOWED_MASS}"
        
        self.mass = mass
        self.radius = radius
        self.gravity = gravity
        self.dscale = dscale
        
        self.center = np.array(coordinate, dtype=float)
        self.prev_center = np.array(coordinate, dtype=float)  # WARNING: it might be misleading cause prev center must be set manually 
        self.orientation = np.array(orientation, dtype=float) # Used for rendering the mass sphere
        
        self.velocity = np.zeros_like(coordinate)
        self.springs = []
        
    def add_spring(self, s):
        self.springs.append(s)
        return
    
    def get_total_spring_forces(self):
        # No gravity or other external forces exist in the current system.
        tot_force = np.zeros_like(self.velocity, dtype=float)
        if self.mass < 1e-20:
            return tot_force # If mass is zero, force is zero by f = ma
        
        for spring in self.springs:
            f_spring =  spring.get_force_on_mass(self)
            assert f_spring.shape == tot_force.shape, f"Calculated force must be a 3D vector, provided {f_spring.shape}."
            tot_force += f_spring
            
        if self.gravity:
            tot_force += self.mass * np.array([0.0, -9.81, 0.0])  # F = mg where g is graviational acceleration
            
        return tot_force
        
class Spring:
    def __init__(self, 
                 beginning_mass : Particle, 
                 ending_mass : Particle,
                 stiffness : float, 
                 damping : float,
                 dscale : float = _DEFAULT_SPRING_SCALE,
                 verbose : bool = VERBOSE
                 ):
        
        # TODO: Are you going to implement squared norms for optimized performance?
        self.k = stiffness
        self.kd = damping
        self.distance_scale = dscale
        self.rest_length = np.linalg.norm(beginning_mass.center - ending_mass.center)
        
        if verbose:
            if self.rest_length < 1e-20:
                print(">> WARNING: Spring initialized at length zero.")

        self.m1 = beginning_mass
        self.m2 = ending_mass
        
    def get_force_on_mass(self, mass : Particle, verbose=VERBOSE):
        
        distance = np.linalg.norm(self.m1.center - self.m2.center)
        
        if verbose:
            if np.abs(distance - self.rest_length) < 1e-8:
               if np.linalg.norm(self.m1.velocity) + np.linalg.norm(self.m2.velocity) < 1e-8: 
                   print(f">>> Balance reached at distance {np.round(distance,6)} with spring rest length {np.round(self.rest_length,6)}")
        
        spring_force_amount  = (distance - self.rest_length) * self.k * self.distance_scale
        
        if distance < 1e-10:
            distance = 1e-10
            
        # Find speed of contraction/expansion for damping force
        normalized_dir = (self.m2.center - self.m1.center) / distance
        assert np.linalg.norm(normalized_dir) < 1.0+1e-12, f"ERROR: Expected normalized direction provided {normalized_dir} is not normalized."
        assert np.linalg.norm(normalized_dir) > 1.0-1e-12, f"ERROR: Expected normalized direction provided {normalized_dir} is not normalized."

        s1 = np.dot(self.m1.velocity, normalized_dir)
        s2 = np.dot(self.m2.velocity, normalized_dir)
        damping_force_amount = -self.kd * (s1 + s2)
        
        force = None
        if self.m1 == mass:
            force = (spring_force_amount + damping_force_amount) * normalized_dir
        elif self.m2 == mass:
            force = (-spring_force_amount + damping_force_amount) * normalized_dir
        else:
            print(">> WARNING: Unexpected case occured, given mass location does not exist for this spring. No force is exerted.")
              
        assert not np.any(force > 1e10), f"WARNING: System got unstable with force {force}, stopping execution..."
        return force
        
        
class MassSpringSystem:
    def __init__(self, dt):
        print(">> Initiated empty mass-spring system")
        self.masses = []
        self.fixed_indices = []
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
            acc = force / self.masses[i].mass
        
            velocity = self.masses[i].velocity + acc * dt
            previous_position = self.masses[i].center.copy()
           
            self.masses[i].prev_center = previous_position
            self.masses[i].center += velocity * dt * self.masses[i].dscale
            self.masses[i].velocity = (self.masses[i].center - previous_position) / dt
    
    
    def simulate_zero_length(self, dt):
        """
        WARNING: This assumes the spring is zero-length spring. I.e. masses
        of the spring are at the same location.

        Parameters
        ----------
        dt : float, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """
        if dt is None: dt = 1.0
        assert dt <= 1.0, f"Please provide a smaller time step, expected <= 1.0, got {dt}."
        
        n_masses = len(self.masses)
        for i in range(n_masses):
            # Constraint: If a mass is zero, i.e. fixed mass, don't exert any force (f=ma=0).
            if self.masses[i].mass < 1e-12:
                continue
            else:
                m = self.masses[i]
                p_current = m.center
                velocity = p_current - m.prev_center
                p_dragged = p_current + velocity
                
                forces = np.zeros(3)
                for spring in m.springs:
                    if spring.m1 == m:
                        target = spring.m2.center
                    elif spring.m2 == m:
                        target = spring.m1.center
                    else:
                        print(">> Unexpected copy error occured.")
                    
                    damping = spring.kd
                    stiffness = spring.k
                    
                    f_d = velocity * (1.0 - damping)
                    f_s = (target - p_dragged) * stiffness
                    forces += f_d + f_s
                
                p_new = p_current + forces
                
                self.masses[i].center = p_new
                self.masses[i].prev_center = p_current
                
                #previous_position = self.masses[i].center.copy()
                #self.masses[i].velocity = (p_new - previous_position) #/ dt
            
    
    def add_mass(self, mass_coordinate, mass=_DEFAULT_MASS, dscale=_DEFAULT_MASS_SCALE,
                 gravity=False, verbose=VERBOSE):
        mass = Particle(mass_coordinate, mass=mass, dscale=dscale, gravity=gravity)
        
        if type(mass) is Particle:
            if verbose: print(f">> Added mass at {mass.center}")
            self.masses.append(mass)
            return len(self.masses) - 1  # Return the index of the appended mass
        else:
            print(f"Expected Particle class, got {type(mass)}")
            
    def fix_mass(self, mass_idx, verbose=VERBOSE):
        self.masses[mass_idx].mass = 0.0
        self.fixed_indices.append(mass_idx)
        if verbose: print(f">> Fixed mass at location {self.masses[mass_idx].center}")
        return
    
    def get_free_mass_indices(self):
        indices = np.arange(0, len(self.masses))
        free_mass_indices = np.delete(indices, self.fixed_indices)
        return np.array(free_mass_indices)
        
    def remove_mass(self, mass_idx):
        # TODO: remove mass dictionary entry
        print("WARNING: This function remove_mass() have not been implemented yet.")
        pass
    
    def translate_mass(self, mass_idx, translate_vec):
        # TODO: why don't you write a typecheck function in sanity.py?
        assert type(mass_idx) is int, f"Expected mass_idx to be int, got {type(mass_idx)}"
        assert mass_idx < len(self.masses), "Provided mass index is out of bounds."
        
        self.masses[mass_idx].prev_center = self.masses[mass_idx].center
        self.masses[mass_idx].center += translate_vec
        return
    
    def update_mass_location(self, mass_idx, new_location):
        if type(mass_idx) is int:
            assert mass_idx < len(self.masses)
            self.masses[mass_idx].center = new_location
        else:
            print(">> Please provide a valid mass index as type int.")
    
    def connect_masses(self, 
                       first_mass_idx : int, 
                       second_mass_idx : int, 
                       stiffness : float = _DEFAULT_STIFFNESS, 
                       damping : float = _DEFAULT_DAMPING,
                       dscale : float = _DEFAULT_SPRING_SCALE):
        
        assert type(first_mass_idx) == int and type(second_mass_idx) == int, f"Expected type int, got {type(first_mass_idx)}."
        assert first_mass_idx != second_mass_idx, "Cannot connect particle to itself."
        
        spring = Spring(self.masses[first_mass_idx], 
                        self.masses[second_mass_idx], 
                        stiffness=stiffness, damping=damping, dscale=dscale)
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
    
    def get_spring_meshes(self):
        # TODO: return a zigzag mesh instead of a straight line
        meshes = []
        for line in self.connections:
            mass_first = self.masses[line[0]]
            mass_second = self.masses[line[1]]
            
            connection_mesh =  pv.Line(mass_first.center, mass_second.center)
            meshes.append(connection_mesh)
        return meshes