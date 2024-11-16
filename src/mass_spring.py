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
import math
import numpy as np
import pyvista as pv
from numpy import linalg as LA

from .utils.sanity_check import _is_equal
from .global_vars import _SPACE_DIMS_, VERBOSE

_DEFAULT_STIFFNESS = 1.5
_DEFAULT_DAMPING = 0.5
_DEFAULT_MASS = 2.5
_DEFAULT_SPRING_SCALE = 1.
_DEFAULT_MASS_SCALE = 1.

# =============================================================================
# Particle Class
# =============================================================================
class Particle:
    def __init__(self, 
                 coordinate, 
                 mass=_DEFAULT_MASS, 
                 dscale=_DEFAULT_MASS_SCALE,
                 radius=0.05,
                 gravity=False):
        """
        Point particle class to represent masses in a mass spring system.

        Parameters
        ----------
        coordinate : np.ndarray
            3D coordinate of the particle.
        mass : TYPE, optional
            DESCRIPTION. The default is _DEFAULT_MASS.
        dscale : TYPE, optional
            DESCRIPTION. The default is _DEFAULT_MASS_SCALE.
        radius : TYPE, optional
            DESCRIPTION. The default is 0.05.
        gravity : bool or np.ndarray or List, optional
            Sets the gravitational acceleration of the particle.
            It can be either provided as a 3d vector or can be set True
            to set it [0.0, 0.0, -9.81]. The default is False.

        Returns
        -------
        None.

        """
        MAX_ALLOWED_MASS = 999
        assert mass < MAX_ALLOWED_MASS, f"Provided mass {mass} is greater than maximum allowed mass {MAX_ALLOWED_MASS}"
        
        self.mass = mass
        
        if mass > 1e-15: self.w = 1 / mass 
        else: 
            self.w = 0.0
            print(">> WARNING: Found zero mass, initializing its weight to zero.")
        
        self.radius = radius
        self.dscale = dscale
        
        self.center = np.array(coordinate, dtype=float)
        self.prev_center = np.array(coordinate, dtype=float)  # WARNING: it might be misleading cause prev center must be set manually 
        
        self.velocity = np.zeros(_SPACE_DIMS_)
        self.springs = []
        self.distace_constraint = None
        if gravity is not None and gravity is not False:
            # If gravity is set to True:
            if gravity is True:
                self.gravity = np.array([0.0, 0.0, -9.81])
            # If gravity is provided via an array:
            else: 
                if type(gravity) is list or type(gravity) is np.ndarray:
                    assert len(gravity) == 3, f"Expected gravity to have length 3, got {len(gravity)}"
                    self.gravity = np.array(gravity)
        # If gravity is set to False:
        else:
            self.gravity = np.array([0.0, 0.0, 0.0]) # Set no gravitational acceleration.
    
    def get_opposite_mass(self, spring_idx):
        spr = self.springs[spring_idx]
        if spr.m1 == self:
            return True, spr.m2
        elif spr.m2 == self:
            return False, spr.m1
        else:
            raise ValueError("Unexpected case. The mass not found on the spring.")
        
    def add_spring(self, spring):
        self.springs.append(spring)
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
            
        
        tot_force += self.mass * self.gravity # F = mg where g is graviational acceleration
        return tot_force

# =============================================================================
# Spring Class
# =============================================================================
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
        self.rest_length = LA.norm(beginning_mass.center - ending_mass.center)
        
        if verbose:
            if self.rest_length < 1e-20:
                print(">> WARNING: Spring initialized at length zero.")

        self.m1 = beginning_mass
        self.m2 = ending_mass
        
        
        
    def get_force_on_mass(self, mass : Particle, tol=1e-12, verbose=VERBOSE):
        
        distance = LA.norm(self.m1.center - self.m2.center)
        spring_force_amount  = (distance - self.rest_length) * self.k * self.distance_scale
        
        # Avoid division by zero
        if distance <  tol:
            distance = tol
            
        # Find the spring direction and normalize it (if it's not a zero vector like in point springs)
        normalized_dir = (self.m2.center - self.m1.center) / distance
        if LA.norm(normalized_dir) > tol: # If the direction is not a zero vector
            assert LA.norm(normalized_dir) < 1.0+tol, f"Expected normalized direction. Provided {normalized_dir} has norm {LA.norm(normalized_dir)}."
            assert LA.norm(normalized_dir) > 1.0-tol, f"Expected normalized direction. Provided {normalized_dir} has norm {LA.norm(normalized_dir)}."
        
        # Find speed of contraction/expansion for damping force
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
        
        # if verbose:
        #     if LA.norm(self.m1.velocity) + LA.norm(self.m2.velocity) < tol: 
        #         print(f">>> Equilibrium reached at distance {np.round(distance,6)} with spring rest length {np.round(self.rest_length,6)}")
            
        assert not np.any(np.abs(force) > 1e10), f"WARNING: System got unstable with force {force}, stopping execution..."
        return force
        
# =============================================================================
# Mass Spring System Class     
# =============================================================================
class MassSpringSystem:
    def __init__(self, dt, mode="PBD", edge_constraint=False):
        print(">> INFO: Initiated empty mass-spring system")
        self.masses = []
        self.fixed_indices = []
        self.connections = []
        self.rest_lengths = [] # Store the rest lengths for quick access in constraint projections
        self.dt =  dt
        
        print(">> INFO: Simulation integrator is set to ", mode)
        self.integration_mode = mode
        self.edge_constraint = edge_constraint
    
    def satisfy_edge_constraints(self, P, alpha, dt=None):
        
        if dt is None: dt = self.dt # Option to set custom time step
        
        for spring_idx, edge in enumerate(self.connections):
            idx1, idx2 = edge
            w1 = self.masses[idx1].w
            w2 = self.masses[idx2].w
            spring_vec = P[idx1] - P[idx2] #P[idx2] - P[idx1]
            spring_len = np.linalg.norm(spring_vec)
            
            if spring_len < 1e-20:
                continue # Avoid division by zero
    
            C = spring_len - self.rest_lengths[spring_idx]
            grad_C1 = spring_vec / spring_len
            grad_C2 = - spring_vec / spring_len
            assert math.isclose(np.linalg.norm(grad_C1), 1.0) # Gradients are 1 for distance constratins
            assert math.isclose(np.linalg.norm(grad_C2), 1.0) # Gradients are 1 for distance constratins
            
            complience = alpha / dt / dt # alpha / dt^2
            grad_sum = w1 + w2  # Gradients are 1 for distance constratins
            lmbd = - C / (grad_sum + complience)
            
            delta_x1 = lmbd * w1 * grad_C1
            delta_x2 = lmbd * w2 * grad_C2
            
            P[idx1] += delta_x1
            P[idx2] += delta_x2
        
        return P
       
        
    def simulate(self, dt=None, integration=None, alpha=0.0):
        """
        Simulate the mass-spring system. Updates the mass locations in the 
        system.
        
        Parameters
        ----------
        dt : float, optional
            Timestep for the integration. When set to None, the time step of the simulator settings will
            be used. The default is None.
            
        integration : str, optional
            Type of integration to be used in the simulation. The default is None.
            If set to None, the default simulator will be used.
            Available options are (case insensitive): {PBD, Verlet, Euler}
            
        Returns
        -------
        None.
        
        """
        if integration is None: integration = self.integration_mode
        
        assert type(integration) == str, f"Expected str type at integration parameter, got {type(integration)}."
        integration = integration.upper()
        
        if integration == "PBD":
                self.simulate_pbd(dt, alpha=alpha)
                
        elif integration == "VERLET":
                 self.simulate_verlet(dt)
        
        elif integration == "EULER":
                 self.simulate_euler(dt)
         
        else:
                print(f"WARNING: Invalid integration scheme {integration} is given. Choosing default simulation...")
                self.simulate_pbd(dt, alpha=alpha)
    
        return
        
    def simulate_pbd(self, dt=None, alpha=0.0):
        """
        Default simulator of mass spring system based on Position Based Dynamics
        (excluding the constraint projections).

        Parameters
        ----------
        dt : float, optional
            Timestep for the integration. When set to None, the time step of the simulator settings will
            be used. The default is None.

        Returns
        -------
        None.

        """
        # Setup variables
        if dt is None:  dt = self.dt
        n_masses = len(self.masses)
        
        # Compute velocities
        velocities = np.empty((n_masses, 3))
        for i in range(n_masses):
            w = self.masses[i].w
            force = self.masses[i].get_total_spring_forces() 
            velocity = self.masses[i].velocity + dt * w * force
            
            # Optionally, damp velocities
            velocity = velocity * self.masses[i].dscale
            velocities[i] = velocity
        
        # Store initially simulated locations in P
        P = np.empty((n_masses, 3))
        for i in range(n_masses):
            P[i] = self.masses[i].center + dt * velocities[i]
        
        # Solve for constraints C (I omit collisions though, only distance is applied) 
        #C  = self.generate_constraints(P)
        #P = self.project_constraints(C, P) # Optionally you could iterate (algorithm line 9 in PBD paper)
        if self.edge_constraint:
            P = self.satisfy_edge_constraints(P, alpha=alpha)
        
        # Update final mass locations and velocities
        for i in range(n_masses):
            velocity = (P[i] - self.masses[i].center) / dt
            self.masses[i].velocity = velocity
            self.masses[i].center = P[i]
            
            
    def simulate_euler(self, dt=None):
        """
        Mass-Spring simulation with explicit Euler Integration (TODO: needs verification).

        Parameters
        ----------
        dt : float, optional
            Timestep for the integration. When set to None, the time step of the simulator settings will
            be used. The default is None.

        Returns
        -------
        None.

        """
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
            #previous_position = self.masses[i].center.copy()            
            #self.masses[i].prev_center = previous_position
            self.masses[i].center += velocity * dt * self.masses[i].dscale
            
    
    def simulate_verlet(self, dt=None):
        """
        Uses Explicit Verlet Integration 
        WARNING: This simulation has not been tested yet so don't rely on t.
        
        Parameters
        ----------
        dt : float, optional
            Timestep for the integration. When set to None, the time step of the simulator settings will
            be used. The default is None.

        Returns
        -------
        None.

        """
        if dt is None: dt = self.dt
        assert dt <= 1.0, f"Please provide a smaller time step, expected <= 1.0, got {dt}."
        
        n_masses = len(self.masses)
        for i in range(n_masses):
            # Constraint: If a mass is zero, i.e. fixed mass, don't exert any force (f=ma=0).
            if self.masses[i].mass < 1e-12:
                continue
            else:
                force = self.masses[i].get_total_spring_forces()
                
                p_prev = self.masses[i].prev_center 
                self.masses[i].prev_center  = self.masses[i].center
                
                p_new =  p_prev + dt * self.masses[i].velocity + force * dt * dt / self.masses[i].mass
                velocity = (p_new - p_prev) / dt
                # Optionlly dampen the velocity 
                velocity = velocity * self.masses[i].dscale
                
                self.masses[i].velocity = velocity
                self.masses[i].center = p_new
                
               
            
    
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
        # Constraint: If a mass is zero, don't exert any force (f=ma=0)
        # that makes the mass fixed in space (world position still can be changed globally)
        # Also this allows us to not divide by zero in the acceleration computation.
        self.masses[mass_idx].mass = 0.0
        self.masses[mass_idx].w = 0.0
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
        self.rest_lengths.append(spring.rest_length)
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
        # TODO: is this function even used? we should remove it.
        meshes = []
        for line in self.connections:
            mass_first = self.masses[line[0]]
            mass_second = self.masses[line[1]]
            
            connection_mesh =  pv.Line(mass_first.center, mass_second.center)
            meshes.append(connection_mesh)
        return meshes