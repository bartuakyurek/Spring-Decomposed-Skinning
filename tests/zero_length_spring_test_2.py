#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 06:06:11 2024

@author: bartu
"""
import numpy as np
import pyvista as pv

import __init__
from mass_spring import MassSpringSystem
        
# -------------------------------- MAIN ---------------------------------------
# -----------------------------------------------------------------------------
# Create zero-length spring for this test.
# -----------------------------------------------------------------------------

# Create a mass spring system container
dt = 1. / 24
MASS = 1.
ENABLE_GRAVITY = True
APPLY_FORCE_STEP = 25 # Set to X for applying force every X frames 
STOP_FORCE_STEP = 100 # Set to the step where you want to stop applying force
FORCE_SCALE = 1. / 10 # Scale the np.random to apply a small force

# Add masses at vertex locations
mass_spring_system = MassSpringSystem(dt)
spring_location = np.array([0.,0.,0.])
for i in range(2): # Add two masses at the same location
    mass_spring_system.add_mass(mass_coordinate=spring_location, 
                                mass=MASS,
                                gravity=ENABLE_GRAVITY)

# Connect masses with springs and
# fix either one of the masses connected to zero-length spring
mass_spring_system.connect_masses(int(0), int(1), stiffness=100)
mass_spring_system.fix_mass(0)


# Apply rigid transformation (rotation, translation) to fixed mass
t = np.array([0.1, 0.1, -0.2])
#mass_spring_system.translate_mass(0, t)


# -----------------------------------------------------------------------------
# Add mass and spring PolyData to PyVista Plotter.
# -----------------------------------------------------------------------------
# TODO: this chunk of code is often copy-pasted between tests, why don't you 
# create a subroutine for them?
RENDER = True
plotter = pv.Plotter(notebook=False, off_screen=not RENDER)

initial_mass_locations = mass_spring_system.get_mass_locations()
mass_point_cloud = pv.PolyData(initial_mass_locations)
_ = plotter.add_mesh(mass_point_cloud, render_points_as_spheres=True,
                 show_vertices=True)
 
# Add springs connections actors in between to PyVista Plotter
spring_meshes = mass_spring_system.get_spring_meshes()
for spring_mesh in spring_meshes:
    plotter.add_mesh(spring_mesh)
    
# -----------------------------------------------------------------------------
# Define simulation loop every time PyVista timer calls this callback.
# -----------------------------------------------------------------------------
def callback(step):
   
    if ((step+1) % 50) == 0:
        print(">> Step ", step+1)
        
    if (step) % APPLY_FORCE_STEP == 0 and step < STOP_FORCE_STEP:
        print(">> Force applied... at step ", step)
        #mass_spring_system.translate_mass(0, np.random.randn(3) * FORCE_SCALE)
        mass_spring_system.masses[1].velocity += np.random.randn(3) * FORCE_SCALE

    SIMULATION_TYPE = 0    # TODO: Refactor this.
    if SIMULATION_TYPE == 0:
        mass_spring_system.simulate()
    else:
     mass_spring_system.simulate_zero_length()

    # Step 2 - Get current mass positions and update rendered particles
    cur_mass_locations = mass_spring_system.get_mass_locations()
    mass_point_cloud.points = cur_mass_locations 
    
    # Step 3 - Update the renderd connections based on new locations
    for i, mass_idx_tuple in enumerate(mass_spring_system.connections):
        spring_meshes[i].points[0] = cur_mass_locations[mass_idx_tuple[0]]
        spring_meshes[i].points[1] = cur_mass_locations[mass_idx_tuple[1]]

# -----------------------------------------------------------------------------
# Add timer event and set options for plotter before show().
# -----------------------------------------------------------------------------
# Note that "duration" might be misleading, it is not the duration of callback but 
# rather duration of timer that waits before calling the callback function.
dt_milliseconds = int(dt * 1000) 
n_simulation_steps = 500
plotter.add_timer_event(max_steps=n_simulation_steps, duration=dt_milliseconds, callback=callback)

plotter.enable_mesh_picking(left_clicking=True)#, pickable_window=False)
plotter.camera_position = 'zx'

plotter.show()