#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:24:36 2024

@author: bartu
"""

import numpy as np
import pyvista as pv

from mass_spring import MassSpringSystem
        
# -------------------------------- MAIN ---------------------------------------
# -----------------------------------------------------------------------------
RENDER = True
plotter = pv.Plotter(notebook=False, off_screen=not RENDER)

# Initiate a mass spring system container
dt = 1. / 24
mass_spring_system = MassSpringSystem(dt)

# -----------------------------------------------------------------------------
# Create masses. Connect masses together. Fixate some of the masses
# -----------------------------------------------------------------------------
n_masses =  20
mass_weight = 10
for i in range(n_masses):
    # Add masses at random locations
    mass_spring_system.add_mass(mass_coordinate=np.random.rand(3), mass=mass_weight)

# TODO: connect masses (maybe you could read a json?)
#mass_spring_system.connect_masses(0, 1)
#mass_spring_system.fix_mass(0)
pass

# -----------------------------------------------------------------------------
# Add masses initially to PyVista Plotter.
# -----------------------------------------------------------------------------
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
    
    # Step 1 - Apply forces (if any) and simulate
    if(step < 1):
        print(">> Simulation started.")
        print(f">> {step} Force applied.")
        SELECTED_MASS = 1 
        mass_spring_system.translate_mass(SELECTED_MASS, np.array([0.0,0.3,-0.2]))
        
    if ((step+1) % 50) == 0:
        print(">> Step ", step)
        
    mass_spring_system.simulate()

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
plotter.camera_position = 'zy'

plotter.show()





