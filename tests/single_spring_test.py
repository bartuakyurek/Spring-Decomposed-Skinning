#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 07:31:12 2024

@author: bartu
"""
import numpy as np
import pyvista as pv

import __init__
from mass_spring import MassSpringSystem
        
# -------------------------------- MAIN ---------------------------------------
# -----------------------------------------------------------------------------
RENDER = True
plotter = pv.Plotter(notebook=False, off_screen=not RENDER)

# Initiate a mass spring system container
dt = 1. / 24
mass_spring_system = MassSpringSystem(dt)

# Add masses to container and connect them, and fix some of them.
n_masses =  2
mass = 10
mass_spring_system.add_mass(mass_coordinate=np.array([0,0,0]), mass=mass)
mass_spring_system.add_mass(mass_coordinate=np.array([0,0,1.0]), mass=mass)
mass_spring_system.connect_masses(0, 1)
mass_spring_system.fix_mass(0)
    
# Add masses with their initial locations to PyVista Plotter
initial_mass_locations = mass_spring_system.get_mass_locations()
mass_point_cloud = pv.PolyData(initial_mass_locations)
_ = plotter.add_mesh(mass_point_cloud, render_points_as_spheres=True,
                 show_vertices=True)
 
# Add springs connections actors in between to PyVista Plotter
spring_meshes = mass_spring_system.get_spring_meshes()
for spring_mesh in spring_meshes:
    plotter.add_mesh(spring_mesh)
     
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

# Note that "duration" might be misleading, it is not the duration of callback but 
# rather duration of timer that waits before calling the callback function.
dt_milliseconds = int(dt * 1000) 
n_simulation_steps = 500
plotter.add_timer_event(max_steps=n_simulation_steps, duration=dt_milliseconds, callback=callback)

plotter.enable_mesh_picking(left_clicking=True)#, pickable_window=False)
plotter.camera_position = 'zy'

plotter.show()





