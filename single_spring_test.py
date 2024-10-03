#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 07:31:12 2024

@author: bartu
"""
import numpy as np
import pyvista as pv

from mass_spring import MassSpringSystem
        

# -------------------------------- MAIN ---------------------------------------
# -----------------------------------------------------------------------------
RENDER = False
plotter = pv.Plotter(notebook=False, off_screen=not RENDER)

# Initiate a mass spring system container
dt = 1. / 24
mass_spring_system = MassSpringSystem(dt)

# Add masses to container and connect them, and fix some of them.
n_masses =  2
mass_spring_system.add_mass(mass_coordinate=np.array([0,0,0]))
mass_spring_system.add_mass(mass_coordinate=np.array([0,0,1]))
mass_spring_system.connect_masses(0, 1)
mass_spring_system.fix_mass(0)
    
# Add masses with their initial locations to PyVista Plotter
initial_mass_locations = mass_spring_system.get_mass_locations()
mass_point_cloud = pv.PolyData(initial_mass_locations)
_ = plotter.add_mesh(mass_point_cloud, render_points_as_spheres=True,
                 show_vertices=True)
# Affine transform widget duplicate masses, which is not intentional
#plotter.add_affine_transform_widget(mass_actor)

# Add springs connections actors in between to PyVista Plotter
spring_meshes = mass_spring_system.get_spring_meshes()
for spring_mesh in spring_meshes:
    plotter.add_mesh(spring_mesh)
    
# Open a gif
#plotter.open_gif("./results/sample.gif")
plotter.open_movie("./results/sample.mp4")
plotter.write_frame()

n_frames = 50
for frame in range(n_frames-1):
        
    if frame == 0:
        SELECTED_MASS = 1 
        mass_spring_system.translate_mass(SELECTED_MASS, np.array([0.1,0.3,0.4]))
      
    mass_spring_system.simulate()
    cur_mass_locations = mass_spring_system.get_mass_locations()
    mass_point_cloud.points = cur_mass_locations
       
    # Step 3 - Update the renderd connections based on new locations
    for i, mass_idx_tuple in enumerate(mass_spring_system.connections):
        spring_meshes[i].points[0] = cur_mass_locations[mass_idx_tuple[0]]
        spring_meshes[i].points[1] = cur_mass_locations[mass_idx_tuple[1]]
    
    plotter.add_text(f"Iteration: {frame}", name='time-label')
    plotter.write_frame()

# Closes and finalizes movie
print(">> Plotter closed.")
plotter.close()
plotter.deep_clean()


"""
def callback(step):
    
    # Step 1 - Apply forces (if any) and simulate
    if(step < 1):
        print(">> Force applied.")
        SELECTED_MASS = 1 
        mass_spring_system.translate_mass(SELECTED_MASS, np.array([0.0,0.3,0.0]))
        
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
n_simulation_steps = 150
plotter.add_timer_event(max_steps=n_simulation_steps, duration=dt_milliseconds, callback=callback)

plotter.enable_mesh_picking(left_clicking=True)#, pickable_window=False)
plotter.camera_position = 'zy'
#plotter.camera.azimuth = -90
#cam_pos = [(0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
plotter.show()#(cpos=cam_pos)
"""







