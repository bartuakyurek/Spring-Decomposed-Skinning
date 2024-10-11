#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 06:06:11 2024

@author: bartu
"""
import igl
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
mass_spring_system = MassSpringSystem(dt)

# Add masses at vertex locations
sphere_mesh = pv.Sphere(radius=1.0)
vertices = sphere_mesh.points
faces = sphere_mesh.regular_faces
edges = igl.edges(faces)
for i in range(len(vertices)):
    mass_weight = 1
    mass_spring_system.add_mass(mass_coordinate=vertices[i], mass=mass_weight)
    
    
for i in range(len(vertices)):
    mass_spring_system.add_mass(mass_coordinate=vertices[i], mass=mass_weight) 
    #mass_spring_system.fix_mass(i)
    # These masses should be invisible 
    # TODO: add invisibility option to second mass 

# Connect every vertex to zero length spring
for i in range(len(vertices)):
    mass_spring_system.connect_masses(int(i), int(len(vertices)+i), stiffness=100)
    
    

# Apply rigid transformation (rotation, translation) to the whole mesh
t = np.array([0.4, 0.1, -0.2])
for i in range(len(vertices)):
    mass_spring_system.translate_mass(i, t)


# -----------------------------------------------------------------------------
# Add mass and spring PolyData to PyVista Plotter.
# -----------------------------------------------------------------------------
# TODO: this chunk of code is often copy-pasted between tests, why don't you 
# create a subroutine for them?
RENDER = True
plotter = pv.Plotter(notebook=False, off_screen=not RENDER)

initial_mass_locations = mass_spring_system.get_mass_locations()
mass_point_cloud = pv.PolyData(initial_mass_locations, faces=sphere_mesh.faces)
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
n_simulation_steps = 50
plotter.add_timer_event(max_steps=n_simulation_steps, duration=dt_milliseconds, callback=callback)

plotter.enable_mesh_picking(left_clicking=True)#, pickable_window=False)
plotter.camera_position = 'zx'

plotter.show()