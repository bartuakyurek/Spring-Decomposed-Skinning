#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:24:36 2024

@author: bartu
"""
import igl
import json
import numpy as np
import pyvista as pv

import __init__
from mass_spring import MassSpringSystem
        
# -------------------------------- MAIN ---------------------------------------
# -----------------------------------------------------------------------------
# Load mass-spring data from .json and .obj files
# -----------------------------------------------------------------------------

data_path = "../data/Mass-Spring/"
filename = "horizontal-chain"
obj_path = data_path + filename + ".obj"
json_path = data_path + filename +".json"

# Load the locations and connectivity for mass-springs.
lattice_mesh = igl.read_obj(obj_path)
lattice_verts = lattice_mesh[0]
lattice_faces = lattice_mesh[3]

# Open and read the JSON file for mass-spring parameters.
with open(json_path, 'r') as file:
    data = json.load(file)
    fixed_pts = data['b']
    k = data['k']

# -----------------------------------------------------------------------------
# Create masses. Connect masses together. Fixate some of the masses
# -----------------------------------------------------------------------------
# Declare parameters
TIME_STEP = 1. / 30  # dt
MASS = 10.0
STIFFNESS = 500 # k = 500
DAMPING = 150.0       # Setting it 0.0 explodes the system.
MASS_DSCALE = 0.5     # This slows down the system
SPRING_DSCALE = 5.    # This is for scaling the spring force
GRAVITY = np.array([0.0, 0.0, -9.81]) 

# Initiate a mass spring system container
mass_spring_system = MassSpringSystem(TIME_STEP)

# Add masses at vertex locations
n_masses = lattice_verts.shape[0]
for i in range(n_masses):
    
    mass_spring_system.add_mass(mass_coordinate=lattice_verts[i], 
                                mass=MASS, 
                                gravity=GRAVITY,
                                dscale=MASS_DSCALE)

# Add springs at the edges
for face in lattice_faces:
    for f in range(len(face)-1):
        # For stability, either increase it to 10~ if the system is not moving, 
        # or decrease it < 1.0 if the system is overflowing
        # In this test, try setting it 0.25, you'll see how system overflows slowly 
        mass_spring_system.connect_masses(int(face[f]), int(face[f+1]), 
                                          stiffness=STIFFNESS, 
                                          dscale=SPRING_DSCALE)

# Fix certain masses' motion
#mass_spring_system.fix_mass(n_masses-1)
for idx in fixed_pts:
    mass_spring_system.fix_mass(idx-1)

# -----------------------------------------------------------------------------
# Create renderer
# -----------------------------------------------------------------------------
RENDER = True
plotter = pv.Plotter(notebook=False, off_screen=not RENDER)

# -----------------------------------------------------------------------------
# Add mass and spring PolyData to PyVista Plotter.
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
    
    prev_mass_locations = mass_spring_system.get_mass_locations()
    
    # Step 0 - Apply forces (if any) 
    if(step < 1):
        print(">> Simulation started.")
        print(f">> Force applied at step {step}.")
        SELECTED_MASS = 2 
        mass_spring_system.translate_mass(SELECTED_MASS, np.array([0.,0.,0.]))
        
    if ((step+1) % 50) == 0:
        print(">> Step ", step+1)
        
    # Step 1 - Simulate the system
    try:
        mass_spring_system.simulate()
    except AssertionError:
        plotter.close()
        raise
        
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
dt_milliseconds = int(TIME_STEP * 1000) 
n_simulation_steps = 500
plotter.add_timer_event(max_steps=n_simulation_steps, duration=dt_milliseconds, callback=callback)

plotter.enable_mesh_picking(left_clicking=True)#, pickable_window=False)
plotter.camera_position = 'xz'
plotter.show()





