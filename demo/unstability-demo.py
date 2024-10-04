#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:24:36 2024

@author: bartu
"""
import os
import sys
import igl
import json
import numpy as np
import pyvista as pv

import __init__
from mass_spring import MassSpringSystem
        
# TODO: Add *args to set n_simulation steps on console?

# -------------------------------- MAIN ---------------------------------------
# -----------------------------------------------------------------------------
# Load mass-spring data from .json and .obj files
# -----------------------------------------------------------------------------

data_path = "../data/Mass-Spring/"
filename = "horizontal-chain"
obj_path = data_path + filename + ".obj"
json_path = data_path + filename +".json"

# Open and read the JSON file
with open(json_path, 'r') as file:
    data = json.load(file)
    fixed_pts = data['b']
    k = data['k']
    
# -----------------------------------------------------------------------------
# Create masses. Connect masses together. Fixate some of the masses
# -----------------------------------------------------------------------------
# Initiate a mass spring system container
dt = 1. / 24
mass_spring_system = MassSpringSystem(dt)

lattice_mesh = igl.read_obj(obj_path)
lattice_verts = lattice_mesh[0]
lattice_faces = lattice_mesh[3]
num_masses = lattice_verts.shape[0]

# Add masses at vertex locations
for i in range(num_masses):
    mass_weight = 1
    mass_spring_system.add_mass(mass_coordinate=lattice_verts[i], 
                                mass=mass_weight, 
                                gravity=True)

# Add springs at the edges
for face in lattice_faces:
    for f in range(len(face)-1):
        spring_scale = 0.25
        # For stability, either increase it to 10~ if the system is not moving, 
        # or decrease it < 1.0 if the system is overflowing
        # In this test, try setting it 0.25, you'll see how system overflows slowly 
        mass_spring_system.connect_masses(int(face[f]), int(face[f+1]), stiffness=k, dscale=spring_scale)

# Fix certain masses' motion
mass_spring_system.fix_mass(num_masses-1)
#for idx in fixed_pts:
#    mass_spring_system.fix_mass(idx-1)

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
    
    # Step 1 - Apply forces (if any) and simulate
    if(step < 1):
        print(">> Simulation started.")
        print(f">> Step {step} - Force applied.")
        SELECTED_MASS = 2 
        mass_spring_system.translate_mass(SELECTED_MASS, np.array([0.0,0.01,0.01]))
    
    if ((step+1) % 50) == 0:
        print(">> Step ", step+1)
    
    if (step+1) >= n_simulation_steps:
        print(">> Simulation ended.")
        
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
n_simulation_steps = 600
plotter.add_timer_event(max_steps=n_simulation_steps, duration=dt_milliseconds, callback=callback)

plotter.enable_mesh_picking(left_clicking=True)#, pickable_window=False)
plotter.camera_position = 'zx'

plotter.show()





