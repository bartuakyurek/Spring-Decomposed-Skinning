#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 13:24:36 2024

@author: bartu
"""
import json
import numpy as np
import pyvista as pv

from mass_spring import MassSpringSystem
        
# -------------------------------- MAIN ---------------------------------------
# -----------------------------------------------------------------------------
# Load mass-spring data from .json and .obj files
# -----------------------------------------------------------------------------

data_path = "./data/Mass-Spring/"
filename = "flag"
obj_path = data_path + filename + ".obj"
json_path = data_path + filename +".json"

reader = pv.get_reader(obj_path)
lattice_mesh = reader.read()

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

lattice_verts = lattice_mesh.points
num_masses = lattice_verts.shape[0]
lattice_faces = lattice_mesh.regular_faces 

# Add masses at vertex locations
for i in range(num_masses):
    mass_weight = 1
    mass_spring_system.add_mass(mass_coordinate=lattice_verts[i], mass=mass_weight)

# Add springs at the edges
for face in lattice_faces:
    for f in range(len(face)-1):
        mass_spring_system.connect_masses(int(face[f]), int(face[f+1]), stiffness=k)

for idx in fixed_pts:
    mass_spring_system.fix_mass(idx)

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
n_simulation_steps = 200
plotter.add_timer_event(max_steps=n_simulation_steps, duration=dt_milliseconds, callback=callback)

plotter.enable_mesh_picking(left_clicking=True)#, pickable_window=False)
plotter.camera_position = 'zy'

plotter.show()





