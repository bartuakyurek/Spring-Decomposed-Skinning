#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:22:09 2024

@author: bartu
"""
import numpy as np
import pyvista as pv
from matplotlib import cm

def _get_padded_edges(edges, n_points_per_edge=2):
    # We must "pad" the edges to indicate to vtk how many points per edge
    padding = np.empty(edges.shape[0], int) 
    padding[:] = n_points_per_edge
    edges_w_padding = np.vstack((padding, edges.T)).T
    
    return edges_w_padding

def add_mesh(plotter, verts, faces, triangular=True, opacity=1.0, return_actor=False, color=[0.8, 0.8, 1.0]):
    
    assert triangular and faces.shape[-1] == 3, ">> WARNING: Non-triangular meshes are not supported yet"
    assert verts.shape[1] == 3, f"Expected vertices to have shape (n_verts, 3), got {verts.shape}."
    
    faces_w_padding = _get_padded_edges(faces, 3)
    mesh = pv.PolyData(verts, faces_w_padding)
    
    # Set vertex colors
    
    n_verts = len(verts)
    default_color = np.array(color)
    default_color = np.reshape(default_color, (1,3))
    colors = np.repeat(default_color, n_verts, axis=0)
    mesh['vert_colors'] = colors
    
    #n_verts = len(verts)
    #mesh['weights'] = np.ones((n_verts),dtype=float)
    
    # Add mesh
    mesh_actor = plotter.add_mesh(
                                    mesh,
                                    scalars='vert_colors',
                                    rgb=True,
                                    #scalars = 'weights',
                                    #cmap='jet',
                                    lighting=True,
                                    show_edges=False,
                                    opacity=opacity
                                )
    
    if return_actor:
        return mesh, mesh_actor
    
    return mesh

def set_mesh_color_scalars(plotter, actor, mesh, scalars):
    # See this example to update mesh scalars
    # https://docs.pyvista.org/examples/01-filter/collisions.html
    assert type(actor) is pv.Actor, f"Expected actor to be a PyVista Actor, got {type(actor)}"
    assert len(scalars.shape) == 1, f"Expected scalars to have shape (n_verts,). Got {scalars.shape}."
   
    colors = cm.jet(scalars)[:, 0:3]
    mesh['vert_colors'] = colors
    return

def add_skeleton(plotter, joint_locations, edges, colors=None):
    # pl: PyVista Plotter
    # joint_locations: (n_joints, 3) numpy.ndarray of 3D coordinates per joint
    # edges: (n_edges, 2) numpy.ndarray consisting of two joint indices per edge
    
    edges_w_padding = _get_padded_edges(edges, 2)
    skel_mesh = pv.PolyData(joint_locations, edges_w_padding)
    
    if colors is None:
        colors = range(edges.shape[0])

    plotter.add_mesh(skel_mesh, scalars=colors,
                    render_lines_as_tubes=True,
                    style='wireframe',
                    line_width=10,
                    cmap='jet',
                    show_scalar_bar=False)
    
    return skel_mesh
