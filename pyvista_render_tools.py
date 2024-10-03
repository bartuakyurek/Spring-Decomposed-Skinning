#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:22:09 2024

@author: bartu
"""
import numpy as np
import pyvista as pv

def _get_padded_edges(edges, n_points_per_edge=2):
    # We must "pad" the edges to indicate to vtk how many points per edge
    padding = np.empty(edges.shape[0], int) 
    padding[:] = n_points_per_edge
    edges_w_padding = np.vstack((padding, edges.T)).T
    
    return edges_w_padding

def add_mesh(plotter, verts, faces, triangular=True):
    
    assert triangular and faces.shape[-1] == 3, ">> WARNING: Non-triangular meshes are not supported yet"
        
    faces_w_padding = _get_padded_edges(faces, 3)
    mesh = pv.PolyData(verts, faces)
    
    plotter.add_mesh(
        mesh,
        #scalars=z.ravel(),
        lighting=True,
        show_edges=True,
       
    )
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
