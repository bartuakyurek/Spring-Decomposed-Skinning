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


def set_mesh_color(mesh, color):
    """
    Set the mesh vertex colors to a single provided color.

    Parameters
    ----------
    mesh : pv.PolyData
        PyVista mesh whose colors will be updated to a single color.
    color : list or np.ndarray
        RGB color where every element is in range [0, 1]

    Returns
    -------
    None.

    """
    assert type(mesh) is pv.PolyData, f"Expected actor to be a PolyData, got {type(mesh)}"
    assert len(color) == 3, f"Expected color to have length 3, got {len(color)}"
    assert np.min(color) >= 0.0 and np.max(color) <= 1.0, "Expected colors to be in range [0., 1.]"
    n_verts = len(mesh.points)
    default_color = np.array(color)
    default_color = np.reshape(default_color, (1,3))
    colors = np.repeat(default_color, n_verts, axis=0)
    mesh['vert_colors'] = colors
    return

# This is a feature-envy function, we don't really need it because every time I want 
# to pass an extra parameter on plotter.add_mesh I have to modify here too, it's unnecessary
def add_mesh(plotter,
             verts, 
             faces, 
             opacity=1.0, 
             return_actor=False, 
             color=[0.8, 0.8, 1.0],
             texture=None,
             pbr=False, 
             metallic=1.0, 
             roughness=0.5):
    
    assert faces.shape[-1] == 3, f"Non-triangular meshes are not supported yet. Expected faces has shape (n_faces, 3), got {faces.shape}"
    assert verts.shape[-1] == 3, f"Expected vertices to have shape (n_verts, 3), got {verts.shape}."
    
    faces_w_padding = _get_padded_edges(faces, 3)
    mesh = pv.PolyData(verts, faces_w_padding)
    
    # Set vertex colors
    set_mesh_color(mesh, color)
    
    # Add mesh
    mesh_actor = plotter.add_mesh(
                                    mesh,
                                    scalars='vert_colors',
                                    rgb=True,
                                    lighting=True,
                                    show_edges=False,
                                    opacity=opacity,
                                    texture=texture,
                                    pbr=pbr, 
                                    metallic=metallic,
                                    roughness=roughness,
                                )
    
    if return_actor:
        return mesh, mesh_actor
    
    return mesh

def set_mesh_color_scalars(mesh, scalars):
    # See this example to update mesh scalars
    # https://docs.pyvista.org/examples/01-filter/collisions.html
    assert type(mesh) is pv.PolyData, f"Expected actor to be a PolyData, got {type(mesh)}"
    assert len(scalars.shape) == 1, f"Expected scalars to have shape (n_verts,). Got {scalars.shape}."
   
    colors = cm.jet(scalars)[:, 0:3]
    mesh['vert_colors'] = colors
    return


def add_skeleton(plotter, joint_locations, edges, colors=None, return_actor=False):
    # pl: PyVista Plotter
    # joint_locations: (n_joints, 3) numpy.ndarray of 3D coordinates per joint
    # edges: (n_edges, 2) numpy.ndarray consisting of two joint indices per edge
    
    edges_w_padding = _get_padded_edges(edges, 2)
    skel_mesh = pv.PolyData(joint_locations, edges_w_padding)
    
    if colors is None:
        colors = range(edges.shape[0])

    actor = plotter.add_mesh(skel_mesh, scalars=colors,
                    render_lines_as_tubes=True,
                    style='wireframe',
                    line_width=10,
                    cmap='jet',
                    show_scalar_bar=False)
    
    if return_actor: return skel_mesh, actor
    return skel_mesh
