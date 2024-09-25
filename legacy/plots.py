#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:19:53 2024

@author: bartu


DISCLAIMER: Visualization is based on PyVista examples,
            See: https://docs.pyvista.org/version/stable/examples/00-load/create-truss#sphx-glr-examples-00-load-create-truss-py
"""
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

def _add_padded_column(base_array, number_to_pad):
    """

    Parameters
    ----------
    base_array : 2D array of ints
        Depicts connections in a topology, e.g. faces array for a 2D or 3D mesh
    number_to_pad : int
        a single integer to be padded in the first column of base_array

    Returns
    -------
    New 2D array with number_to_pad added to the first column of base_array.
    Use number_to_pad = 2 for edges, = 3 for triangular faces
    """
    padding = np.empty(base_array.shape[0], int) 
    padding[:] = number_to_pad
    return np.vstack((padding, base_array.T)).T
    
def _add_obj_mesh_to_plot(pl, obj_path, opacity=0.8):
    reader = pv.get_reader(obj_path)
    mesh = reader.read()
    pl.add_mesh(mesh, opacity=opacity)

def _add_skeleton_to_plot(pl, nodes, edges):
    # We must "pad" the edges to indicate to vtk how many points per edge
    edges_w_padding = _add_padded_column(edges, 2)
    skel_mesh = pv.PolyData(nodes, edges_w_padding)
    colors = range(edges.shape[0])
    
    pl.add_mesh(skel_mesh, scalars=colors,
                render_lines_as_tubes=True,
                style='wireframe',
                line_width=10,
                cmap='jet',
                show_scalar_bar=False)

# Visualizes a .obj file with corresponding skeleton
# TODO modify the function below to use two of the functions above
# TODO: implement an animating function with the modified function described above
def plot_obj_w_skeleton(obj_path, nodes, edges, obj_opacity=0.8):
   
    pl = pv.Plotter()
    _add_obj_mesh_to_plot(pl, obj_path, opacity=obj_opacity)
    _add_skeleton_to_plot(pl, nodes, edges)
    
    pl.view_xy()
    pl.show()


def plot_verts(verts, faces, opacity=0.8):
    pl = pv.Plotter()
    
    # We must "pad" the edges to indicate to vtk how many points per face
    # We are working with triangular data
   
    mesh = pv.PolyData(verts, _add_padded_column(faces, 3))
    pl.add_mesh(mesh, opacity=opacity)
    
    pl.view_xy()
    pl.show()
    


def matplot_skeleton(joint_locations, kintree):
    """

    Parameters
    ----------
    joint_locations : array
        Array object including joint locations of a skeleton.
    kintree : 2D array
        2D array object of kinematic tree of skeleton, 
        every instance has the joint indices of a bone's 
        beginning joint and ending joint.

    Returns
    -------
    None.
 
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # XYZ coordinate locations
    x = 0   
    y = 1
    z = 2
    for bone in kintree:
        bone_begin_idx, bone_end_idx = bone
        bone_begin_coord = joint_locations[bone_begin_idx]
        bone_end_coord = joint_locations[bone_end_idx]
        
        ax.plot([bone_begin_coord[x], bone_end_coord[x]], 
                [bone_begin_coord[y],bone_end_coord[y]],
                zs=[bone_begin_coord[z],bone_end_coord[z]])
        
        ax.scatter(joint_locations[:,x], joint_locations[:,y], joint_locations[:,z], zdir='z', s=20)