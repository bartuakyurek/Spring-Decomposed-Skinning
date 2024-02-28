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

# Visualizes a .obj file with corresponding skeleton
# TODO: implement another function that takes SMPL result directly (not .obj file) adds the mesh to the plot
# TODO: implement a function that adds skeleton to the plot 
# TODO modify the function below to use two of the functions above
# TODO: implement an animating function with the modified function described above
def plot_obj_w_skeleton(result_path, nodes, edges):
    """
    
    Parameters
    ----------
    result_path : str
        path of the .obj file
    nodes : array
        3D joint locations of the skeleton
    edges : 2D array
        kinematic tree of the skeleton

    Returns
    -------
    None.
    
    """
    reader = pv.get_reader(result_path)
    mesh = reader.read()
    
    pl = pv.Plotter()
    pl.add_mesh(mesh, opacity=0.6)
    
    # We must "pad" the edges to indicate to vtk how many points per edge
    padding = np.empty(edges.shape[0], int) * 2
    padding[:] = 2
    edges_w_padding = np.vstack((padding, edges.T)).T
 
    skel_mesh = pv.PolyData(nodes, edges_w_padding)
    colors = range(edges.shape[0])
    
    pl.add_mesh(skel_mesh, scalars=colors,
                render_lines_as_tubes=True,
                style='wireframe',
                line_width=10,
                cmap='jet',
                show_scalar_bar=False)
    
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