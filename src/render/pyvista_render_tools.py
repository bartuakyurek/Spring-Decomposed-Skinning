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
             show_edges=False,
             return_actor=False, 
             color=[0.8, 0.8, 1.0],
             texture=None,
             pbr=False, 
             metallic=1.0, 
             roughness=0.5,
             smooth_shading=False):
    
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
                                    show_edges=show_edges,
                                    opacity=opacity,
                                    texture=texture,
                                    pbr=pbr, 
                                    metallic=metallic,
                                    roughness=roughness,
                                    smooth_shading=smooth_shading
                                )
    
    if return_actor:
        return mesh, mesh_actor
    
    return mesh

def set_mesh_color_scalars(mesh, scalars, cmap=cm.jet):
    # See this example to update mesh scalars
    # https://docs.pyvista.org/examples/01-filter/collisions.html
    assert type(mesh) is pv.PolyData, f"Expected actor to be a PolyData, got {type(mesh)}"
    assert len(scalars.shape) == 1, f"Expected scalars to have shape (n_verts,). Got {scalars.shape}."
   
    colors = cmap(scalars)[:, 0:3]
    mesh['vert_colors'] = colors
    return


def add_skeleton(plotter, joint_locations, edges, bone_color=None, colors=None, joint_size=20, return_actor=False):
    # pl: PyVista Plotter
    # joint_locations: (n_joints, 3) numpy.ndarray of 3D coordinates per joint
    # edges: (n_edges, 2) numpy.ndarray consisting of two joint indices per edge
    
    edges_w_padding = _get_padded_edges(edges, 2)
    skel_mesh = pv.PolyData(joint_locations, edges_w_padding)
    
    if colors is None:
        colors = range(edges.shape[0])
        cmap = 'jet'
        
    if bone_color:
        colors = np.zeros_like(colors)
        cmap = [bone_color]
        
    tube_actor = plotter.add_mesh(skel_mesh, scalars=colors,
                    render_lines_as_tubes=True,
                    style='wireframe',
                    line_width=10,
                    cmap=cmap,
                    show_scalar_bar=False)
    
    # Add spheres to indicate joints
    sphere_actor = plotter.add_mesh(skel_mesh, 
                    point_size=joint_size,
                    render_points_as_spheres = True,
                    style='points',
                    scalars=colors,
                    cmap=cmap,
                    show_scalar_bar=False)
    
    if return_actor: return skel_mesh,  (tube_actor, sphere_actor)
    return skel_mesh


def color_bones(skel_mesh, n_bones, default_color, alt_idxs=None, alt_color=None):
    """
    

    Parameters
    ----------
    skel_mesh : pv.PolyData
        The bone joints positions in the skeleton.
    n_bones : int
        Number of bones in the skeleton.
    default_color : str or tuple
        Sets the default bone colors.
    alt_idxs : array of int, optional
        If provided, the bones at the indices alt_idxs will be colored to
        alt_color. The default is None.
    alt_color : str or tuple, optional
        If alt_idxs are provided, the indexed bones will be colored
        to alt_color. The default is None.

    Returns
    -------
    cmap : array
        Array holding colors to pass into plotter.add_mesh(cmap=cmap) in PyVista.

    """
    # Define colors
    colors = np.zeros((n_bones))
    if alt_idxs is not None: colors[alt_idxs] = 1.0

    cmap = [default_color, alt_color]
    if np.sum(colors) == n_bones:
        cmap = [alt_color] # If all are spring bones
    if alt_idxs is None: 
        cmap = [default_color]
    
    skel_mesh['colors'] = colors
    return cmap

def add_skeleton_from_Skeleton(plotter, skeleton, alt_idxs=None, is_smpl=False,
                               default_bone_color="#FFFFFF", alt_bone_color="#BEACE6", 
                               joint_size=20,
                               return_actor=False,
                               exclude_root=True):
    # Given the Skeleton instance and helper indices array, this function
    # creates a mesh and colors the bones respectively.
    #print("WARNING: It is assumed helper_idxs includes root bone so they are one index more than the usual. TODO: resolve it...")
    if is_smpl:
        if alt_idxs is not None: alt_idxs = np.array(alt_idxs) - 1
    
    # Define joint-edges
    joint_locations = skeleton.get_rest_bone_locations(exclude_root=exclude_root)
    n_bones = int(len(joint_locations) / 2)
    edges = np.reshape(np.arange(0, 2*n_bones), (n_bones, 2))
    edges_w_padding = _get_padded_edges(edges, 2)
    skel_mesh = pv.PolyData(joint_locations, edges_w_padding)
    
    
    cmap = color_bones(skel_mesh, n_bones, default_bone_color, 
                        alt_idxs=alt_idxs,
                        alt_color=alt_bone_color)
    
    # Add tubes for bones
    bones_actor = plotter.add_mesh(skel_mesh, 
                    render_lines_as_tubes=True,
                    style='wireframe',
                    line_width=10,
                    scalars='colors',
                    cmap = cmap,
                    show_scalar_bar=False)
    
    # Add spheres for joints
    joints_actor = plotter.add_mesh(skel_mesh, 
                    point_size=joint_size,
                    render_points_as_spheres = True,
                    style='points',
                    scalars='colors',
                    cmap = cmap,
                    show_scalar_bar=False)
    
   
    
    if return_actor: return skel_mesh, (bones_actor, joints_actor)
    return skel_mesh


"""
def save_texture_coordinates_as_obj(mesh_polydata):
    mesh_polydata.texture_map_to_plane(inplace=True)
    tc = mesh_polydata.active_texture_coordinates
    
    x, y = np.meshgrid(tc[:,0], tc[:,1])
    z = np.zeros_like(x)
    
    grid = pv.StructuredGrid(x, y, z)
"""
