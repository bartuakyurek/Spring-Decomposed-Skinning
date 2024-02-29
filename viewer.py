#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 11:12:08 2024

@author: bartu
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


class Animation:
    def __init__(self, verts, faces, starting_mesh):
        self.anim_verts = verts
        self.faces = _add_padded_column(faces, 3)
        
        # Expected PyVista mesh type
        self.output = starting_mesh 
        
        # default parameters
        self.kwargs = {
            'current_frame_idx': 0
        }

    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()

    def update(self):
        # This is where you call your simulation
        frame_idx = self.kwargs['current_frame_idx']
        verts = self.anim_verts[frame_idx]
        result = pv.PolyData(verts, self.faces) 
        self.output.copy_from(result)
        
        return

class Viewer:
  def __init__(self):
    self.pl = pv.Plotter()
    

  def add_animated_mesh(self, verts, faces):
      
    starting_mesh = pv.PolyData(verts[0], _add_padded_column(faces, 3))
    self.engine = Animation(verts, faces, starting_mesh)
    self.pl.add_mesh(starting_mesh, opacity=0.8)
    
    
    self.pl.add_slider_widget(
        callback=lambda value: self.engine('current_frame_idx', int(value)),
        rng=[0, verts.shape[0]-1],
        value=0,
        title="Frame Number",
        pointa=(0.025, 0.1),
        pointb=(0.31, 0.1),
        style='modern',
        interaction_event='always'
    )
    
    self.pl.view_xy()
    self.pl.show()
    


    
    

