#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:50:19 2024

@author: bartu
"""
import numpy as np
from sanity_check import _check_or_convert_numpy

def generate_zigzag(start_point : np.ndarray, 
                    end_point   : np.ndarray,
                    n_zigzag    : int = 10,
                    height      : float = 1.0,
                    offset_percent  : float = 10.0,
                    ):
    """
    Generate a zigzag pattern between two points. Returns the vertex and
    edge data that can be used for rendering purposes.

    Parameters
    ----------
    start_point : np.ndarray
        The location vector of the starting point of this zigzag.
    end_point : np.ndarray
        The location vector of the ending point of this zigzag.
    n_zigzag : int, optional
        Number of zigzag lines to be produced. The default is 10.
    height : float, optional
        Height of the zigzag lines that is the halfway between the
        minimum and maximum points of a zigzag. The default is 1.0.
    offset_percent : float, optional
        Determines how much the zigzag that will cover the line in between
        starting and ending point. If set to 0.0 (i.e. 0%) the zigzag will cover
        the whole space between the tips. If set to 100%, the zigzag will not 
        appear at all.
        The default is 10.0, i.e. 10%.

    Returns
    -------
    zigzag_points : np.ndarray
        World locations of the zigzag points that can be used as a vertex data
        in the renderer. The points array has shape (n_vertex, 3).
    zigzag_edges : np.ndarray
        An array to hold connectivity data for the zigzag_points that can be
        used as edge data in the renderer. The edges array has shape (n_edges, 2)
        where n_edges = (n_vertex - 1).

    """
    # -------------------------------------------------------------------------
    # Pre-computation checks
    # -------------------------------------------------------------------------    
    start_point = _check_or_convert_numpy(start_point)
    end_point   = _check_or_convert_numpy(end_point)
    assert start_point.shape == end_point.shape, f"Provided endpoints must have the same shape. Got {start_point.shape} and {end_point.shape}"
    assert offset_percent >= 0.0 and offset_percent <= 100.0, f"Provided offset percentage is expected to be in range [0, 100], got {offset_percent}."
    
    # -------------------------------------------------------------------------
    # Create arrays to hold zigzag data
    # -------------------------------------------------------------------------
    
    zigzag_points = np.empty((n_zigzag + 4, 3)) 
    zigzag_edges = np.empty((n_zigzag + 2 + 1, 2)) # 2 is for offsets
    
    # -------------------------------------------------------------------------
    # Compute zigzag points
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Create zigzag edges
    # -------------------------------------------------------------------------
    
    
    # -------------------------------------------------------------------------
    # Return
    # -------------------------------------------------------------------------
    return zigzag_points, zigzag_edges


if __name__ == "__main__":
    
    print(">> Testing with even points...")
    N_even = 10
    start_origin = [0, 0, 0]
    end_x = [10, 0, 0]
    
    generate_zigzag(start_origin, end_x, n_zigzag=N_even)
    
    
    #print(">> Testing with 0 percent offset...")

    #print(">> Testing with 100 percent offset...")
    # Expecting zigzag to not appear at all 
    
    #Set n_zigzag to 0 and 1, and 2, and 5.

    