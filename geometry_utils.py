#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:50:19 2024

@author: bartu
"""
import numpy as np
from sanity_check import _check_or_convert_numpy

def get_perpendicular(vec, scale=1.0):

    x, y, z = vec
    
    if x != 0:
        perp = np.array([-(y + z)/x, 1, 1])
    elif y != 0:
        perp = np.array([1, -(x + z)/y, 1])
    elif z != 0:
        perp = np.array([(1, 1, -(x + y)/z)])
    else:
        raise ValueError(f">> Cannot compute perpendicular vector for vector {vec}.")    
    
    # Rescale the vector with respect to provided scale
    # (normalize first, then scale)
    
    # Sanity check: the vector must be perpendicular to given vector
    assert np.dot(perp, vec) < 1e-20, ">> Caught unexpected error. Returned vector must be perpendicular to given vector."
    return perp

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
    assert type(n_zigzag) is int, f"Expected n_zigzag to be type int, got {type(n_zigzag)}." 
    # -------------------------------------------------------------------------
    # Create arrays to hold zigzag data
    # -------------------------------------------------------------------------
    n_extrema = int(n_zigzag * 2)                   # One up one down per zigzag
    tot_points = n_extrema + 4
    zigzag_edges = np.empty((tot_points-1, 2), dtype=int) 
    zigzag_edges[:,0] = np.arange(tot_points-1)     # Create zigzag edges
    zigzag_edges[:,1] = np.arange(1, tot_points)    # [0,1]
                                                    # [1,2]
                                                    #  ...
                                                    # [i, i+1]
                                                    #  ...
                                                    # [tot_pts, tot_pts-1]
               
    zigzag_axis = end_point - start_point           # Convert offset percent
    axis_norm = np.linalg.norm(zigzag_axis)         # to an actual distance.
    assert np.dot(axis_norm, axis_norm) > 1e-20, f"Please provide valid endpoints. Endpoints must have different coordinates, got {start_point}, and {end_point}."
    
    tot_offset = axis_norm * (offset_percent / 100) # Then convert it to a vector.
    offset_vec = (tot_offset / 2) * (zigzag_axis/axis_norm)  
    
    zigzag_points = np.empty((tot_points, 3))       # Initialize zigzag points
    zigzag_points[0] = start_point                  # with start and end locations.
    zigzag_points[-1] = end_point
    
    # -------------------------------------------------------------------------
    # Compute zigzag points
    # -------------------------------------------------------------------------
    n_maxima = int(n_zigzag) 
    n_minima = int(n_zigzag)
    assert n_extrema == (n_maxima + n_minima), ">> Caught unexpected error."
    
    zig_roots = np.linspace(start_point + offset_vec, 
                            end_point - offset_vec, 
                            n_extrema + 2)
    # Sanity check the number of extrema matches with the array without the tips 
    assert n_extrema == len(zig_roots)-2, ">> Caught unexpected error."
    
    # If number of zigzag is greater than zero, compute the extrema
    if n_zigzag > 0:
        maxima_idxs = np.arange(1, n_extrema+1, step=2)
        minima_idxs = np.arange(2, n_extrema+2, step=2)
        assert len(maxima_idxs) == n_maxima       # Sanity check
        assert len(minima_idxs) == n_minima       # Sanity check
        
        maxima_pts = zig_roots[maxima_idxs]
        minima_pts = zig_roots[minima_idxs]
        assert maxima_pts.shape == (n_maxima, 3), ">> Caught unexpected error."
        assert minima_pts.shape == (n_minima, 3), ">> Caught unexpected error."
        # At this point you could also assert every point lies on the axis line
        # as a sanity check but that'd be overkill for now.

        # Compute the perpendicular vector to relocate the extrema
        zig_vec = get_perpendicular(zigzag_axis, height)
        maxima_pts += zig_vec
        minima_pts -= zig_vec
        zig_roots[maxima_idxs] = maxima_pts
        zig_roots[minima_idxs] = minima_pts
    # -------------------------------------------------------------------------
    # Insert the computed zigzag data ins return
    # -------------------------------------------------------------------------
    zigzag_points[1:-1] = zig_roots
    return zigzag_points, zigzag_edges

def _test_zigzags(start, end, n_zigzag, offset_percent):
    pts, edges = generate_zigzag(start, end, n_zigzag, offset_percent=offset_percent)
    
    print("Points:\n", np.round(pts,4))
    print("Edges:\n", edges)
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_title(f"Zigzag line with {n_zigzag} zigzags and {offset_percent}% offset.")
    
    for edge in edges:
        vec_start = pts[edge[0]]
        vec_end = pts[edge[1]]
        ax.plot([vec_start[0], vec_end[0]], 
                [vec_start[1], vec_end[1]],
                zs=[vec_start[2], vec_end[2]]
                )

    
    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # TODO:  Write a testing function to test the same procedure.
    offset_percents = [0, 1, 10, 25, 99, 100]
    n_zigzags = [0, 1, 2, 5, 10]
    origin = [0, 0, 0]
    end = [10, 0, 0]
    random_start = np.random.randn(3)
    random_end   = np.random.randn(3)

    for offset in offset_percents:
        for n in n_zigzags:
            print(f">> Testing with {n} zigzag...")
            _test_zigzags(start=origin, end=end, 
                          n_zigzag=n, offset_percent=offset)
            
            _test_zigzags(start=random_start, end=random_end, 
                          n_zigzag=n, offset_percent=offset)
    
    

    