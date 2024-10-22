#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:50:19 2024

@author: bartu
"""
import numpy as np

def scale_vector(vec, scale):
    """
    A basic function to scale a vector into desired norm.

    Parameters
    ----------
    vec : np.ndarray
        A vector who will be rescaled.
    scale : float
        A scalar to determine the new norm of the given vector.

    Returns
    -------
    scaled_vec : np.ndarray
        A vector with the same shape of the input, that has norm [scale].
    """
    
    scaled_vec = vec / np.linalg.norm(vec)
    scaled_vec *= scale
    assert np.linalg.norm(scaled_vec) - scale < 1e-12,  f">> Caught unexpected error. The scaled vector\
                                                             does not preserve its intended length {scale},\
                                                             it has norm {np.linalg.norm(scaled_vec)}."
    return scaled_vec

def get_perpendicular(vec, scale=1.0):
    """
    Simple function to obtain a perpendicular vector. 

    Parameters
    ----------
    vec : np.ndarray or list
        3D vector which will be perpendicular to the returned vector. It's
        expected to be a non-zero vector.
    scale : float, optional
        Determines the length of the returned vector. The default is 1.0.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    perp : np.ndarray
        3D vector that is perpendicular to the given vector. Two of its 
        dimensions are 1.0.
    """
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
    
    if scale != 0.0:
        scaled_perp = scale_vector(perp, scale)
    else:
        raise ValueError(">> Provided scale must be nonzero.") 
    return scaled_perp


    
    

    