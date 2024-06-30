#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:20:39 2024

@author: bartu
"""
import numpy as np
import matplotlib

cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.1, 0.5, 0.5),
                 (0.2, 0.0, 0.0),
                 (0.4, 0.2, 0.2),
                 (0.6, 0.0, 0.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),
        'green':((0.0, 0.0, 0.0),
                 (0.1, 0.0, 0.0),
                 (0.2, 0.0, 0.0),
                 (0.4, 1.0, 1.0),
                 (0.6, 1.0, 1.0),
                 (0.8, 1.0, 1.0),
                 (1.0, 0.0, 0.0)),
        'blue': ((0.0, 0.0, 0.0),
                 (0.1, 0.5, 0.5),
                 (0.2, 1.0, 1.0),
                 (0.4, 1.0, 1.0),
                 (0.6, 0.0, 0.0),
                 (0.8, 0.0, 0.0),
                 (1.0, 0.0, 0.0))}

my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)

def get_color_by_val(dist):
    color = [0., 0., 0.]
    for c in range(3):
        val = dist * 2.0           # Get normalized distance value
        val = (-1 if val > 2.0 else val) # Sanity check to not get >2.0
     
        if val < 1.:
            if c == 0:                                            ## R
                color[c] = 0.
            elif c == 1:                                          ## G
                color[c] = float(val)
            else:                                                 ## B
                color[c] = 1. - colors[f, i, 1] # 1 - green
        else:
            if c == 0:                                            ## R
                color[c] = float(val) - 1.
            elif c == 1:                                          ## G
                color[c] = 1. - colors[f, i, 0] # 1 - red
            else:                                                 ## B
                color[c] = 0.
            
    return color
    
def get_delta_distance(orig_verts, deformed_verts):
    assert orig_verts.shape == deformed_verts.shape, "Input arrays must be in the same shape."
    assert orig_verts.shape[-1] == 3 or orig_verts.shape[-1] == 2, "Expected shape 2 or 3 in axis -1, found {}.".format(orig_verts.shape[1])
    
    dists = np.linalg.norm(orig_verts - deformed_verts, axis=-1)
    return dists
    
def normalize_dists(dists):
    max_dist = np.max(dists, axis=-1)
   
    dists /= max_dist[:,None]
    return dists
    
def load_anim_npz(path):
    # This is just a generic way to open a .npz file, if you like to customize
    # what you save in your animation .npz, you can modify this function to load it properly.
    with np.load(path) as data:
        verts = data['arr_0']
    return verts

    
END_FRAME = 58 # I put a hard cap on animation because the loaded ones are not in the same shape (Blender Scripting's fault...)
rigid_verts = load_anim_npz("./results/anim_rigid.npz")[:END_FRAME] 
jiggle_verts = load_anim_npz("./results/anim_jiggle.npz")[:END_FRAME]


dists = get_delta_distance(rigid_verts, jiggle_verts)
dists = normalize_dists(dists)



############# Some Sanity Checks ########################################################
#print(dists[dists > 0.1])
num_frames = len(dists)
num_verts = jiggle_verts.shape[1]
assert num_verts > 300, "WARNING: found num_verts < 300, probably you should check v.shape"
assert np.sum((rigid_verts-jiggle_verts)**2) > 1e-9, "WARNING: Jiggle verts are the same with rigid verts."

############# GET THE RED-BLUE COLOR MAPPING #############################################
colors = np.empty_like(jiggle_verts)
# Loop over frames
for f in range(num_frames):
    # Loop over verts
    for i in range(num_verts):
        # Loop over color dims
        dist = dists[f, i]
        for c in range(3):
            
            val = dist * 2.0           # Get normalized distance value
            val = (-1 if val > 2.0 else val) # Sanity check to not get >2.0
         
            if val < 1.:
                if c == 0:                                            ## R
                    colors[f, i, c] = 0.
                elif c == 1:                                          ## G
                    colors[f, i, c] = float(val)
                else:                                                 ## B
                    colors[f, i, c] = 1. - colors[f, i, 1] # 1 - green
            else:
                if c == 0:                                            ## R
                    colors[f, i, c] = float(val) - 1.
                elif c == 1:                                          ## G
                    colors[f, i, c] = 1. - colors[f, i, 0] # 1 - red
                else:                                                 ## B
                    colors[f, i, c] = 0.

np.savez("./results/rb_color_mapping.npz", colors)

