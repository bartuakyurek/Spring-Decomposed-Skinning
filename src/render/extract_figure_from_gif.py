#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is created to extract frames in a GIF and display them
side by side using Matplotlib. The results are intended to be used
in my thesis figures.

The script is intended to be directly run here.

Created on Wed Nov 20 14:05:13 2024
@author: bartu
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


GIF_NAME = "spot_high_PBD.gif" # "spot_helpers_columns.gif"
ROW_GIF_PATH_1 = "../../assets/visualization_gifs/" + GIF_NAME
CROP = [300, -300, None, None]

def _assert_valid_frames(gif_im, requested_frames):
    """
    Check if the given list exceeds the frames in a GIF or not.
    
    gif_im : PIL.Image 
    requested_frames : List of int
    """
    # Check if the requested keyframes exceed the GIF frames
    n_frames = gif_im.n_frames
    assert np.max(requested_frames) < n_frames, f"Given keyframes exceed the number of available {n_frames} frames in the GIF."
    return
    
def extract_gif_frames(gif_path, keyframes, crop=None):
    """
    Extract the images from a gif given the list of frame indices.

    Parameters
    ----------
    gif_path : str
        Path to the GIF to extract images at certain frames.
    keyframes : list or array of int
        Indices of frames in the GIF to be extracted.
    
    crop: tuple
        Tuple of crop coordinates (vertical_start, vertical_end, horizontal_start, horizontal_end)

    Returns
    -------
    imgs : list of np.ndarray
        List of images as numpy arrays to be displayed with matplotlib.
    """
    
    if crop: 
        assert len(crop) == 4, "Expected crop parameter to have 4 integers for (vertical_start, vertical_end, horizontal_start, horizontal_end) coordinates"
        print(">> WARNING: Crop option is not tested yet.")
        
    imgs = [] 
    with Image.open(gif_path) as im:
        _assert_valid_frames(im, keyframes)
        for i in keyframes:
            im.seek(i) #n_frames // num_key_frames * i)
            img_np = np.asarray(im)
            
            if crop:
                v_start, v_end, h_start, h_end = crop
                img_np = img_np[h_start:h_end, v_start:v_end] # Why is this unintuitive?
                
            imgs.append(img_np)
            
    return imgs
            
def _show_subplot_row(axs_row, row_imgs): #, spacing): #, ref_img=None):
    for i, img in enumerate(row_imgs):
        axs_row[i].imshow(img)
    return

def _turn_off_ax_ticks(ax):
    # Hide X and Y axes label marks
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    
    # Hide X and Y axes tick marks
    ax.set_xticks([])
    ax.set_yticks([])
    return

def compose_plot(row_gif_paths, keyframes, crop=None):
                 #v_spacing=0.0, h_spacing=0.0):
                 #ref_imgs=None):
    
    n_rows = len(row_gif_paths)
    n_cols = len(keyframes)
    assert n_rows >= 1 and n_cols >= 1
    #if ref_imgs is not None: n_cols += 1
    #else: ref_imgs = [None for _ in range(n_rows)]
    
    _, axs = plt.subplots(n_rows, n_cols) #, layout='constrained')
    [_turn_off_ax_ticks(ax) for ax in axs]
    
    for i, gif_path in enumerate(row_gif_paths):
        row_imgs = extract_gif_frames(gif_path, keyframes, crop)     
        
        if n_rows == 1: axs_row = axs
        else: axs_row = axs[i]
        _show_subplot_row(axs_row, row_imgs) #, v_spacing)
                
    plt.show()
    return

# def get_image_np(img_path):
#     img = np.asarray(Image.open(img_path))
#     return img

if __name__ == "__main__":
    
    keyframes = [10 * (i+1) for i in range(7)]  # Select the keyframes you want to display
    gif_paths = [ROW_GIF_PATH_1]

    compose_plot(gif_paths, keyframes, CROP)
    
    