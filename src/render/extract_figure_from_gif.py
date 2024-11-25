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
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# =============================================================================
# 
# =============================================================================
model_name = "smpl_50004_8" #"spot_cc" #"cloth"  
SELECTED_MODELS = [ #f"{model_name}_opaque",
                    f"{model_name}_skel",
                   ]
                   #"cloth_rig1", "cloth_rig3"]
                   #f"{model_name}_skel",
                   #f"{model_name}_cc"]
                   #f"{model_name}_opaque"]
                   #f"{model_name}_cc"] # See datadict for available options

spot_frames     = [1, 21, 31, 41, 51]
monstera_frames = [1, 25, 50, 100, 124, 149, 189]
duck_frames     = [1, 11, 63, 76, 110, 130, 149]
cloth_frames    = [1, 17, 31, 42, 48, 56, 67]
smpl_8_frames     = [1, 29, 47, 60, 65, 78, 87, 106] # pose_id[8]
smpl_10_frames     = [1, 41, 91, 119, 124, 167, 172, 198] # pose_id[10]

# =============================================================================
REL_GIF_PATH = "../../results/gifs/"

datadict = { 
    
    "spot_opaque" : {
                     'gif_path' : REL_GIF_PATH + "spot.gif",
                     'crop'     : [630, -350, None, None],
                     'keyframes':  spot_frames,
    },
    "spot_skel" : {
                     'gif_path' : REL_GIF_PATH + "spot.gif",
                     'crop'     : [330, -600, None, None],
                     'keyframes':  spot_frames,
    },
    
    "spot_cc_opaque" : {
                     'gif_path' : REL_GIF_PATH + "spot_cc.gif",
                     'crop'     : [1050, -450, None, None],
                     'keyframes':  spot_frames,
    },
    "spot_cc_skel" : {
                     'gif_path' : REL_GIF_PATH + "spot_cc.gif",
                     'crop'     : [450, -1050, None, None],
                     'keyframes':  spot_frames,
    },
    
    "spot_exaggerated_opaque" : {
                     'gif_path' : REL_GIF_PATH + "spot_exaggerated.gif",
                     'crop'     : [960, -440, None, None],
                     'keyframes':  spot_frames,
    },
    "spot_exaggerated_skel" : {
                     'gif_path' : REL_GIF_PATH + "spot_exaggerated.gif",
                     'crop'     : [440, -960, None, None],
                     'keyframes':  spot_frames,
    },
    
    "spot_helper_opaque" : {
                     'gif_path' : REL_GIF_PATH + "spot_helper_opaque.gif",
                     'crop'     : [300, -300, None, None],
                     'keyframes':  spot_frames,
    },
    "spot_helper_transparent" : {
                     'gif_path' : REL_GIF_PATH + "spot_helper_transparent.gif",
                     'crop'     : [820, -770, None, None],
                     'keyframes':  spot_frames,
    },
    
    "monstera_shake_skel" : {
                     'gif_path' : REL_GIF_PATH + "monstera_shake_skel.gif",
                     'crop'     : [None, None, None, None],
                     'keyframes':  monstera_frames,
    },
    "monstera_shake_opaque" : {
                     'gif_path' : REL_GIF_PATH + "monstera_shake_opaque.gif",
                     'crop'     : [None, None, None, None],
                     'keyframes': monstera_frames,
    },
    "monstera_shake_cc" : {
                     'gif_path' : REL_GIF_PATH + "monstera_shake_cc.gif",
                     'crop'     : [None, None, None, None],
                     'keyframes':  monstera_frames,
    },
    
    "duck_skel" : {
                     'gif_path' : REL_GIF_PATH + "duck_skel.gif",
                     'crop'     : [None, None, None, None],
                     'keyframes':  duck_frames,
    },
    "duck_opaque" : {
                     'gif_path' : REL_GIF_PATH + "duck_opaque.gif",
                     'crop'     : [None, None, None, None],
                     'keyframes': duck_frames,
    },
    "duck_cc" : {
                     'gif_path' : REL_GIF_PATH + "duck_cc.gif",
                     'crop'     : [None, None, None, None],
                     'keyframes':  duck_frames,
    },
    
    
    "cloth_skel" : {
                     'gif_path' : REL_GIF_PATH + "cloth_skel.gif",
                     'crop'     : [None, None, None, None],
                     'keyframes':  cloth_frames,
    },
    "cloth_opaque" : {
                     'gif_path' : REL_GIF_PATH + "cloth_opaque.gif",
                     'crop'     : [None, None, None, None],
                     'keyframes': cloth_frames,
    },
    "cloth_cc" : {
                     'gif_path' : REL_GIF_PATH + "cloth_cc.gif",
                     'crop'     : [None, None, None, None],
                     'keyframes':  cloth_frames,
    },
    
    "cloth_rig1" : {
                     'gif_path' : REL_GIF_PATH + "cloth_rig1.gif",
                     'crop'     : [None, None, None, None],
                     'keyframes':  cloth_frames,
    },
    "cloth_rig3" : {
                     'gif_path' : REL_GIF_PATH + "cloth_rig3.gif",
                     'crop'     : [None, None, None, None],
                     'keyframes':  cloth_frames,
    },

    
    "smpl_50004_10_cc"     : {
                    'gif_path' : REL_GIF_PATH + "smpl_50004_10_cc.gif",
                    'crop'     : [250, -250, None, None],
                    'keyframes':  smpl_10_frames,
    },
    
    "smpl_50004_10_skel"     : {
                    'gif_path' : REL_GIF_PATH + "smpl_50004_10_skel.gif",
                    'crop'     : [250, -250, None, None],
                    'keyframes':  smpl_10_frames,
    },
    
    "smpl_50004_8_cc"     : {
                    'gif_path' : REL_GIF_PATH + "smpl_50004_8_cc.gif",
                    'crop'     : [250, -250, None, None],
                    'keyframes':  smpl_8_frames,
    },
    
    "smpl_50004_8_skel"     : {
                    'gif_path' : REL_GIF_PATH + "smpl_50004_8_skel.gif",
                    'crop'     : [250, -250, None, None],
                    'keyframes':  smpl_8_frames,
    },
    
    }
# Same across all figures
V_SPACE = 0.0
H_SPACE = 0.0
SAVE_PATH = "../../results/figures/" 
# =============================================================================
def _assert_valid_frames(gif_im, requested_frames):
    """
    Check if the given list exceeds the frames in a GIF or not.
    
    gif_im : PIL.Image 
    requested_frames : List of int
    """
    # Check if the requested keyframes exceed the GIF frames
    n_frames = gif_im.n_frames
    print(">> INFO: Found",n_frames,"frames.")
    assert np.max(requested_frames) < n_frames, f"Given keyframes exceed the number of available {n_frames} frames in the GIF."
    return

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

def _set_ax_boundary(ax, edges=['top', 'right', 'bottom', 'left'], visibility=True):
    """
    Add or remove border lines to a subplot axis. 
    """
    for edge in edges:
        ax.spines[edge].set_visible(visibility)

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
        Tuple of crop coordinates (left_pix, right_pix, top_pix, bottom_pix)
        indicates the indices of the left-right-top-bottom boundary pixels.
        E.g. (100, -150, None, 200) will crop 100 pixels from left, 150 pixels
        from right, and #vertical_pixels - 200 pixels from the bottom (i.e.
        crops bottom until 200th pixel row, if it was -200 then it'd crop 
        200 pixels from the bottom).

    Returns
    -------
    imgs : list of np.ndarray
        List of images as numpy arrays to be displayed with matplotlib.
    """
    
    if crop: 
        assert len(crop) == 4, "Expected crop parameter to have 4 integers for (vertical_start, vertical_end, horizontal_start, horizontal_end) coordinates"
        
    imgs = [] 
    with Image.open(gif_path) as im:
        _assert_valid_frames(im, keyframes)
        for i in keyframes:
            im.seek(i) #n_frames // num_key_frames * i)
            img_np = np.asarray(im, dtype=np.int64)
            
            if crop:
                left_pix, right_pix, top_pix, bottom_pix = crop
                img_np = img_np[top_pix:bottom_pix, left_pix:right_pix] 
                
            imgs.append(img_np)
            
    return imgs

# =============================================================================
#         
# =============================================================================
def compose_plot(row_gif_paths, keyframes, 
                 crop=None, dpi=600, figsize=(12.5, 7.5),
                 v_spacing=0.05, h_spacing=0.1,
                 save_path=None):
    """
    Generate a figure that is composed by a selection of frames of GIFs. 
    This function is intended to be used in figures in my thesis results.

    Parameters
    ----------
    row_gif_paths : list of str
        List of GIF paths to extract images at certain frames.
    keyframes : list or array of int
        Indices of frames in the GIF to be extracted.
    crop : tuple of int, optional
        Represents the boundaries of the gif images. If set to None,
        the images will be taken as a None. To use it provide 
        (left_pixel, right_pixel, top_pixel, bottom_pixel)
        where vertical is for cropping from left and right, horizontal is for
        cropping top and bottom pixels,e.g. (100, -100, None, -400) will
        crop 100 pixels from left, 100 pixels from right, 400 pixels from bottom. 
        The default is None.
    dpi : int, optional
        To specify the dpi resolution of the plot. The higher it is, the higher
        quality the plot is. The default is 1200.
    figsize : tuple of int, optional
        To feed in subplot figsize property (otherwise the resolution drastically drops).
    v_spacing : int, optional
        Vertical spacing between the keyframes. The default is 0.05. 
    h_spacing : int, optional
        Horizontal spacing between the keyframes. The default is 0.1. 
    save_path : str, optional
        If provided, the resulting plot will be saved as a png
        at the save_path. The default is None.

    Returns
    -------
    None.

    """
    
    n_rows = len(row_gif_paths)
    n_cols = len(keyframes)
    assert n_rows >= 1 and n_cols >= 1
    
    plt.figure(dpi=dpi) 
    _, axs = plt.subplots(n_rows, n_cols, figsize=figsize) 
    plt.subplots_adjust(wspace=v_spacing, 
                        hspace=h_spacing)
        
    [_turn_off_ax_ticks(ax) for ax in axs]
    [_set_ax_boundary(ax, visibility=False) for ax in axs] # Remove border lines 
    
    for i, gif_path in enumerate(row_gif_paths):
        row_imgs = extract_gif_frames(gif_path, keyframes, crop)     
        
        if n_rows == 1: axs_row = axs
        else: axs_row = axs[i]
        _show_subplot_row(axs_row, row_imgs) #, v_spacing)
                
    if save_path: plt.savefig(save_path, dpi=dpi)
    plt.show()
    return


# def get_image_np(img_path):
#     img = np.asarray(Image.open(img_path))
#     return img

# =============================================================================
# 
# =============================================================================
if __name__ == "__main__":
    
    for SELECTED_MODEL in SELECTED_MODELS:
        model_data = datadict[SELECTED_MODEL]
        keyframes = model_data['keyframes']  # Select the keyframes you want to display
        gif_paths = [model_data["gif_path"]] 
        crop = model_data["crop"]
        
        compose_plot(gif_paths, keyframes, crop,
                     v_spacing=V_SPACE, h_spacing=H_SPACE,
                     save_path=os.path.join(SAVE_PATH,  f"{SELECTED_MODEL}_result_{len(keyframes)}.png")
                     )
    
    