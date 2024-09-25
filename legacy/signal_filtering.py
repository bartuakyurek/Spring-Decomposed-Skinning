#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 08:02:33 2023

@author: bartu
"""

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

import scipy

MESH_VERTEX_ID = 2000
NUM_FRAMES = 239 # TO-DO: do not hardcore it!
DATA_PATH = "/Users/Bartu/Documents/Datasets/DFAUST/50004_jiggle_on_toes/"


def get_coordinate_signals(DATA_PATH):
    x_t, y_t, z_t = [], [], []
    for i in range(NUM_FRAMES):
        
        mesh_path = DATA_PATH + "00"
        
        if i < 10:
            mesh_path += "00"
        elif i < 100:
            mesh_path += "0"
        
        mesh = pv.read(mesh_path + str(i) + ".obj")
        vertex_coor = mesh.points[MESH_VERTEX_ID]
        x_t.append(vertex_coor[0])
        y_t.append(vertex_coor[1])
        z_t.append(vertex_coor[2])
        
    return x_t, y_t, z_t

def draw_same_length_signals(signals, title = ""):
    
    if len(signals) == 0:
        return
    
    t = range(len(signals[0]))
    
    fig = plt.figure()
    gs = fig.add_gridspec(len(signals), hspace=0)
    axs = gs.subplots(sharex=True, sharey=False)
    fig.suptitle(title)
    
    colors = ['tab:blue','tab:orange', 'tab:green']
    for i in range(len(signals)):
        axs[i].plot(t, signals[i], colors[i % 3])

    plt.show()
    
def draw_side_by_side(left_data, right_data, title = ""):
    
    fig = plt.figure()
    fig.suptitle(title)
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(left_data, 'tab:orange')
    ax2.plot(right_data)
    
    plt.show()
    
def pass_filter(filter_type, data, cutoff, sample_rate, poles = 5):
    
    assert filter_type == 'lowpass' or filter_type == 'highpass', "pass_filter() filter_type must be either \'lowpass\' or \'highpass\'"
    sos = scipy.signal.butter(poles, cutoff, filter_type, fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    
    return filtered_data


def get_FFT(signal):
    FFT = np.fft.fft(signal)
    new_N=int(len(FFT)/2) 
    f_nat=1
    freqs = np.linspace(10**-12, f_nat/2, new_N, endpoint=True)
    periods =1.0/(freqs)
    FFT_abs=np.abs(FFT)
    amplitudes = 2*FFT_abs[0:int(len(FFT)/2.)]/len(periods)
    
    return freqs, amplitudes
    


# TODO: if we are using plots.py, move this one there
# else, remove plots.py from the project.
def draw_FFT(freqs, amplitudes, color='black', show=True):
    
    periods =1.0/(freqs)
    
    plt.plot(periods, amplitudes,color=color)
    plt.xlabel('Period ($h$)',fontsize=15)
    plt.ylabel('Amplitude',fontsize= 15)
    plt.title('(Fast) Fourier Transform Method Algorithm',fontsize=15)
    plt.grid(True)
    plt.xlim(0,200)
    if show:
        plt.show()
    
    

if __name__ == "__main__":
    
    # OPTION - 1
    # Draw x/y/z vs. t graphs
    # -------------------------------------------------------
    x_t, y_t, z_t = get_coordinate_signals(DATA_PATH)
    draw_same_length_signals([x_t, y_t, z_t], "Change in coordinates x/y/z vs. frames for vertex " + str(MESH_VERTEX_ID) + " ")
    # -------------------------------------------------------

    # OPTION - 2
    # Apply a filter to the original data
    # -------------------------------------------------------
    #signal = y_t
    #filtered_signal = pass_filter('highpass', signal, cutoff=11, sample_rate=24)
    #draw_same_length_signals([signal, filtered_signal])
    # -------------------------------------------------------

    # OPTION - 3
    # Draw FFT graph
    # -------------------------------------------------------
    # draw_FFT(z_t) --> function changed
    # -------------------------------------------------------

    """
    selected_vert = 4000
    
    target_x = target_verts[:,selected_vert, 0]
    target_y = target_verts[:,selected_vert, 1]
    target_z = target_verts[:,selected_vert, 2]
    
    smpl_x = smpl_verts[:,selected_vert, 0]
    smpl_y = smpl_verts[:,selected_vert, 1]
    smpl_z = smpl_verts[:,selected_vert, 2]
    
    draw_same_length_signals([target_x, target_y, target_z], " Original for vertex " + str(selected_vert))
    draw_same_length_signals([smpl_x, smpl_y, smpl_z], " SMPL for vertex " + str(selected_vert))
    
    target_x_freqs, target_x_amps = get_FFT(target_x)
    smpl_x_freqs, smpl_x_amps = get_FFT(smpl_x)
    
    draw_FFT(target_x_freqs, target_x_amps, show=False)
    draw_FFT(smpl_x_freqs, smpl_x_amps, color='red')
    """

