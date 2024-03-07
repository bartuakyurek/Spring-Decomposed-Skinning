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

def draw_FFT(signal, color='black', show=True):
    FFT = np.fft.fft(signal)
    new_N=int(len(FFT)/2) 
    f_nat=1
    freqs = np.linspace(10**-12, f_nat/2, new_N, endpoint=True)
    new_Xph=1.0/(freqs)
    FFT_abs=np.abs(FFT)
    
    amplitudes = 2*FFT_abs[0:int(len(FFT)/2.)]/len(new_Xph)
    plt.plot(new_Xph, amplitudes,color=color)
    plt.xlabel('Period ($h$)',fontsize=15)
    plt.ylabel('Amplitude',fontsize= 15)
    plt.title('(Fast) Fourier Transform Method Algorithm',fontsize=15)
    plt.grid(True)
    plt.xlim(0,200)
    if show:
        plt.show()
    
    return freqs, amplitudes

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
    draw_FFT(z_t)
    # -------------------------------------------------------



