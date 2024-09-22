#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:57:13 2024

@author: bartu
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from viewer import Viewer


class Matplot_Viewer(Viewer):
    def __init__(self):
        super().__init__()
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')
        
    def run(self):
        plt.show()
        

    def animation_callback(self):
        pass