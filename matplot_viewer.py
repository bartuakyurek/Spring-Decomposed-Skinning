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
        
        self.xdata, self.ydata = [], []
        self.ln = self.ax.plot([], [], 'ro')
        
    def init(self):
        import numpy as np
        self.ax.set_xlim(0, 2*np.pi)
        self.ax.set_ylim(-1, 1)
        return self.ln
            
    
    def run(self):
        import numpy as np
        ani = FuncAnimation(self.fig, 
                            self.update, 
                            frames=np.linspace(0, 2*np.pi, 128),
                            init_func=self.init, blit=True)
        plt.show()
        

    def update(self, frame):
        self.xdata.append(frame)
        self.ydata.append(np.sin(frame))
        self.ln.set_data(xdata, ydata)
        return ln