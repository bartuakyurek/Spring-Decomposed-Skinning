#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:57:13 2024

@author: bartu
"""

import matplotlib.pyplot as plt

from viewer import Viewer

class Matplot_Viewer(Viewer):
    def __init__(self):
        super().__init__()

    def run(self):
        print("Is animating? ", self.is_animating)
        plt.show()