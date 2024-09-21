#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:46:50 2024

@author: bartu


TODO: this should be not viewer.py but rather a canvas for a single mesh, so that we can write a generic viewer and 
add this canvas to there. The naming is confusing right now. It is pretty specific to spring rig animation. 
"""

class Viewer:
    def __init__(self):
        self.is_animating = False
        self.dt = 1. / 30
        
    def run(self):
        pass
        
    def set_time_step_in_seconds(self, step):
        self.dt = step
    
    