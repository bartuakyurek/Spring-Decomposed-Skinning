#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:46:50 2024

@author: bartu


TODO: this should be not viewer.py but rather a canvas for a single mesh, so that we can write a generic viewer and 
add this canvas to there. The naming is confusing right now. It is pretty specific to spring rig animation. 
"""
from scene_node import Scene_Node

class Viewer:
    def __init__(self):
        self.is_animating : bool  = False
        self.dt           : float = 1. / 30
        self.current_frame: int   = 0
        self.max_frames   : int   = 100
        
        self.nodes        : dict  = {}
    
    #========== Public Functions ==============================================
   
    def run(self):
        if self.is_animating:
            self._next_frame()
        
    #=========== Setters ======================================================
    def set_time_step_in_seconds(self, step):
        self.dt = step
        
    def set_max_frames(self, cap: int):
        self.max_frames = cap
        
    def add_scene_node(self, node):
        self.nodes[node.node_type] = node
        
    def update_scene_node(self, node_key, V, F=None):
        try:
            node = self.nodes[node_key] 
        except KeyError:
            print("ERROR: Scene object not found.")
            
        pass
    
    #========== Private Functions =============================================
    def _next_frame(self):
        self.current_frame += 1
        pass
        