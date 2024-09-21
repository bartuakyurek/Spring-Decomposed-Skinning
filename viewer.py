#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:46:50 2024

@author: bartu


TODO: this should be not viewer.py but rather a canvas for a single mesh, so that we can write a generic viewer and 
add this canvas to there. The naming is confusing right now. It is pretty specific to spring rig animation. 
"""
from typeguard import typechecked

from scene_node import Scene_Node

class Viewer:
    def __init__(self):
        self.is_animating : bool  = False
        self.dt           : float = 1. / 30
        self.current_frame: int   = 0
        self.max_frames   : int   = 100
        self.nodes        : dict  = {}
        
        self.seperator    : str = "_"
    
    #========== Public Functions ==============================================
    def run(self):
        if self.is_animating:
            self._next_frame()
        
    #=========== Setters ======================================================
    @typechecked
    def set_time_step_in_seconds(self, step : float):
        if step > 0.1:
            print(f">> WARNING: Time step is too large: {step} seconds.")
        self.dt = step
        
    @typechecked
    def set_max_frames(self, cap : int):
        if cap > 500:
            print(f">> WARNING: Maximum number of frames might be too large: {cap} frames.")
        self.max_frames = round(cap)  # In case the input is not an integer value
    
    @typechecked
    # Check the key root which is in the dictionary as "root_001_some_numbers"
    # Vulnerability: If there are node types with the same root, e.g. "Prism_Cube" and "Prism_Cylinder"
    #                then they both will be treated as the same type.
    def __count_key_root_occurence_in_dict(self,
                                           root_name   : str,
                                           key_seperator  : str,
                                           dictionary : dict) -> int:
        n_instance = 0
        for key in self.nodes:
            key_root = key.split(key_seperator)[0]
            
            if key_root == root_name:
                n_instance += 1
                
        return n_instance
        
    @typechecked
    def add_scene_node(self, node):
        
        n_instance = self.__check_key_in_dict(self.nodes, self.seperator, node.node_type)
        node_key = node.get_node_type() + self.seperator + str(n_instance)
        
        self.nodes[node_key] = node
        print(f">> INFO: Added {node_key}")
    
    
    
    #========== Private Functions =============================================
    def _next_frame(self):
        self.current_frame += 1
        pass
        