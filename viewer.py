#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:46:50 2024

@author: bartu


TODO: this should be not viewer.py but rather a canvas for a single mesh, so that we can write a generic viewer and 
add this canvas to there. The naming is confusing right now. It is pretty specific to spring rig animation. 
"""

from scene_node import Scene_Node
from dict_utils import _count_key_root_occurence_in_dict


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
        
        for node_key in self.nodes:
            node = self.nodes[node_key]
            verts = node.vertices
            faces = node.faces
            
            self.render_node(verts, faces)

        self.launch()
            
    def render_node(self, verts, faces):
        pass
    
    def launch(self):
        pass
    
    def set_time_step_in_seconds(self, step : float):
        if step > 0.1:
            print(f">> WARNING: Time step is too large: {step} seconds.")
        self.dt = step
        

    def set_max_frames(self, cap : int):
        if cap > 500:
            print(f">> WARNING: Maximum number of frames might be too large: {cap} frames.")
        self.max_frames = round(cap)  # In case the input is not an integer value
    

    # Check the key root which is in the dictionary as "root_001_some_numbers"
    # Vulnerability: If there are node types with the same root, e.g. "Prism_Cube" and "Prism_Cylinder"
    #                then they both will be treated as the same type.
        
    def add_scene_node(self, node):
        
        n_instance = _count_key_root_occurence_in_dict(self.nodes, self.seperator, node.node_type)
        node_key = node.get_node_type() + self.seperator + str(n_instance)
        
        self.nodes[node_key] = node
        print(f">> INFO: Added {node_key}")
    
    
    #========== Private Functions =============================================
    def _next_frame(self):
        self.current_frame += 1
        print("INFO: switching to next frame.")
        