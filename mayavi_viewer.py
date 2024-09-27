#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:08:09 2024

@author: bartu
"""

from viewer import Viewer

class Mayavi_Viewer(Viewer):
    def __init__(self):
        super().__init__()
        pass
        
    def run(self):
        
        for node_key in self.nodes:
            node = self.nodes[node_key]
            verts = node.vertices
            faces = node.faces
            
            self.render_node(verts, faces)  
        
        self.launch()      

    def render_node(self, verts, faces):
        
        # create a mesh
        # add mesh to scene
        pass
    
    
    def launch(self):
        pass