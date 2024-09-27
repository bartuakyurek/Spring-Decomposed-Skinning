#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:08:09 2024

@author: bartu
"""

import numpy as np
from mayavi import mlab
from tvtk.api import tvtk
from itertools import count
from dataclasses import dataclass
from tvtk.common import configure_input

from viewer import Viewer


def _create_mayavi_figure(background_color=(1,1,1), size=(800,800)):
    fig = mlab.figure(bgcolor=background_color, size=size)
    fig.scene.z_plus_view()
    return fig

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