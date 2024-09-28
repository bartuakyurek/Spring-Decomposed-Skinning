#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 09:04:04 2024

@author: bartu

PyGame based viewer to display meshes and associated animations

DISCLAIMER: This file is based on 
https://coderslegacy.com/python/python-pygame-tutorial/

"""

import sys
import pygame
from pygame import QUIT
from viewer import Viewer

class PyGameViewer(Viewer):
    def __init__(self, width=600, height=600):
        super().__init__()
        self.DISPLAYSURF = pygame.display.set_mode((width,height))
        
    def run(self):
        super().run()
        
    def render_node(self, verts, faces):
        pass
        
    def launch(self):
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
            pygame.display.update()