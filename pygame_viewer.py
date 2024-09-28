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
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders

from viewer import Viewer

class PyGameViewer(Viewer):
    def __init__(self, width=600, height=600):
        super().__init__()
        self.DISPLAYSURF = pygame.display.set_mode((width,height))
        
    def render_scene(self):
        super().render_scene()
        self.launch()
        
    def render_node(self, verts, faces):
        pass
        
    def launch(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    #if event.key == pygame.K_r:
                        # reset()
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            glClearColor(0.3, 0.3, 0.3, 1.0)
            
            #timeStep()
            #render_scene()
            pygame.display.flip()
            
        pygame.quit()
        pbd.Timing.printAverageTimes()