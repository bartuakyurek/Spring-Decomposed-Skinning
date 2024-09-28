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
from pygame_render_tools import initGL

from viewer import Viewer

class PyGameViewer(Viewer):
    def __init__(self, width=600, height=600):
        super().__init__()
        self.WIDTH = width
        self.HEIGHT = height
        self.DISPLAYSURF = pygame.display.set_mode((width,height))
        
    def _render_scene(self):
       super()._render_scene()
        
    def _render_node(self, verts, faces):
        #sim = pbd.Simulation.getCurrent()
        #model = sim.getModel()
        #pd = model.getParticles()   
        #tetModel = model.getTetModels()[0]
        #offset = tetModel.getIndexOffset()
        #drawMesh(pd, tetModel.getSurfaceMesh(), offset, [0,0.2,0.7])
        
        glPushMatrix()
        glLoadIdentity()
        drawText([-0.95,0.9], "Time: {:.2f}".format(pbd.TimeManager.getCurrent().getTime()))
        glPopMatrix()
        
    def time_step(self):
        self.current_frame += 1
        # -> Update mesh normals
        return
    
    def launch(self):
        initGL(self.WIDTH, self.HEIGHT)  
        gluLookAt (5, 4, 15, 5, -1, 0, 0, 1, 0)

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
            
            if self.is_animating:
                self.time_step()
                
            self._render_scene()
            pygame.display.flip()
            
        pygame.quit()
        
if __name__ == '__main__':
    viewer = PyGameViewer()
    viewer.launch()