#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 09:09:13 2024
@author: bartu

DISCLAIMER: 
# Copyright (c) 2018, Felix Schill.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# vispy: gallery 2

"""
import sys
import numpy as np

from vispy import scene
from vispy.color import Color
from vispy.visuals.transforms import STTransform


class MassVisual(scene.visuals.Compound):

    def __init__(self, 
                 center, 
                 size=10.0, 
                 editable=True,
                 selectable=True,
                 on_select_callback=None,
                 callback_argument=None,
                 *args, **kwargs): 
        
        scene.visuals.Compound.__init__(self, [], *args, **kwargs)
        self.unfreeze()

        if type(center) is not np.ndarray:
            center = np.array(center)
    
        self.center = np.reshape(center, (1,-1))
        self.sphere = scene.visuals.Markers(
                                            pos = self.center,
                                            spherical=True,
                                            size=size,
                                            antialias=0,
                                            face_color=Color("#e88834"),
                                            edge_color='white',
                                            parent=self
                                            )
        
        self.editable = editable
        self._selectable = selectable
        self._on_select_callback = on_select_callback
        self._callback_argument = callback_argument

        self.freeze()


class Canvas(scene.SceneCanvas):
    """ A simple test canvas for drawing demo """
    
    def __init__(self, width=800, height=800, bgcolor='#ab43d3'):
       
        scene.SceneCanvas.__init__(self, keys='interactive',
                                   size=(width, height), 
                                   bgcolor=bgcolor,
                                   show=True)

        self.unfreeze()

        self.view = self.central_widget.add_view()
        self.view.camera = 'arcball'
        self.view.camera.set_range(x=[-5, 5])

        self.objects = []
        self.freeze()

    def add_mass_visuals(self, locations):

        for pt in locations:
            mass_vis = MassVisual(center=pt, parent=self.view.scene)
            self.objects.append(mass_vis)

if __name__ == '__main__' and sys.flags.interactive == 0:

    canvas = Canvas()

    canvas.add_mass_visuals([
                            [0.0, 0.0, 1.0],
                            [2.0, 0.0, 1.0],
                            [0.0, 1.0, 0.0],
                            ])
    
    #random_points = np.random.rand(20, 3) * 10
    #canvas.add_mass_visuals(random_points)
    
    canvas.app.run()
