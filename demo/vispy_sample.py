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

class EditVisual(scene.visuals.Compound):
    def __init__(self, 
                 editable=True, 
                 selectable=True, 
                 on_select_callback=None,
                 callback_argument=None, 
                 *args, **kwargs):
        scene.visuals.Compound.__init__(self, [], *args, **kwargs)
        self.unfreeze()
        self.editable = editable
        self._selectable = selectable
        self._on_select_callback = on_select_callback
        self._callback_argument = callback_argument
        
        self.drag_reference = [0, 0, 0]
        self.freeze()

    def add_subvisual(self, visual):
        scene.visuals.Compound.add_subvisual(self, visual)
        visual.interactive = True
        self.control_points.update_bounds()
        self.control_points.visible(False)

    def select(self, val, obj=None):
        if self.selectable:
            self.control_points.visible(val)
            if self._on_select_callback is not None:
                self._on_select_callback(self._callback_argument)

    def start_move(self, start):
        self.drag_reference = start[0:2] - self.control_points.get_center()

    def move(self, end):
        if self.editable:
            shift = end[0:2] - self.drag_reference
            self.set_center(shift)

    def update_from_controlpoints(self):
        None

    @property
    def selectable(self):
        return self._selectable

    @selectable.setter
    def selectable(self, val):
        self._selectable = val

    @property
    def center(self):
        return self.control_points.get_center()

    @center.setter
    # this method redirects to set_center. Override set_center in subclasses.
    def center(self, val):
        self.set_center(val)

    # override this method in subclass
    def set_center(self, val):
        self.control_points.set_center(val[0:2])

    def select_creation_controlpoint(self):
        self.control_points.select(True, self.control_points.control_points[2])


class EditRectVisual(EditVisual):
    def __init__(self, center=[0, 0], width=20, height=20, *args, **kwargs):
        EditVisual.__init__(self, *args, **kwargs)
        self.unfreeze()
        self.rect = scene.visuals.Rectangle(center=center, width=width,
                                            height=height,
                                            color=Color("#e88834"),
                                            border_color="white",
                                            radius=0, parent=self)
        self.rect.interactive = True

        self.freeze()
        self.add_subvisual(self.rect)
        self.control_points.update_bounds()
        self.control_points.visible(False)

    def set_center(self, val):
        self.control_points.set_center(val[0:2])
        self.rect.center = val[0:2]

    def update_from_controlpoints(self):
        try:
            self.rect.width = abs(self.control_points._width)
        except ValueError:
            None
        try:
            self.rect.height = abs(self.control_points._height)
        except ValueError:
            None




class MassVisual(scene.visuals.Compound):

    def __init__(self, 
                 center, 
                 radius=1.0, 
                 *args, **kwargs): 
        
        scene.visuals.Compound.__init__(self, [])
        self.unfreeze()

        if type(center) is not np.ndarray:
            center = np.array(center)
    
        self.center = center
        sphere = scene.visuals.Markers(parent=self, 
                                       spherical=True,
                                       face_color=Color("#e88834"),
                                       light_color='white',)
        
        self.freeze()


class Canvas(scene.SceneCanvas):
    """ A simple test canvas for drawing demo """

    def __init__(self, width=800, height=800, bgcolor='brown'):
       
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

    canvas.app.run()
