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
import time
import numpy as np

from vispy import scene
from vispy.color import Color
from vispy.visuals.transforms import STTransform
from vispy.visuals.filters import MarkerPickingFilter

class MassVisual(scene.visuals.Compound):

    def __init__(self, 
                 center, 
                 size=10.0, 
                 selectable=True,
                 editable=True,
                 on_select_callback=None,
                 callback_argument=None,
                 *args, **kwargs): 
        
        scene.visuals.Compound.__init__(self, [], *args, **kwargs)
        
        self.unfreeze()

        if type(center) is not np.ndarray:
            center = np.array(center)
        if center.shape == (3,):
            center = np.reshape(center, (1,-1))

        self.size = size
        self.center = center
        self.drag_reference = None  # Initiate to None if start_move() is forgotten to call

        self.DEFAULT_COLOR =  Color("orange")
        self.SELECT_COLOR = Color("yellow")
        self.color = self.DEFAULT_COLOR

        self.marker = scene.visuals.Markers(
                                            pos        = self.center,
                                            spherical  = True,
                                            size       = self.size,
                                            antialias  = 0,
                                            face_color = self.DEFAULT_COLOR,
                                            parent     = self,
                                            #*args, **kwargs
                                            )
        # We need to set interactive True to enable picking.
        self.marker.interactive = True
        self.editable = editable
        self._selectable = selectable
        self._on_select_callback = on_select_callback
        self._callback_argument = callback_argument

        self.freeze()

    def set_data(self, center=None, size=None, color=None):
        if center is not None:
            self.center = center
        if size is not None:
            self.size = size
        if color is not None:
            self.color = color
        # Note that you need to provide position data for set_data to work.
        # Otherwise setting only the colors does not work. 
        self.marker.set_data(pos=self.center, 
                             face_color=self.color, 
                             size=self.size)

    def select(self):
        print(">> INFO: Select() called...")
        if self._selectable:
            print(">> INFO: Updating selected colors...")
            self.set_data(color=self.SELECT_COLOR)
    
            if self._on_select_callback is not None:
                self._on_select_callback(self._callback_argument)
        return
    
    def deselect(self):
        print(">> INFO: Deselect() called...")
        if self._selectable:
            print(">> INFO: Updating deselect colors...")
            self.set_data(color=self.DEFAULT_COLOR)
        return
    
    def start_move(self, world_pos):
        self.drag_reference = world_pos[0:3] - self.center 
        return

    def move(self, world_pos):
        
        if self.editable:  
            shift = world_pos[0:3] - self.drag_reference
            self.set_data(center=shift)
        return

class Canvas(scene.SceneCanvas):
    """ A simple test canvas for drawing demo """
    
    def __init__(self, width=800, height=800, bgcolor='#e8ebef'):
       
        scene.SceneCanvas.__init__(self, 
                                   keys='interactive',
                                   size=(width, height), 
                                   bgcolor=bgcolor,
                                   show=True)

        self.unfreeze()
        self.view = self.central_widget.add_view()
        self.view.camera = 'arcball'
        self.view.camera.set_range(x=[-5, 5])
        
        self.selected_object = None
        self.objects = []
        self.freeze()

    def add_mass_visuals(self, locations):
        for pt in locations:
            mass_vis = MassVisual(center=pt, parent=self.view.scene)
            self.objects.append(mass_vis)
        return
    
    def on_mouse_press(self, event):
        mouse_pos = event.pos 

        self.view.interactive = False
        selected = self.visual_at(mouse_pos)     
        self.view.interactive = True

        # Deselect previously selected
        if self.selected_object is not None:
            self.selected_object.deselect()
            self.selected_object = None

        # Left click
        if event.button == 1:
            if selected is not None:
                self.selected_object = selected.parent
                self.selected_object.select()

                tr = self.scene.node_transform(self.selected_object)
                world_pos = tr.map(mouse_pos)
                self.selected_object.start_move(world_pos)
            else:
                print(">> INFO: You clicked on empty space.")
        # Right click
        if event.button == 2: 
            print(">> WARNING: Right click doesn't do anything.")


    def on_mouse_move(self, event):

        if event.button == 1: # If clicked
            if self.selected_object is not None:
                self.view.camera._viewbox.events.mouse_move.disconnect(
                    self.view.camera.viewbox_mouse_event)
                
                # update transform to selected object
                tr = self.scene.node_transform(self.selected_object)
                world_pos = tr.map(event.pos)
             
                self.selected_object.move(world_pos)
            else:
                self.view.camera._viewbox.events.mouse_move.connect(
                    self.view.camera.viewbox_mouse_event)
                
        else:
            pass
  

if __name__ == '__main__' and sys.flags.interactive == 0:

    canvas = Canvas()

    canvas.add_mass_visuals([
                            [0.0, 0.0, 1.5],
                            [2.0, 0.0, 1.5],
                            [0.0, -1.5, 0.0],
                            ])
    
    #random_points = np.random.rand(20, 3) * 10
    #canvas.add_mass_visuals(random_points)

    canvas.app.run()
