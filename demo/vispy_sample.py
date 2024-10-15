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
from vispy.visuals.filters import MarkerPickingFilter

class MassVisual(scene.visuals.Compound):

    def __init__(self, 
                 center, 
                 color,
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
        if center.shape == (3,):
            center = np.reshape(center, (1,-1))

        self.center = center
        self.markers = scene.visuals.Markers(
                                            pos = self.center,
                                            spherical=True,
                                            size=size,
                                            antialias=0,
                                            face_color=Color(color),
                                            parent=self
                                            #*args, **kwargs
                                            )
        
        scene.visuals.Compound.add_subvisual(self, self.markers)
        self.picking_filter = MarkerPickingFilter()
        self.markers.attach(self.picking_filter)

        self.editable = editable
        self._selectable = selectable
        self._on_select_callback = on_select_callback
        self._callback_argument = callback_argument

        self.freeze()

    def select(self):
        print(">> INFO: Selected called...")
        if self.selectable:
            self.sphere.set_data(edge_color="white")
            if self._on_select_callback is not None:
                self._on_select_callback(self._callback_argument)



class Canvas(scene.SceneCanvas):
    """ A simple test canvas for drawing demo """
    
    def __init__(self, width=800, height=800, bgcolor='#dcd5ef'):
       
        scene.SceneCanvas.__init__(self, keys='interactive',
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

    def add_mass_visuals(self, locations, color="#e88834"):
        for pt in locations:
            mass_vis = MassVisual(center=pt, color=color, parent=self.view.scene)
            self.objects.append(mass_vis)
        return
    
    def on_mouse_press(self, event):
 
        tr = self.scene.node_transform(self.view.scene)
        pos = tr.map(event.pos)

        self.view.interactive = False
        selected = self.visual_at(event.pos)
        self.view.interactive = True

        # Deselect previously selected
        if self.selected_object is not None:
            self.selected_object.select(False)
            self.selected_object = None
        
        # Left click
        if event.button == 1:
            if selected is not None:
                print(">> WARNING: Implement what happens after selection")
                
                # update transform to selected object
                self.selected_object = selected.parent
                tr = self.scene.node_transform(self.selected_object)
                pos = tr.map(event.pos)

                self.selected_object.select()
            else:
                print(">> INFO: You clicked on empty space.")
        # Right click
        if event.button == 2: 
            print(">> WARNING: Right click doesn't do anything.")


    def on_mouse_move(self, event):

        if event.button == 1:
            if self.selected_object is not None:
                self.view.camera._viewbox.events.mouse_move.disconnect(
                    self.view.camera.viewbox_mouse_event)
                
                # update transform to selected object
                print(">> Implement on_mouse_move...")
                tr = self.scene.node_transform(self.selected_object)
                pos = tr.map(event.pos)
                #self.selected_object.move(pos[0:2])
            else:
                self.view.camera._viewbox.events.mouse_move.connect(
                    self.view.camera.viewbox_mouse_event)
        else:
            pass

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
