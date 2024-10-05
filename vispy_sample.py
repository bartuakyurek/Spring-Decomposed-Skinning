#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 09:09:13 2024
@author: bartu

"""
import numpy as np

from vispy import gloo, app, scene
from vispy.gloo import Program
from vispy.geometry import create_sphere


class Canvas(app.Canvas):
    def __init__(self):
        super().__init__(size=(600, 600), 
                         title='Mass-Spring Demo',
                         keys='interactive')
        
        self.objects = []
        
        gloo.set_viewport(0, 0, *self.physical_size)
        gloo.set_clear_color('white')

        self.timer = app.Timer('auto', self.on_timer)
        self.clock = 0
        self.timer.start()

        self.show()

    def on_draw(self, event):
        gloo.clear()

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_timer(self, event):
        self.clock += 0.001 * 1000.0 / 60.
    
        self.update()

    def on_key_press(self, event):
        if event.text == ' ':
            if self.timer.running:
                self.timer.stop()
            else:
                self.timer.start()


if __name__ == '__main__':
    canvas = Canvas()
    app.run()
