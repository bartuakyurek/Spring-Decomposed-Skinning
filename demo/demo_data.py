#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 13:04:21 2024

@author: bartu
"""

import numpy as np

mass_coords = np.array([
                        [0.0, 0.1, 0.0],   # 0
                        [0.0, 0.2, 0.0],   # 1
                        [0.3, 0.1, 0.0],   # 2
                        [0.2, 0.3, 0.0],   # 3
                        ]) * 100 # Scale for canvas

spring_connections = np.array([
                                [0, 1],
                                [1, 2],
                                [0, 3],
                                [2, 3],
                                ])