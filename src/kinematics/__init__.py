#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

parent_dir = os.path.abspath(os.path.join('..'))
parent_parent_dir = os.path.abspath(os.path.join('..', '..'))

if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
if parent_parent_dir not in sys.path:
    sys.path.append(parent_parent_dir)
    