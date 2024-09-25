#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 12:37:49 2024

@author: bartu
"""

class Scene_Node:
    def __init__(self,  V, F):
        assert len(V.shape) == 2 and len(F.shape) == 2, f"Expected scene nodes points shape (N, 2) or (N, 3). Provided are V{V.shape} and F{F.shape}."
        
        self.vertices = V
        self.faces = F
        self.node_type = None
        
    def get_node_type(self):
        if self.node_type is None:
            return "None"
        if type(self.node_type) is not str:
            print(">> WARNING: Non-string node type encountered. Please implement a mapping.")
            return "non_string_node_type"
        else:
            return self.node_type
        

class Mesh(Scene_Node):
    def __init__(self, V, F):
        super().__init__(V, F)
        self.node_type = "Mesh"
        
        
class Armature(Scene_Node):
    def __init__(self):
        # Create a single bone armature?
        pass
    
    def __init__(self, joints, kintree):
        
        def _get_armature_mesh(joints, kintree):
            pass
        
        V, F = _get_armature_mesh(joints, kintree)
        super().__init__(V, F)
        self.node_type = "Armature"
        
    def add_joint(location, parent_joint):
        pass
        
    def update_transforms(theta):
        pass
    
class Sphere(Scene_Node):
    
    def __init__(self, radius):
        
        def _get_sphere_mesh(radius):
            pass
        
        V, F = _get_sphere_mesh(radius)
        super().__init__(V, F)
        self.node_type = "Sphere"
        
    
class Mass(Sphere):
    def __init__(self, radius, weight=1.0):
        
        super().__init__(radius)
        self.node_type = "Mass"