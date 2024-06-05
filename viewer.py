#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:46:50 2024

@author: bartu


TODO: this should be not viewer.py but rather a canvas for a single mesh, so that we can write a generic viewer and 
add this canvas to there. The naming is confusing right now. It is pretty specific to spring rig animation. 
"""

from itertools import count
from mayavi import mlab
from tvtk.api import tvtk
from tvtk.common import configure_input
import numpy as np
import pickle
import torch
from dataclasses import dataclass

from spring import Spring
from math_utils import perpendicular_vector

@dataclass
class Actor_Animation_Data:
    verts_numpy: np.ndarray
    verts_polydata: tvtk.PolyData
    verts_actor: tvtk.Actor
    
@dataclass
class Armature_Animation_Data:
    joints_numpy: np.ndarray
    joints_mlab_points: tvtk.PolyData
    bones_mlab_tubes: tvtk.Actor
    
@dataclass
class Spring_Armature_Animation_Data:
    parent_bone_indices: list
    springs: list
    mass_locations_numpy: np.ndarray
    mass_locations_mlab_points: tvtk.PolyData
    #springs_mlab_tubes: tvtk.Actor
    
def _create_mayavi_figure(background_color=(1,1,1), size=(800,800)):
    fig = mlab.figure(bgcolor=background_color, size=size)
    fig.scene.z_plus_view()
    return fig

class Viewer:
    def __init__(self):
        self.figure = _create_mayavi_figure()
        
        # SUGG: You can make it a list of items to be rendered, each item object (like in blender)
        # can have a specific type with specific polydata to be rendered. That ensures extensibility
        # but you don't need it for this application right now.
        self.mesh = None #es = []
        self.skeleton = None #s = []
        self.spring_rig = None

        self.spring_parent_indicators = None
        self.animation_colors = None
        self.kintree = None
    
        self.render_skeleton = False
        
    def run_animation(self, save_jpg_dir=None):
        
        @mlab.animate(delay=40, ui=False)
        def update_animation():  
            for i in count():
                frame_idx = i % len(self.mesh.verts_numpy)
                self.mesh.verts_polydata.points = self.mesh.verts_numpy[frame_idx] 
                
                # Make sure to just save jpegs up to animation period
                if save_jpg_dir and i < len(self.mesh.verts_numpy):
                    mlab.savefig(save_jpg_dir.format(frame_idx),magnification=1)
                
                if self.render_skeleton:
                    self._update_skeleton_nodes(frame_idx)
                    if self.spring_rig:
                        self._update_parent_indicators(frame_idx)
                        
                if self.animation_colors:
                    self._update_mesh_color(frame_idx)
                
                if self.figure.scene: # if not closed
                    self.figure.scene.render()
                    yield # Continue render from current frame
        
        animation_decorator = update_animation()
        mlab.show()    
    
    def set_mesh_animation(self, verts, faces=None):
        
        if torch.is_tensor(verts):
            verts = verts.detach().cpu().numpy()
        
        if faces is None:
            f = open('./body_models/smpl/male/model.pkl', 'rb')
            params = pickle.load(f)
            faces = params['f']
            
        mesh_polydata = tvtk.PolyData(points=verts[0], polys=faces)
        normals = tvtk.PolyDataNormals()
        configure_input(normals, mesh_polydata)
        
        mapper = tvtk.PolyDataMapper()
        configure_input(mapper, normals)
        actor = tvtk.Actor(mapper=mapper)
        actor.property.set(edge_color=(0.5, 0.5, 0.5), ambient=0.0,
                           specular=0.15, specular_power=128., 
                           shading=True, diffuse=0.8)

        self.figure.scene.add_actor(actor)
        self.mesh = Actor_Animation_Data(verts, mesh_polydata, actor)
    
    def set_mesh_opacity(self, opacity):
        actor = self.mesh.verts_actor
        actor.property.set(opacity=opacity)
        actor.property.backface_culling = True
        
    def set_mesh_animation_colors(self, colors):
        self.animation_colors = colors
        pass
        # TODO:set mesh colors (not a straightforward thing to do in mayavi so... i'll switch libraries...)
        
    def _update_mesh_color(self, frame_idx):
        colors = self.animation_colors[frame_idx]
        

    def _find_bone_midpoints(self, joints, kintree, selected_bone_indices):
        parent_bone_mid_coordinates = []
        for bone_idx in selected_bone_indices:
            begin = kintree[bone_idx][0]
            end = kintree[bone_idx][1]
            current_nodes = joints
            
            half_bone_vector = (current_nodes[end] - current_nodes[begin]) / 2.
            mid_coordinate = current_nodes[begin] + half_bone_vector
            parent_bone_mid_coordinates.append(mid_coordinate)
            
        return np.array(parent_bone_mid_coordinates)
    
    def _update_skeleton_nodes(self, frame_idx):
        current_nodes = self.skeleton.joints_numpy[frame_idx]
        x, y, z = current_nodes[:, 0], current_nodes[:, 1], current_nodes[:, 2]
        self.skeleton.joints_mlab_points.mlab_source.set(x=x, y=y, z=z)
    
    
    def _update_parent_indicators(self, frame_idx):
        
        parent_bone_mid_coordinates = self._find_bone_midpoints(self.skeleton.joints_numpy[frame_idx], 
                                                                self.kintree, 
                                                                self.spring_rig.parent_bone_indices)
        x = parent_bone_mid_coordinates[:,0]
        y = parent_bone_mid_coordinates[:,1]
        z = parent_bone_mid_coordinates[:,2]
        
        self.spring_parent_indicators.mlab_source.set(x=x, y=y, z=z)
        # TODO: this shouldn't be here, store mid coordinates somewhere else
        if len(self.spring_rig.springs) > 0:
            self._update_mass_spring_locations(frame_idx, parent_bone_mid_coordinates)
        
        
    def _update_mass_spring_locations(self, frame_idx, parent_bone_mid_coordinates):
        new_mass_locations = []
        for i, spring in enumerate(self.spring_rig.springs):
            spring.simulate()
            
            if frame_idx > 0:
                parent_idx = self.spring_rig.parent_bone_indices[i]
                delta_parents = self.skeleton.joints_numpy[frame_idx, parent_idx] - self.skeleton.joints_numpy[frame_idx-1, parent_idx]
                if np.sum(delta_parents ** 2) > 1e-5:
                    spring.update_connection(parent_bone_mid_coordinates[i])
             
            new_mass_locations.append(spring.mass_coord)
        
        mass_loc = np.array(new_mass_locations)
        
        
        self.spring_rig.mass_locations_mlab_points.mlab_source.set(x=mass_loc[:,0],
                                                                   y=mass_loc[:,1],
                                                                   z=mass_loc[:,2])
        

    ## TODO: rename it to add_sphere_nodes as mass-spring also uses it but they are not joints!
    ## Also add expected types like skeleton: ...
    ## Also rename skeleton to something like coordinates or xyz, it is vague!!
    def _add_joint_nodes(self, skeleton, node_scale=0.03, color=(1,0,0)):
        x, y, z = skeleton[:,0], skeleton[:,1], skeleton[:,2]
        nodes = mlab.points3d(x, y, z, scale_factor=node_scale, resolution=20)
        red_color = np.repeat(color, len(skeleton), axis=0)
        nodes.mlab_source.dataset.point_data.scalars = red_color
       
        return nodes
    
    
    # NOT WORKING -------------------------
    def _recolor_joint_nodes(self, joint_indices=None, color=(0.,1.,1.)):
        
        # If joint indices are not specified, color all joints
        if not joint_indices:
            num_joints = self.skeleton.joints_numpy.shape[0]
            colors = np.repeat(color, num_joints, axis=0)
            self.skeleton.joints_mlab_points.mlab_source.dataset.point_data.scalars = colors
            # TODO: REFACTOR THOSE NAMES BRO
        else:
            for idx in joint_indices:
                self.skeleton.joints_mlab_points.mlab_source.dataset.point_data.scalars[idx] = color            
    # END OF NOT WORKING -------------------------
    
    def _add_bone_tubes(self, nodes, node_scale, kintree, color=(0.8, 0.8, 1.)):
        # Add bones as tubes
        nodes.mlab_source.dataset.lines = np.array(kintree)
        tubes = mlab.pipeline.tube(nodes, tube_radius= node_scale * 0.2)
        mlab.pipeline.surface(tubes, color=color)
        return tubes
    
    def set_skeletal_animation(self, joints, kintree, node_scale=0.03):
        
        nodes = self._add_joint_nodes(joints[0], node_scale)
        tubes = self._add_bone_tubes( nodes, node_scale, kintree)
        
        self.skeleton = Armature_Animation_Data(joints, nodes, tubes)
        self.render_skeleton = True
                
    def set_spring_rig(self, parent_bones, kintree):
       

        self.spring_rig = Spring_Armature_Animation_Data(parent_bones, [], None, None)
        self.kintree = kintree
        
        parent_bone_mid_coordinates = self._find_bone_midpoints(self.skeleton.joints_numpy[0], 
                                                                self.kintree, 
                                                                self.spring_rig.parent_bone_indices)
        
        self.spring_parent_indicators = self._add_joint_nodes(parent_bone_mid_coordinates, 
                                                               color=(1,1,1),
                                                               node_scale=0.03)
        
        rest_locations = []
        for i, mid_coord in enumerate(parent_bone_mid_coordinates):
            joints = self.skeleton.joints_numpy[0]
            #TODO make rest vector normalized, perpendicular to parent bone
            begin_idx, end_idx = kintree[parent_bones[i]]
            parent_bone_vector = joints[end_idx] - joints[begin_idx]
            
            spring_length = 0.1
            rest_vector = perpendicular_vector(parent_bone_vector) * spring_length
            
            spring = Spring(mid_coord, rest_vector)
            self.spring_rig.springs.append(spring)
            rest_locations.append(rest_vector)
        
        self.spring_rig.mass_locations_numpy = np.array(rest_locations)
        self.spring_rig.mass_locations_mlab_points = self._add_joint_nodes(self.spring_rig.mass_locations_numpy, 
                                                                           node_scale=0.02, color=(1,1,1))

        
        