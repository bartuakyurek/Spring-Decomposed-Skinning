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
    
def _create_mayavi_figure(background_color=(1,1,1), size=(800,800)):
    fig = mlab.figure(bgcolor=background_color, size=size)
    fig.scene.z_plus_view()
    return fig

def _get_red_color_by_shape(shape):
    red_color = np.zeros(shape)
    red_color[:,0] = 1.
    return red_color

class Viewer:
    def __init__(self):
        self.figure = _create_mayavi_figure()
        
        # SUGG: You can make it a list of items to be rendered, each item object (like in blender)
        # can have a specific type with specific polydata to be rendered. That ensures extensibility
        # but you don't need it for this application right now.
        self.mesh = None #es = []
        self.skeleton = None #s = []
        
        
    
    def run_animation(self, render_skeleton=True, save_jpg_dir=None):
        @mlab.animate(delay=40, ui=False)
        def update_animation():
            

                for i in count():
                    frame = i % len(self.mesh.verts_numpy)
                    self.mesh.verts_polydata.points = self.mesh.verts_numpy[frame] 
                    
                    # Make sure to just save jpegs up to animation period
                    if save_jpg_dir and i < len(self.mesh.verts_numpy):
                        mlab.savefig(save_jpg_dir.format(frame),magnification=1)
                        #mlab.options.offscreen = True
                        #figure.scene.movie_maker.record = True
                    
                    # Set skeleton here
                    if render_skeleton:
                        #skeleton.joints_mlab_points = self._add_joint_nodes(skeleton.joints_numpy[frame])
                        current_nodes = self.skeleton.joints_numpy[frame]
                        x, y, z = current_nodes[:, 0], current_nodes[:, 1], current_nodes[:, 2]
                        self.skeleton.joints_mlab_points.mlab_source.set(x=x, y=y, z=z)
                        
                    #plt.mlab_source.set(x=x, y=y, z=z)
                    # ...
                    
                    self.figure.scene.render()
                    yield
        
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
    
    def _add_joint_nodes(self, skeleton, node_scale=0.03):
        x, y, z = skeleton[:,0], skeleton[:,1], skeleton[:,2]
        nodes = mlab.points3d(x, y, z, scale_factor=node_scale, resolution=20)
        nodes.mlab_source.dataset.point_data.scalars = _get_red_color_by_shape((skeleton.shape[0], 3))
        return nodes
    
    def _add_bone_tubes(self, nodes, node_scale, kintree, color=(0.8, 0.8, 0)):
        # Add bones as tubes
        nodes.mlab_source.dataset.lines = np.array(kintree)
        tubes = mlab.pipeline.tube(nodes, tube_radius= node_scale * 0.2)
        mlab.pipeline.surface(tubes, color=color)
        return tubes
    
    def set_skeletal_animation(self, joints, kintree, node_scale=0.03):
        
        nodes = self._add_joint_nodes(joints[0], node_scale)
        tubes = self._add_bone_tubes( nodes, node_scale, kintree)
        self.skeleton = Armature_Animation_Data(joints, nodes, tubes)
        
        
        
        