#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:46:50 2024

@author: bartu
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
class Mesh_Animation_Data:
    verts_numpy: np.ndarray
    verts_polydata: tvtk.PolyData
    verts_actor: tvtk.Actor
   
    
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
        self.meshes = []
        self.skeletons = []
        
    
    def run_animation(self, save_jpg_dir=None):
        @mlab.animate(delay=40, ui=False)
        def update_animation():
            for mesh in self.meshes:
              
                for i in count():
                    frame = i % len(mesh.verts_numpy)
                    mesh.verts_polydata.points = mesh.verts_numpy[frame] 
                    
                    # Make sure to just save jpegs up to animation period
                    if save_jpg_dir and i < len(mesh.verts_numpy):
                        mlab.savefig(save_jpg_dir.format(frame),magnification=1)
                        #mlab.options.offscreen = True
                        #figure.scene.movie_maker.record = True
                        
                    # Set skeleton here
                    #plt.mlab_source.set(x=x, y=y, z=z)
                    # ...
                    
                    self.figure.scene.render()
                    yield
        
        animation_decorator = update_animation()
        mlab.show()    
    
    def add_mesh_animation(self, verts, faces=None):
        
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
        self.meshes.append(Mesh_Animation_Data(verts,mesh_polydata, actor))
    
    def set_mesh_opacity(self, opacity, mesh_idx=0):
        
        actor = self.meshes[mesh_idx].verts_actor
        actor.property.set(opacity=opacity)
        actor.property.backface_culling = True
    
    def add_skeletal_animation(self, joints, kintree, node_scale=0.03):
        if torch.is_tensor(joints):
            joints = joints.detach().cpu().numpy()
            
        
        skeleton = joints[0]
        color = np.zeros((skeleton.shape[0], 3))
        color[:,0] = 1.
        nodes = mlab.points3d(skeleton[:,0], 
                            skeleton[:,1], 
                            skeleton[:,2], 
                            scale_factor=node_scale, resolution=20)
        
        nodes.mlab_source.dataset.point_data.scalars = _get_red_color_by_shape((skeleton.shape[0], 3))
        
        nodes.mlab_source.dataset.lines = np.array(kintree)
        
        # Use a tube fiter to plot tubes on the link, varying the radius with the
        # scalar value
        tube = mlab.pipeline.tube(nodes, tube_radius=node_scale*0.2)
        tube.filter.radius_factor = 1.
        tube.filter.vary_radius = 'vary_radius_by_scalar'
        
        yellow=(0.8, 0.8, 0)
        mlab.pipeline.surface(tube, color=yellow)
        
        # Visualize the local atomic density
        mlab.pipeline.volume(mlab.pipeline.gaussian_splatter(nodes))
        
        
        mlab.show()
        
        
        
        