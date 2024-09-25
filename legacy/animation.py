# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:50:18 2020

@author: Haoran6
"""

# pip install mayavi (tvtk comes with it)

from itertools import count
from mayavi import mlab
from tvtk.api import tvtk
from tvtk.common import configure_input
import pickle
import torch

import cv2
import os
from os.path import isfile, join

    
def create_window(background_color=(1,1,1)):
    fig = mlab.figure(bgcolor=background_color)
    fig.scene.z_plus_view()
    return fig


def run_animation(figure, verts, verts_polydata, 
                  joints=None, joints_pd=None,
                  jpg_dir=None):
    @mlab.animate(delay=20, ui=False)
    def update_animation():
        
        for i in count():
            frame = i % len(verts)
            verts_polydata.points = verts[frame] 
            
            # Make sure to just save jpegs up to animation period
            if jpg_dir and i < len(verts): 
                mlab.savefig(jpg_dir.format(frame),magnification=1)
                #mlab.options.offscreen = True
                #figure.scene.movie_maker.record = True
                
            # Set skeleton here
            #plt.mlab_source.set(x=x, y=y, z=z)
            # ...
            
            figure.scene.render()
            yield
    
    a = update_animation()
    mlab.show()    

def add_skeletal_animation(fig):
    pass

def add_mesh_animation(fig, verts, faces=None):
    
    if torch.is_tensor(verts):
        verts = verts.detach().cpu().numpy()
    
    if faces is None:
        f = open('./body_models/smpl/male/model.pkl', 'rb')
        params = pickle.load(f)
        faces = params['f']
        
    pd = tvtk.PolyData(points=verts[0], polys=faces)
    normals = tvtk.PolyDataNormals()
    configure_input(normals, pd)
    
    mapper = tvtk.PolyDataMapper()
    configure_input(mapper, normals)
    actor = tvtk.Actor(mapper=mapper)
    actor.property.set(edge_color=(0.5, 0.5, 0.5), ambient=0.0,
                       specular=0.15, specular_power=128., shading=True, diffuse=0.8)

    fig.scene.add_actor(actor)
    return pd


def png2video(pathIn, fps=30):
    pathOut = 'video.avi'
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    
    if files[0] == '.DS_Store':
        files = files[1:-1]
        
    files.sort(key = lambda x: int(x[0:-4]))
    for i in range(len(files)):
        filename=pathIn + files[i]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()

if __name__ == '__main__':
    print(">> verts_animation.py tests are not implemented yet.")
    png2video('./rendered_jpgs/')