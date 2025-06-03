"""

This file is used to obtain dynamic animation via Controllable PBD method.
Given a tetmesh, .tgf of point handles, and associated weights as .txt 
(tip: extract weights via numpy's save as txt) this script runs Controllable PBD
and saves the computed mesh vertices (and additional info I've used for visualization)
in a .npz file.

-------------------------------------------------------------------------------
- DISCLAIMER
-------------------------------------------------------------------------------
This script is taken/adapted from 
https://github.com/yoharol/Controllable_PBD_3D/
by Wu and Umetani's paper (2023). Please refer to their
webpage https://yoharol.github.io/pages/control_pbd/
Also thank you Wu et al. for making the repository publicly available.

-------------------------------------------------------------------------------
@bartu
-------------------------------------------------------------------------------
Edited to retrieve input mesh, handles, simulated handle locations and vertices
data. I've also added custom input to handles to get different simulation results.

Major I've followed to setup (see comments with ----):
    - path_to_cpbd variable to link the source code path (downloaded from github)
    assumes the Controllable PBD source code is at the outer directory.

"""



import os
import sys
path_to_cpbd = "../Controllable_PBD_3D/" # ----
sys.path.insert(0, path_to_cpbd)

import taichi as ti
import numpy as np

from interface import render_funcs, mesh_render_3d
from data import tet_data, points_data
from cons import deform3d, framework
import compdyn.base, compdyn.inverse, compdyn.IK, compdyn.point
from utils import objs
from scipy.spatial.transform import Rotation

import math
import time

ti.init(arch=ti.x64, cpu_max_num_threads=1)

# =============================================================================
# Editable parameters ----
# =============================================================================
modelname = 'elephant' # "elephant", "elephant_helpers", "spot_helpers", "spot" or "spot_high"

# Spot
#idxs = [4, 11, 14] # Indices to translate the handles (there are 8) 
#fixed = [0, 1, 2, 3, 4, 11, 14] # Fixed handles --> make sure to include one free index because only fixed indices can have user inputs (otherwise output is static)


# Elephant
idxs = [i for i in range(15)]
fixed = [i for i in range(15)]

trans_base = np.array([0, 0, 0.], dtype=np.float32)  # relative translation 
pose_base = np.array([0.,  0., 10.]) # xyz rotation degrees

# Elephant_helpers -> move upper body, fix every bone except ears and nose
#idxs = [i for i in range(37)]   #[0, 1, 2, 3, 4, 5, 12, 20, 28,29,30,31,32,33,34,35,36] #[i for i in range(28)] #[0, 1, 2, 3, 4, 5, 12, 20, 28]
#fixed = [0,1,2,3,4,5,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46]    #[0, 1, 2, 3, 4, 5, 29,30,31,32,33,34,35,36]   # [i for i in range(28)] # [0, 1, 2, 3, 4, 5, 12, 20, 28] 

#trans_base = np.array([0, 0.0, 0], dtype=np.float32)  # relative translation 
#pose_base = np.array([0.,  0., 3.]) # xyz rotation degrees
decay = 0.0 # Dampen the user transforms over time, range [0.0, inf) 
            # will be used in pose_base / e^(decay * t)

start_frame = 0
stop_movement_frame = 10
end_frame = 120
save_npz = True

data_path = "./data/"
save_path =  data_path + f"{modelname}/{modelname}_extracted.npz" 
save_only_surface = False       
# =============================================================================
# Load data
# =============================================================================
tgf_path = os.path.join(data_path, f'{modelname}/{modelname}.tgf')
model_path = os.path.join(data_path, f'{modelname}/{modelname}.mesh')
weight_path = os.path.join(data_path, f'{modelname}/{modelname}_w.txt')

scale = 0.01
repose = (0,0.,0) # To be consistent with my blender rig I reset this #(0.0, 0.7, 0.0) # I guess it acts as global translation of the model
reverse_face=False
remove_duplicate=False # True for spot, but False for Elephant because it creates holes

points = points_data.load_points_data(tgf_path, weight_path, scale, repose)
mesh = tet_data.load_tets(model_path, scale, repose, reverse_face=reverse_face, remove_duplicate=remove_duplicate)
wireframe = [False]
points.set_color(fixed=fixed)

# ========================== init simulation ==========================
g = ti.Vector([0.0, 0.0, 0.0])
fps = 60
substeps = 5
subsub = 1
dt = 1.0 / fps / substeps

pbd = framework.pbd_framework(mesh.v_p, g, dt, damp=0.993)
deform = deform3d.Deform3D(v_p=mesh.v_p,
                           v_p_ref=mesh.v_p_ref,
                           v_invm=mesh.v_invm,
                           t_i=mesh.t_i,
                           t_m=mesh.t_m,
                           dt=dt,
                           hydro_alpha=1e-2,
                           devia_alpha=5e-2)
points_ik = compdyn.IK.PointsIK(v_p=mesh.v_p,
                                v_p_ref=mesh.v_p_ref,
                                v_weights=points.v_weights,
                                v_invm=mesh.v_invm,
                                c_p=points.c_p,
                                c_p_ref=points.c_p_ref,
                                c_p_input=points.c_p_input,
                                fix_trans=fixed)
comp = compdyn.point.CompDynPoint(v_p=mesh.v_p,
                                  v_p_ref=mesh.v_p_ref,
                                  v_p_rig=points_ik.v_p_rig,
                                  v_invm=mesh.v_invm,
                                  c_p=points.c_p,
                                  c_p_ref=points.c_p_ref,
                                  c_rot=points_ik.c_rot,
                                  v_weights=points.v_weights,
                                  dt=dt,
                                  alpha=1e-5,
                                  alpha_fixed=1e-5,
                                  fixed=fixed)
pbd.add_cons(deform, 0)
pbd.add_cons(comp, 1)

ground = objs.Quad(axis1=(10.0, 0.0, 0.0), axis2=(0.0, 0.0, -10.0), pos=0.0)
# pbd.add_collision(ground.collision)

# ========================== init interface ==========================
window = mesh_render_3d.MeshRender3D(res=(700, 700),
                                     title='spot_tet',
                                     kernel='taichi')
window.set_background_color((1, 1, 1, 1))
window.set_camera(eye=(1.0, 2, -2.5), center=(-0.3, 0.7, 0.5))
window.set_lighting((4, 4, -4), (0.96, 0.96, 0.96), (0.2, 0.2, 0.2))
# window.add_render_func(ground.get_render_draw())
window.add_render_func(
    render_funcs.get_mesh_render_func(mesh.v_p,
                                      mesh.f_i,
                                      wireframe,
                                      color=(1.0, 1.0, 1.0)))

window.add_render_func(
    render_funcs.get_points_render_func(points, point_radius=0.04))

# ========================== init status ==========================
pbd.init_rest_status(0)
pbd.init_rest_status(1)

# ========================== usd rneder ==========================
n_handles = len(points.c_p.to_numpy())
rest_pose = np.zeros((n_handles, 3))
rest_t = np.zeros((n_handles, 3))

verts, handles = [], []
handles_rigid = []
handles_pose = []
handles_t = []
if save_npz:
  
  point_color = np.zeros((points.n_points, 3), dtype=np.float32)
  point_color[:] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
  point_color[comp.fixed] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
  mesh.update_surface_verts()
  
  if save_only_surface:
      verts.append(mesh.surface_v_p.to_numpy())
      faces_np = mesh.surface_f_i.to_numpy() 
      weights_np = points.weights_np[mesh.surface_v_i.to_numpy()] # To extract weights of surface
  else:
      verts.append(mesh.v_p.to_numpy()) # surface_v_p
      faces_np = mesh.f_i.to_numpy() 
      weights_np = points.weights_np #[mesh.surface_v_i.to_numpy()] # To extract weights of surface
      
  handles.append(points.c_p.to_numpy()) # Dynamically posed handles
  handles_rigid.append(points.c_p_input.to_numpy()) # Rigidly posed handles
  
  handles_pose.append(np.array(rest_pose))
  handles_t.append(np.array(rest_t))
  def update_usd(frame: int):
    if frame < start_frame or frame > end_frame:
        return
    
    print("update verts and handles at frame", frame)
    mesh.update_surface_verts()    
    
    if save_only_surface:
        verts.append(mesh.surface_v_p.to_numpy())
    else:
        verts.append(mesh.v_p.to_numpy())
    handles.append(points.c_p.to_numpy()) #_input
    handles_rigid.append(points.c_p_input.to_numpy()) #  try also c_p_input, it might be the same
    
    handles_pose.append(np.array(rest_pose))
    handles_t.append(np.array(rest_t))

# =============================================================================
# Set movement here
# =============================================================================
# ========================== use input ========================================
written = [False]

flag = True
def set_movement():
  global flag
  t = window.get_time() - 1.0
  p_input = points_ik.c_p_ref.to_numpy() # Handles at rest
  
  if t > 0.0:
    damp = 1 / np.exp(decay * t)
    translation_vec = trans_base * math.sin(t * math.pi) * damp # Translation from rest -> posed
    
    rotation_degrees = pose_base * math.sin( t * math.pi) * damp # Rotation from rest -> posed
    rot = Rotation.from_euler('xyz', rotation_degrees , degrees=True)
    rot_mat = rot.as_matrix()
    
    for i in idxs:
        if (window.get_total_frames() < stop_movement_frame or np.sum(np.abs(rotation_degrees)) > 1) and flag:
            translation_from_rotation = (rot_mat @ p_input[i]) - p_input[i]
            
            p_input[i] += translation_vec  
            if np.sum(np.abs(rotation_degrees)) > 1e-4:
                p_input[i] += translation_from_rotation 
            
            rest_pose[i] = rotation_degrees # Save rotation for skinning
            rest_t[i] = translation_vec # Save translation for skinning
        else:    
            if flag:
                print(">>> Stopping movement...")
                flag = False #indicate the motion shall end after the first encounter of 0 rotation
            rest_pose[i] = np.zeros(3) # Save rotation for skinning
            rest_t[i] = np.zeros(3) # Save translation for skinning
    
  points.c_p_input.from_numpy(p_input)

t_total = 0.0
t_ik = 0.0
t_pbd = 0.0
while window.running():

  t = time.time()
  #if window.get_total_frames() < stop_movement_frame:
  set_movement()

  for i in range(substeps):
    t_pbd_current = time.time()
    pbd.make_prediction()
    pbd.preupdate_cons(0)
    pbd.preupdate_cons(1)
    for j in range(subsub):
      pbd.update_cons(0)
    tik = time.time()
    points_ik.ik()
    t_ik += time.time() - tik
    t_pbd += time.time() - t_pbd_current
    pbd.update_cons(1)
    pbd.update_vel()

  t_total += time.time() - t
  if window.get_total_frames() == end_frame:
    print(f'average time: {t_total / end_frame * 1000} ms')
    print(f'average ik time: {t_ik / end_frame * 1000} ms')
    print(f'average pbd+ik time: {t_pbd / end_frame * 1000} ms')

  if save_npz:
    update_usd(window.get_total_frames())

  window.pre_update()
  window.render()
  window.show()

# =============================================================================
# Save after window loop
# =============================================================================
if save_npz:
  verts_np = np.array(verts)
  handles_np = np.array(handles)
  faces_np = np.reshape(faces_np, (-1,3)) 
  handles_rigid_np = np.array(handles_rigid)
  
  handles_pose_np = np.array(handles_pose)
  handles_t_np = np.array(handles_t)
  
  print(">> WARNING: Assuming every face is triangular.")
  
  data  = {"verts_yoharol":verts_np, 
           "faces": faces_np, 
           "weights" : weights_np,
           "handles_yoharol":handles_np,
           "fixed_yoharol" : fixed,
           "user_input" : idxs,
           "handles_rigid": handles_rigid_np,
           "handles_t" : handles_t_np,
           "handles_pose" : handles_pose_np}
  
  np.savez(save_path, **data)
  print("Saved data at ", save_path)
  
window.terminate()
