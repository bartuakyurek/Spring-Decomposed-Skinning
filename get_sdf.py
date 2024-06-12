#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:54:30 2024

@author: bartu

This file extracts SMPL poses from ground truth of DFAUST dataset,
to inspect the difference between scanned human registrations and 
SMPL model output.

DISCLAIMER: The base code is borrowed from https://github.com/haoranchen06/DBS
"""

import igl





if __name__ == "__main__":
    print(">> Testing get_sdf.py...")
    
    
    import os
    import numpy as np
    import torch
    from smpl_torch_batch import SMPLModel    
    from viewer import Viewer
    
    # TODO: use './data/female_bpts2dbs.pt' 
    # TODO: turn shuffle on for training dataset
    # TODO: create validation and test splits as well
    bpts2dbs_data = torch.load('./data/50004_dataset.pt')
    training_data = bpts2dbs_data 
    data_loader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=False)
    device = "cpu"
    smpl_model = SMPLModel(device=device, model_path='./body_models/smpl/female/model.pkl')
    
    for data in data_loader:
       beta_pose_trans_seq = data[0].squeeze().type(torch.float64)
       betas = beta_pose_trans_seq[:,:10]
       
       ## Set pose to zero, to get T-pose to be used in SDF calculation
       pose = beta_pose_trans_seq[:,10:82]
       pose = torch.zeros_like(pose)
       
       trans = beta_pose_trans_seq[:,82:] 
       
       target_verts = data[1].squeeze()
       smpl_verts, joints = smpl_model(betas, pose, trans)

       # -----------------------------------------------------------------------
       v = smpl_verts[0].detach().cpu().numpy() # Just get one mesh, not entire anim
       f = smpl_model.faces.astype('int32') 
       
       ## TODO: do not loop over animation, get [somehow] T-pose of SMPL (set theta zero?)
       ## And just get SDF Values there.
      
       n = igl.per_vertex_normals(v, f)
       sdf = igl.shape_diameter_function(v, f, 
                                         v, n, 
                                         num_samples=30)
       
       sdf_max = np.max(sdf)
       sdf_min = np.min(sdf)
       sdf_range = sdf_max - sdf_min
       
       num_partition = 5
       sdf_partition_range = sdf_range / float(num_partition)
       threshold = 0.8 * sdf_max
       
       large_sdf_idx = np.where(sdf >= threshold)
       
       large_sdf_vals = sdf[large_sdf_idx]
       large_sdf_verts = v[large_sdf_idx]
       
       np.savez("./results/large_sdf_idx.npz", large_sdf_idx)
       ## Save the partitions, for colorization.
       
       """
       Broken idk why.
       ## Create viewer canvas and add a mesh (only single mesh is allowed rn)
       single_mesh_viewer = Viewer()
       single_mesh_viewer.set_mesh_animation(v, f)
       single_mesh_viewer.set_mesh_opacity(0.6)
       single_mesh_viewer.run_animation() #, jpg_dir=jpg_path+"{}.jpg")
       """
       
       # Sanity check
       #igl.write_obj("./SMPL-T.obj", v, f)
       
       # -----------------------------------------------------------------------
       # Break from for loop since we only wanna visualize one mesh rn
       break
       
    
         
