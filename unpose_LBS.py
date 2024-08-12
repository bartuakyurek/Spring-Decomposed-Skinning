#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 12:48:38 2024

@author: bartu
"""
import torch

def unpose_verts_batch(target_verts, trans, T):
    T_inv = torch.inverse(T)
    num_batch = target_verts.shape[0]
    
    tmp_verts = target_verts 
    tmp_verts -= torch.reshape(trans, (num_batch, 1, 3)) 
    
    v_homo = torch.cat([tmp_verts,torch.ones([num_batch, 6890, 1])], dim=2)
    v_unposed = torch.matmul(T_inv, torch.unsqueeze(v_homo, dim=-1)).squeeze()[:, :,:3]
    
    return v_unposed