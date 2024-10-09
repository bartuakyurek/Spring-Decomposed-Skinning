#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:15:07 2024

@author: bartu
"""
import torch


@staticmethod
def rodrigues_torch(r):
  """
  DISCLAIMER: This code is taken from https://github.com/CalciferZh/SMPL/blob/master/smpl_tf.py
  
  Rodrigues' rotation formula that turns axis-angle tensor into rotation
  matrix in a batch-ed manner.

  Parameter:
  ----------
  r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].

  Return:
  -------
  Rotation matrix of shape [batch_size * angle_num, 3, 3].

  """
  eps = r.clone().normal_(std=1e-8)
  theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
  theta_dim = theta.shape[0]
  r_hat = r / theta
  cos = torch.cos(theta)
  z_stick = torch.zeros(theta_dim, dtype=torch.float64).to(r.device)
  m = torch.stack(
    (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
     -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
  m = torch.reshape(m, (-1, 3, 3))
  i_cube = (torch.eye(3, dtype=torch.float64).unsqueeze(dim=0) \
           + torch.zeros((theta_dim, 3, 3), dtype=torch.float64)).to(r.device)
  A = r_hat.permute(0, 2, 1)
  dot = torch.matmul(A, r_hat)
  R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
  return R
