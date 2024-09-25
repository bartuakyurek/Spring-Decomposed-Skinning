#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 07:17:13 2024

@author: bartu
"""

import numpy as np
import torch

def MSE_np(ground_truth, predictions):
     
    assert ground_truth.shape == predictions.shape, f"Provided arrays must have the same shape. Got {ground_truth.shape} and {predictions.shape}"
    diff = ground_truth - predictions
    total_diff = np.sum(diff ** 2)
    
    n_samples = ground_truth.shape[0]
    return total_diff / n_samples


def my_cost(ground_truth, predictions):
    
    
    return