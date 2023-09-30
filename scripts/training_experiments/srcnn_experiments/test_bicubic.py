#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 16:37:08 2023

@author: john
"""

import sys
sys.path.append("../../..")

import os
import torch
import numpy as np
import torch.nn as nn

from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from prototypes.data_utils import rescale
from prototypes.srdataset import SRDataset


#%% Load the dataset
data_dir  = 'data/'
data_path = 'srcnn_test_dataset.pth'
data      = torch.load(data_dir + data_path)
Xs, Ys    = data['Xs'], data['Ys']
mu, sigma = data['mu'], data['sigma']
scale     = data['scale_factor']
test_data = SRDataset(Xs, Ys)


#%%
test_loader = DataLoader(test_data, batch_size=1)
criterion   = nn.MSELoss()

times, losses = [], []

for X, Y in test_loader:
    ti = time()
    X = rescale(X, scale)[0]
    tf = time()
    
    times.append(tf - ti)
    loss = criterion(X, Y)
    losses.append(loss.item())
    
    
#%%
test_stats = {
    'times'        : times,
    'losses'       : losses,
    'mu_t'         : np.mean(times),
    'sigma_t'      : np.std(times),
    'mu_l'         : np.mean(losses),
    'sigma_l'      : np.std(losses),
    'data_path'    : data_path,
    'scale_factor' : data['scale_factor']
}
torch.save(test_stats, 
           data_dir + 'tricubic_model_test_results.pth')