#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:03:14 2023

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


#%% Load the model
model_name         = 'srcnn_9-1-5_64-32'
checkpoints_dir    = 'checkpoints/' + model_name + '/'
checkpoint_num     = 6
checkpoint_name    = model_name + f'_{checkpoint_num}.pth'
checkpoint         = torch.load(checkpoints_dir + checkpoint_name)
model              = checkpoint['model']


#%%
test_loader = DataLoader(test_data, batch_size=1)
criterion   = nn.MSELoss()

times, losses = [], []
model.eval()

for X, Y in test_loader:
    ti = time()
    X_norm = (X - mu) / sigma
    X_bicu = rescale(X_norm, scale)
    X_pred = model(X_bicu[0])
    X_rstd = sigma * X_pred + mu
    tf = time()
    
    times.append(tf - ti)
    loss = criterion(X_rstd, Y)
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
           data_dir + checkpoint_name.split('.')[0] + '_test_results.pth')