#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 19:03:02 2023

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
from prototypes.srdataset import SRDataset


#%% Load the dataset
data_dir  = 'data/'
data_path = 'fsrcnn_test_dataset.pth'
data      = torch.load(data_dir + data_path)
Xs, Ys    = data['Xs'], data['Ys']
mu, sigma = data['mu'], data['sigma']
scale     = data['scale_factor']
test_data = SRDataset(Xs, Ys)


#%% Load the model
model_name         = 'fsrcnn_32-5-1'
checkpoints_dir    = 'checkpoints/' + model_name + '/'
checkpoint_num     = 4
checkpoint_name    = model_name + f'_{checkpoint_num}.pth'
checkpoint         = torch.load(checkpoints_dir + checkpoint_name)
model              = checkpoint['model']


#%%
test_loader = DataLoader(test_data, batch_size=1)
criterion   = nn.MSELoss()

times, losses = [], []
model.eval()

for X, Y in test_loader:
    Y = Y.unsqueeze(0)
    ti = time()
    X_norm = (X - mu) / sigma
    X_pred = model(X_norm)
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