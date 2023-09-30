#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:47:41 2023

@author: john
"""

import sys
sys.path.append("../../..")

import torch
import numpy as np
import torch.nn as nn

from prototypes.data_utils import rescale

criterion = nn.MSELoss()


#%% Add tricubic loss to training data
print('Computing tricubuic losses for training data...')
data_dir  = 'data/'
data_path = 'fsrcnn_train_dataset.pth'
data      = torch.load(data_dir + data_path)
Xs, Ys    = data['Xs'], data['Ys']
scale     = data['scale_factor']

loss = []
for X, Y in zip(Xs, Ys):
    X = rescale([X], scale)[0]
    loss.append(criterion(X, Y).item())
    
data['tricubic_loss'] = loss
data['tricubic_loss_avg'] = np.sum(loss)/len(loss)
torch.save(data, data_dir + data_path)


#%% Add tricubic loss to validation data
print('Computing tricubuic losses for validation data...')
data_dir  = 'data/'
data_path = 'fsrcnn_validation_dataset.pth'
data      = torch.load(data_dir + data_path)
Xs, Ys    = data['Xs'], data['Ys']
scale     = data['scale_factor']

loss = []
for X, Y in zip(Xs, Ys):
    X = rescale([X], scale)[0]
    loss.append(criterion(X, Y).item())
    
data['tricubic_loss'] = loss
data['tricubic_loss_avg'] = np.sum(loss)/len(loss)
torch.save(data, data_dir + data_path)


#%% Add tricubic loss to test data
print('Computing tricubuic losses for test data...')
data_dir  = 'data/'
data_path = 'fsrcnn_test_dataset.pth'
data      = torch.load(data_dir + data_path)
Xs, Ys    = data['Xs'], data['Ys']
mu, sigma = data['mu'], data['sigma']
scale     = data['scale_factor']

loss = []
for X, Y in zip(Xs, Ys):
    X = rescale([X], scale)[0]
    Y = torch.tensor(np.reshape(Y, (1, 128, 128, 128))).float()
    loss.append(criterion(X, Y).item())
    
data['tricubic_loss'] = loss
data['tricubic_loss_avg'] = np.sum(loss)/len(loss)
torch.save(data, data_dir + data_path)