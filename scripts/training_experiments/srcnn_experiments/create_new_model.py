#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:01:38 2023

@author: john
"""

import os
import sys
sys.path.append("..")
sys.path.append("../../..")

import torch
import torch.optim as optim

from prototypes.models.srcnn import SRCNN


#%% Define the model
ks = [9, 1, 5]
ns = [64, 32]
model = SRCNN(1, ks, ns)


#%% Define the optimizer
lr = 0.001
lr_weights = [1, 1, 0.1]

learning_rates = [
    {'params': layer.parameters(), 'lr': w * lr}
    for (layer, w) in zip(model.conv_layers, lr_weights)
]

learning_rates += [
    {'params': act.parameters(), 'lr': w * lr}
    for (act, w) in zip(model.activations, lr_weights)
]

optimizer = optim.Adam(learning_rates, lr=lr)


#%% Save the model and optimizer
model_name = 'srcnn_' + '-'.join(map(str, ks)) + '_' + '-'.join(map(str, ns))

checkpoints_dir = 'checkpoints/'
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
    
model_dir = checkpoints_dir + model_name + '/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    
data = {
    'model'         : model,
    'optimizer'     : optimizer,
    'train_history' : [],
    'valid_history' : []
}
torch.save(data, model_dir + model_name + '_0.pth')