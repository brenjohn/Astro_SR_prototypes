#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 18:17:23 2023

@author: john
"""

import os
import sys
sys.path.append("..")
sys.path.append("../../..")

import torch
import torch.optim as optim

from prototypes.models.fsrcnn import FSRCNN


#%% Define the model
d, s, m = 32, 5, 1
model = FSRCNN(d, s, m)


#%% Define the optimizer
lr = 0.01

learning_rates = [
    {'params': model.extraction.parameters(),       'lr': lr},
    {'params': model.extraction_prelu.parameters(), 'lr': lr},
    {'params': model.shrinking.parameters(),        'lr': lr},
    {'params': model.shrinking_prelu.parameters(),  'lr': lr},
    {'params': model.expanding.parameters(),        'lr': lr},
    {'params': model.expanding_prelu.parameters(),  'lr': lr},
    {'params': model.deconv_layer.parameters(),     'lr': 0.1 * lr}
]

learning_rates += [{'params': mapping_layer.parameters(), 'lr': lr}
                   for mapping_layer in model.mappings]

learning_rates += [{'params': prelu.parameters(), 'lr': lr}
                   for prelu in model.mapping_prelus]

optimizer = optim.Adam(learning_rates, lr = lr)


#%% Save the model and optimizer
model_name = f'fsrcnn_{d}-{s}-{m}'

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