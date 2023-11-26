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
torch.set_flush_denormal(True)

from prototypes.models.dsc_ms import DSC, MS, DSC_MS


#%% Define the model
dsc_model = DSC()
ms_model = MS()
model = DSC_MS(dsc_model, ms_model)


#%% Define the optimizer
lr = 0.0001
learning_rates = model.learning_rates(lr)
optimizer = optim.Adam(learning_rates, lr=lr)


#%% Save the model and optimizer
model_name = 'dsc_ms'

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