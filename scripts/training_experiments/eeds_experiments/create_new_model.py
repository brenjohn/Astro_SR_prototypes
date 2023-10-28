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

from prototypes.models.eeds import EEDS, EED, EES


#%% Define the model
scale = 4

fe_n, fe_k  = 64, 3
us_n, us_k  = 4,  16
re_n, re_k  = 16, 3
ms_n, ms_ks = 4, [1, 3, 5, 7]

deep_name = 'D'
deep_name += '-'.join(map(str, [fe_k, us_k, re_k])) + '_'
deep_name += '-'.join(map(str, [fe_n, us_n, re_n, ms_n]))
deep_model = EED(fe_n, fe_k, us_n, us_k, re_n, re_k, ms_n, ms_ks, scale)

fe_n, fe_k = 4, 3
us_n, us_k = 8, 16
re_k       =    5

shallow_name = 'S'
shallow_name += '-'.join(map(str, [fe_k, us_k, re_k])) + '_'
shallow_name += '-'.join(map(str, [fe_n, us_n]))
shallow_model = EES(fe_n, fe_k, us_n, us_k, re_k, scale)

model = EEDS(deep_model, shallow_model)


#%% Define the optimizer
lr = 0.001

learning_rates = model.learning_rates(lr)

optimizer = optim.Adam(learning_rates, lr=lr)


#%% Save the model and optimizer
model_name = 'eeds_' + deep_name + '_' + shallow_name

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