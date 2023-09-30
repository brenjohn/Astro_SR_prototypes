#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:45:51 2023

@author: john
"""

import sys
sys.path.append("..")
sys.path.append("../../..")

import torch


#%% Load the optimizer
checkpoints_dir    = 'checkpoints/'
checkpoint_name    = 'fsrcnn_32-5-1_200.pth'
checkpoint         = torch.load(checkpoints_dir + checkpoint_name)
model              = checkpoint['model']
optimizer          = checkpoint['optimizer']


#%% Update optimizer
for group in optimizer.param_groups:
    group['lr'] *= 0.1


#%% Save the updated optimizer  
checkpoint['model']     = model
checkpoint['optimizer'] = optimizer
torch.save(checkpoint,
           checkpoints_dir + checkpoint_name[:-4] + 'u' + checkpoint_name[-4:])
