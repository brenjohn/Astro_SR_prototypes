#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 18:02:17 2023

@author: john
"""

import os
import sys
sys.path.append("../../..")

import torch
import numpy as np
import matplotlib.pyplot as plt


#%% Load the dataset
data_dir   = 'data/'
data_path  = 'dsc_ms_train_dataset.pth'
data_train = torch.load(data_dir + data_path)
Xs, Ys     = data_train['Xs'], data_train['Ys']

data_path  = 'dsc_ms_validation_dataset.pth'
data_valid = torch.load(data_dir + data_path)
Xs, Ys     = data_valid['Xs'], data_valid['Ys']


#%% Load the model loss history
model_name         = 'dsc_ms'
checkpoints_dir    = 'checkpoints/' + model_name + '/'
checkpoint_num     = 0
checkpoint_name    = model_name + f'_{checkpoint_num}.pth'
checkpoint         = torch.load(checkpoints_dir + checkpoint_name)
loss_train_history = checkpoint['train_history']
loss_valid_history = checkpoint['valid_history']


#%% Plot learning curves
plot_dir = 'plots/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

valid_epochs = [epoch for (epoch, loss) in loss_valid_history]
valid_losses = [loss  for (epoch, loss) in loss_valid_history]

tricubic_train_loss = None
if 'tricubic_loss_avg' in data_train:
    tricubic_train_loss = data_train['tricubic_loss_avg']
    
tricubic_valid_loss = None
if 'tricubic_loss_avg' in data_valid:
    tricubic_valid_loss = data_valid['tricubic_loss_avg']

plt.figure()
plt.plot(loss_train_history,         label = 'train')
plt.plot(valid_epochs, valid_losses, label = 'validation')

if tricubic_train_loss:
    plt.hlines(tricubic_train_loss, 0, len(loss_train_history),
               'black', label='train tricubic')
if tricubic_valid_loss:
    plt.hlines(tricubic_valid_loss, 0, len(loss_train_history),
               'black', linestyle='--', label='validation tricubic')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig(plot_dir + checkpoint_name[:-4] + '_learning_curve.png', dpi=300)