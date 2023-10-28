#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:28:27 2023

@author: john
"""

import os
import sys
sys.path.append("../../..")

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm

criterion = nn.MSELoss()


#%% Load the dataset
data_dir  = 'data/'
data_path = 'srcnn_train_dataset.pth'
data      = torch.load(data_dir + data_path)
Xs, Ys    = data['Xs'], data['Ys']
mu, sigma = data['mu'], data['sigma']
scale     = data['scale_factor']


#%% Load the model
model_name      = 'srcnn_5-3-3-3_64-32-32'
checkpoints_dir = 'checkpoints/' + model_name + '/'
checkpoint_num  = 699
checkpoint_name = model_name + '_{0}.pth'
checkpoint_file = checkpoints_dir + checkpoint_name.format(checkpoint_num)
checkpoint      = torch.load(checkpoint_file)
model           = checkpoint['model']


#%% Plot model output
plot_dir = 'plots/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

output_dir = model_name + '_train_output/'
if not os.path.exists(plot_dir + output_dir):
    os.makedirs(plot_dir + output_dir)
    
progress_bar = tqdm(range(len(Ys)), 
                    desc='Generating output plots', 
                    leave=False)

for i in progress_bar:
    X_lr = Xs[i]
    X_sr = model(X_lr.unsqueeze(0)).squeeze(0)
    Y = Ys[i]
    loss = criterion(X_sr, Y).item()

    fig = plt.figure(figsize=(14, 7))
    
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(np.log10(np.sum(10**Y.numpy()[0, :, :, :], axis=2)))
    ax1.set_title('Original')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(np.log10(np.sum(10**X_lr.numpy()[0, :, :, :], axis=2)))
    ax2.set_title('Down scaled')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(np.log10(np.sum(10**X_sr.detach().numpy()[0, :, :, :], axis=2)))
    ax3.set_title('Model reconstruction')
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    plt.subplots_adjust(wspace = 0.0, hspace=0.1)
    plot_name = f'output_{i}_l_{loss:.5f}.png'
    plt.savefig(plot_dir + output_dir + plot_name, 
                dpi=300)
    plt.close(fig)