#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 18:31:22 2023

@author: john
"""

import os
import sys
sys.path.append("../../..")

import torch
import numpy as np

from scipy.ndimage import gaussian_filter
from prototypes.data_utils import create_patches, rescale, expand_dataset


#%% Load data
print('Loading data...')
data_path = '../../../data/tensors128.npz'
tensors = np.load(data_path)
Xs_orig = [np.log10(t) for t in tensors.values()]


#%% Prepare data
print('Preparing data...')
Xs_blur = [gaussian_filter(X, sigma=0.0) for X in Xs_orig]

# Compute mean and standard deviation
mean  = np.mean(Xs_blur)
sigma = np.std(Xs_blur)

# Normalise data
Xs_prpd = [(X - mean) / sigma for X in Xs_blur]
Xs_prpd = [torch.tensor(np.reshape(X, (1, 128, 128, 128))).float() 
           for X in Xs_prpd]

# Normalise unprocessed data according to mu and sigma of processed data
Xs_norm = [(X - mean) / sigma for X in Xs_orig]
Xs_norm = [torch.tensor(np.reshape(X, (1, 128, 128, 128))).float() 
           for X in Xs_norm]


#%% Cut the dataset into smaller patches
print('Creating patches...')
patches_norm = create_patches(Xs_norm, patch_size = 64)
patches_prpd = create_patches(Xs_prpd, patch_size = 64)


#%% Rescale dataset
print('Scaling data...')
scale_factor = 4

patches_down = rescale(patches_prpd, 1/scale_factor)
patches_rstd = rescale(patches_down, scale_factor)

Xs_down = rescale(Xs_prpd, 1/scale_factor)
Xs_rstd = rescale(Xs_down, scale_factor)

Xs_blur = [torch.tensor(np.reshape(X, (1, 128, 128, 128))).float() 
           for X in Xs_blur]
Xs_ogdn = rescale(Xs_blur, 1/scale_factor) # Down-scaled original Xs


#%% Save the processed dataset
print('Saving datasets...')
data_dir = 'data/'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Training data consists of rescaled patches as input and normalised patches as 
# ground truths.
train_data = {
    'Xs'           : patches_down,
    'Ys'           : patches_norm,
    'mu'           : mean,
    'sigma'        : sigma,
    'scale_factor' : scale_factor
}
torch.save(train_data, data_dir + 'fsrcnn_train_dataset.pth')

# Validation data consists of rescaled and normalised volumes instead of
# patches
valid_data = {
    'Xs'           : Xs_down,
    'Ys'           : Xs_norm,
    'mu'           : mean,
    'sigma'        : sigma,
    'scale_factor' : scale_factor
}
torch.save(valid_data, data_dir + 'fsrcnn_validation_dataset.pth')

# Test data consists of downscaled volumes and orginal volumes so that testing
# can incorporate the bicubic upscaling required by srcnn and the rescaling
# operation to normalise input data and un-normalise model predictions.
test_data = {
    'Xs'           : Xs_ogdn,
    'Ys'           : Xs_orig,
    'mu'           : mean,
    'sigma'        : sigma,
    'scale_factor' : scale_factor
}
torch.save(test_data, data_dir + 'fsrcnn_test_dataset.pth')