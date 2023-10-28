#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 15:20:18 2023

@author: john
"""

import os
import glob
import torch

import matplotlib.pyplot as plt

#%% Find relevant files containing test results.
results_dir = "test_results/"
file_pattern = "*_test_results.pth"
matching_files = glob.glob(os.path.join(results_dir, file_pattern))


#%% Load test results.
all_results = {}
groups = {}
for file_path in matching_files:
    file_name = os.path.basename(file_path).split('.')[0]
    all_results[file_name] = torch.load(file_path)
    
    group_name = file_name.split('_')[0]
    if group_name in groups:
        groups[group_name].append(file_name)
    else:
        groups[group_name] = [file_name]


#%% Extract data points.
names, times, losses, dt, dl = [], [], [], [], []
for name, results in all_results.items():
    names.append(name[:-13])
    times.append(results['mu_t'])
    dt.append(results['sigma_t'])
    dl.append(results['sigma_l'])
    losses.append(results['mu_l'])


#%% Plot data points.
plt.errorbar(times, losses, 
             xerr=dt, yerr=dl, 
             fmt='o', capsize=3, label='Error Bars')
plt.xscale('log')
plt.yscale('log')

for i, name in enumerate(names):
    plt.annotate(name, (times[i], losses[i]), 
                 textcoords = "offset points",
                 fontsize   = 4,
                 xytext     = (2,7), 
                 ha         = 'left')

plt.xlabel('Time (seconds)')
plt.ylabel('Loss')
plt.savefig('test.png', dpi=300)