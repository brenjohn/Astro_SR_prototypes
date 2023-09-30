#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 17:43:23 2023

@author: john
"""

import torch
import numpy as np

from scipy.ndimage import zoom, gaussian_filter


def create_patches(regions, patch_size):
    """
    Create a set of patches from the given set of regions
    """
    patches = []
    
    for region in regions:
        Nc, Nx, Ny, Nz = region.shape
        print('Creating patches...')
        
        for i in range(0, Nx, patch_size):
            print(f'Creating patch set {i}')
            for j in range(0, Ny, patch_size):
                for k in range(0, Nz, patch_size):
                    if i+patch_size <= Nx:
                        if j+patch_size <= Ny:
                            if k+patch_size <= Nz:
                                patch = region[:,
                                               i:i + patch_size, 
                                               j:j + patch_size,
                                               k:k + patch_size]
                                patches.append(patch)
                        
    print("Done!")
    return patches



def rescale(regions, scale_factor):
        """
        Rescale the given regions by the given scale factor.
        """
        rescaled_regions = []
        for region in regions:
            rescaled_region = zoom(region[0, :, :, :],
                                   scale_factor, 
                                   order=3)
            rescaled_regions.append(torch.tensor(rescaled_region).unsqueeze(0))
            
        return rescaled_regions
      

        
def expand_dataset(tensors, f = 6, r = 4):
    print(f'Expanding dataset by factor of {f * r}')
    expanded_dataset = []
    for tensor in tensors:
        expanded_dataset += get_tensor_transformations(tensor, f, r)
    return expanded_dataset



def get_tensor_transformations(tensor, f, r):
    transformed_tensors = []
    
    tesnor_xyz = tensor
    tesnor_zyx = tensor.transpose(2, 1, 0)
    tesnor_xzy = tensor.transpose(0, 2, 1)
    
    tesnor_xyZ = np.flip(tesnor_xyz, 2).copy()
    tesnor_zyX = np.flip(tesnor_zyx, 2).copy()
    tesnor_xzY = np.flip(tesnor_xzy, 2).copy()
    
    ts = [tesnor_xyz, 
          tesnor_zyx, 
          tesnor_xzy, 
          tesnor_xyZ, 
          tesnor_zyX, 
          tesnor_xzY]
    ts = ts[:f]
    
    for t in ts:
        transformed_tensors += get_tensor_plane_transformations(t, r)
    
    return transformed_tensors



def get_tensor_plane_transformations(t, r):
    transformed_tensors = []
    
    for i in range(r):
        t = t.transpose(1, 0, 2)
        transformed_tensors.append(t)
        
        t = np.flip(t, 0).copy()
        transformed_tensors.append(t)
    
    return transformed_tensors