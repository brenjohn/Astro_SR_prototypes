#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 20:36:18 2023

@author: john

This file defines classes used to create pytorch models and datasets for
training a model to upscale a density field.
"""

import torch
import numpy as np

from torch import nn
import torch.nn.init as init


class SRCNN(torch.nn.Module):
    """
    This class defines a basic SRCNN network with a residual connection for
    upsampling a 3D simulation volume.
    
    parameters:
        channels - the number of channels in the volume.
        ks       - the kernel sizes for each convolutional layer in the model.
        ns       - the number of feature maps for each layer.
    """
    
    def __init__(self, channels = 1, ks = [5, 1, 3], ns = [64, 32]):
        
        super().__init__()
        
        self.channels = channels
        self.ks = ks
        self.ns = ns
        
        self.conv_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        # Create the convolutional layers and activation functions.
        in_channels = channels
        for k, n in zip(ks, ns):
            self.conv_layers.append(
                nn.Conv3d(in_channels  = in_channels, 
                          out_channels = n, 
                          kernel_size  = k,
                          stride       = 1,
                          padding      = (k-1)//2,
                          dtype        = torch.float32)
                )
            
            self.activations.append(
                nn.PReLU(num_parameters = n))
            
            in_channels = n
            
        self.conv_layers.append(
            nn.Conv3d(in_channels  = in_channels, 
                      out_channels = channels, 
                      kernel_size  = self.ks[-1],
                      stride       = 1,
                      padding      = (self.ks[-1]-1)//2,
                      dtype        = torch.float32)
            )
        

    def forward(self, x_in):        
        x = x_in
        
        for layer, prelu in zip(self.conv_layers, self.activations):
            x = layer(x)
            x = prelu(x)
        
        x = self.conv_layers[-1](x)
        
        return x_in + x