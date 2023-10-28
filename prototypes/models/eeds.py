#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 14:48:08 2023

@author: john

EEDS - End-to-End-Deep-Shallow network
"""

import torch
import numpy as np

from torch import nn
import torch.nn.init as init


class EEDS(torch.nn.Module):
    """
    This class defines a basic EEDS network.
    
    parameters:
        channels - the number of channels in the volume.
    """
    
    def __init__(self, deep_model, shallow_model):
        super().__init__()
        
        self.deep_model = deep_model
        self.shallow_model = shallow_model
        

    def forward(self, x):
        y_deep = self.deep_model(x)
        y_shallow = self.shallow_model(x)
        return y_deep + y_shallow
    
    
    def learning_rates(self, lr):
        learning_rates = self.deep_model.learning_rates(lr)
        learning_rates += self.shallow_model.learning_rates(lr)
        return learning_rates



class EED(torch.nn.Module):
    """
    This class defines a basic EED network.
    """
    
    def __init__(self, 
                 fe_channels = 64,
                 fe_kernel   = 3,
                 us_channels = 4,
                 us_kernel   = 16,
                 re_channels = 64,
                 re_kernel   = 3,
                 ms_channels = 16,
                 ms_kernels  = [1, 3, 5, 7],
                 scale       = 4):
        
        super().__init__()
        
        # Feature extraction.
        self.fe_channel_expansion = nn.Conv3d(
            in_channels  = 1, 
            out_channels = fe_channels, 
            kernel_size  = fe_kernel,
            stride       = 1,
            padding      = (fe_kernel-1)//2,
            dtype        = torch.float32)
        self.fe_prelu = nn.PReLU(num_parameters=fe_channels)
        self.fe_residual_block = ResBlock(fe_channels, fe_kernel)
        
        # Upsampling.
        self.us_channel_shrinking = nn.Conv3d(
            in_channels  = fe_channels, 
            out_channels = us_channels, 
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            dtype        = torch.float32)
        self.us_channel_shrinking_prelu = nn.PReLU(num_parameters=us_channels)
        self.upsampling = nn.ConvTranspose3d(
            in_channels    = us_channels,
            out_channels   = us_channels,
            kernel_size    = us_kernel,
            stride         = scale,
            padding        = (us_kernel - scale) // 2,
            output_padding = 0,
            bias           = True)
        self.upsampling_prelu = nn.PReLU(num_parameters = us_channels)
        self.us_channel_expansion = nn.Conv3d(
            in_channels  = us_channels,
            out_channels = re_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            dtype        = torch.float32)
        self.us_channel_expansion_prelu = nn.PReLU(num_parameters=re_channels)
        
        # Reconstruction.
        self.re_residual_block1 = ResBlock(re_channels, re_kernel)
        self.re_residual_block2 = ResBlock(re_channels, re_kernel)
        self.re_channel_shrinking = nn.Conv3d(
            in_channels  = re_channels, 
            out_channels = ms_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            dtype        = torch.float32)
        self.re_channel_shrinking_prelu = nn.PReLU(num_parameters=ms_channels)
        self.multi_scale_conv = MultiScaleConv(ms_channels, ms_kernels)
        self.re_output = nn.Conv3d(
            in_channels  = ms_channels * len(ms_kernels), 
            out_channels = 1,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            dtype        = torch.float32)


    def forward(self, x):
        # Feature expansion.
        x = self.fe_channel_expansion(x)
        x = self.fe_prelu(x)
        x = self.fe_residual_block(x)
        
        # Upsampling.
        x = self.us_channel_shrinking(x)
        x = self.us_channel_shrinking_prelu(x)
        x = self.upsampling(x)
        x = self.upsampling_prelu(x)
        x = self.us_channel_expansion(x)
        x = self.us_channel_expansion_prelu(x)
        
        # Reconstruction.
        x = self.re_residual_block1(x)
        x = self.re_residual_block2(x)
        x = self.re_channel_shrinking(x)
        x = self.re_channel_shrinking_prelu(x)
        x = self.multi_scale_conv(x)
        x = self.re_output(x)
        
        return x
    
    
    def learning_rates(self, lr):
        learning_rates = [
            {'params': self.fe_channel_expansion.parameters(),       'lr': lr},
            {'params': self.fe_prelu.parameters(),                   'lr': lr},
            {'params': self.us_channel_shrinking.parameters(),       'lr': lr},
            {'params': self.us_channel_shrinking_prelu.parameters(), 'lr': lr},
            {'params': self.upsampling.parameters(),                 'lr': lr},
            {'params': self.upsampling_prelu.parameters(),           'lr': lr},
            {'params': self.us_channel_expansion.parameters(),       'lr': lr},
            {'params': self.us_channel_expansion_prelu.parameters(), 'lr': lr},
            {'params': self.re_channel_shrinking.parameters(),       'lr': lr},
            {'params': self.re_channel_shrinking_prelu.parameters(), 'lr': lr},
            {'params': self.re_output.parameters(),                  'lr': lr}]
        
        learning_rates += self.fe_residual_block.learning_rates(lr)
        learning_rates += self.re_residual_block1.learning_rates(lr)
        learning_rates += self.re_residual_block2.learning_rates(lr)
        learning_rates += self.multi_scale_conv.learning_rates(lr)
        
        return learning_rates



class EES(torch.nn.Module):
    """
    This class defines a basic EES network.
    """
    
    def __init__(self,
                 fe_channels = 4,
                 fe_kernel   = 3,
                 us_channels = 8,
                 us_kernel   = 16,
                 re_kernel   = 5,
                 scale       = 4):
        
        super().__init__()
        
        # Feature extraction layer.
        self.feature_extraction = nn.Conv3d(
            in_channels  = 1, 
            out_channels = fe_channels, 
            kernel_size  = fe_kernel,
            stride       = 1,
            padding      = (fe_kernel - 1) // 2,
            dtype        = torch.float32)
        self.feature_prelu = nn.PReLU(num_parameters = fe_channels)
        
        # Upsampling layer.
        self.upsampling = nn.ConvTranspose3d(
            in_channels    = fe_channels,
            out_channels   = us_channels,
            kernel_size    = us_kernel,
            stride         = scale,
            padding        = (us_kernel - scale) // 2,
            output_padding = 0,
            bias           = True)
        self.upsampling_prelu = nn.PReLU(num_parameters = us_channels)
        
        # Reconstruction layer.
        self.reconstruction = nn.Conv3d(
            in_channels  = us_channels, 
            out_channels = 1, 
            kernel_size  = re_kernel,
            stride       = 1,
            padding      = (re_kernel - 1) // 2,
            dtype        = torch.float32)
        

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.feature_prelu(x)
        x = self.upsampling(x)
        x = self.upsampling_prelu(x)
        x = self.reconstruction(x)
        return x
    
    
    def learning_rates(self, lr):
        learning_rates = [
            {'params': self.feature_extraction.parameters(), 'lr': lr},
            {'params': self.feature_prelu.parameters(),      'lr': lr},
            {'params': self.upsampling.parameters(),         'lr': lr},
            {'params': self.upsampling_prelu.parameters(),   'lr': lr},
            {'params': self.reconstruction.parameters(),     'lr': lr}]
        
        return learning_rates



class ResBlock(torch.nn.Module):
    
    def __init__(self, channels, kernel_size):
        super().__init__()
        
        self.conv_layer1 = nn.Conv3d(in_channels  = channels, 
                                     out_channels = channels, 
                                     kernel_size  = kernel_size,
                                     stride       = 1,
                                     padding      = (kernel_size-1)//2,
                                     dtype        = torch.float32)
        self.prelu1 = nn.PReLU(num_parameters = channels)
        
        self.conv_layer2 = nn.Conv3d(in_channels  = channels, 
                                     out_channels = channels, 
                                     kernel_size  = kernel_size,
                                     stride       = 1,
                                     padding      = (kernel_size-1)//2,
                                     dtype        = torch.float32)
        self.prelu2 = nn.PReLU(num_parameters = channels)
    
    
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.prelu1(x)
        x = self.conv_layer2(x)
        x = self.prelu2(x)
        return x
    
    
    def learning_rates(self, lr):
        learning_rates = [{'params': self.conv_layer1.parameters(), 'lr': lr},
                          {'params': self.conv_layer2.parameters(), 'lr': lr}]
        
        learning_rates += [{'params': self.prelu1.parameters(), 'lr': lr},
                           {'params': self.prelu2.parameters(), 'lr': lr}]
        
        return learning_rates



class MultiScaleConv(torch.nn.Module):
    
    def __init__(self, channels, kernel_sizes = [1, 3, 5, 7]):
        super().__init__()
        
        self.conv_layers = nn.ModuleList([
            nn.Conv3d(in_channels  = channels,
                      out_channels = channels,
                      kernel_size  = k,
                      stride       = 1,
                      padding      = (k-1)//2,
                      dtype        = torch.float32)
            for k in kernel_sizes])
        
        self.prelus = nn.ModuleList([
            nn.PReLU(num_parameters = channels)
            for k in kernel_sizes])
    
    
    def forward(self, x):
        outputs = [p(c(x)) for c, p in zip(self.conv_layers, self.prelus)]
        return torch.cat(outputs, dim=1)
    
    
    def learning_rates(self, lr):
        learning_rates = [{'params': layer.parameters(), 'lr': lr}
                          for layer in self.conv_layers]
        
        learning_rates += [{'params': act.parameters(), 'lr': lr}
                           for act in self.prelus]
        
        return learning_rates