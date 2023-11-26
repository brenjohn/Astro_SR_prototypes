#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 13:31:03 2023

@author: john
"""

import torch
import numpy as np

from torch import nn
import torch.nn.init as init


class DSC_MS(torch.nn.Module):
    """
    This class defines the Downsampled Skip-Connection / Multi-Scale network.
    
    parameters:
        channels - the number of channels in the volume.
    """
    
    def __init__(self, dsc_model, ms_model):
        super().__init__()
        
        self.dsc_model = dsc_model
        self.ms_model = ms_model
        self.conv = nn.Conv3d(32 + 3, 1, 3, 1, 1, dtype=torch.float32)
        

    def forward(self, x):
        x_dsc = self.dsc_model(x)
        x_ms = self.ms_model(x)
        x_dsc_ms = torch.cat([x_dsc, x_ms], dim=1)
        x_dsc_ms = self.conv(x_dsc_ms)
        return x_dsc_ms
    
    
    def learning_rates(self, lr):
        learning_rates = self.dsc_model.learning_rates(lr)
        learning_rates += self.ms_model.learning_rates(lr)
        learning_rates += [{'params': self.conv.parameters(),  'lr': lr}]
        return learning_rates
    


class DSC(torch.nn.Module):
    """
    This class defines the Downsampled Skip-Connection network.
    
    parameters:
        channels - the number of channels in the volume.
    """
    
    def __init__(self):
        super().__init__()
        
        #===========================   Module 1   ===========================#
        self.down1  = nn.MaxPool3d(8)
        self.conv1a = nn.Conv3d(1, 32, 3, 1, 1, dtype=torch.float32)
        self.act1a  = nn.PReLU(num_parameters = 32)
        self.conv1b = nn.Conv3d(32, 32, 3, 1, 1, dtype=torch.float32)
        self.act1b  = nn.PReLU(num_parameters = 32)
        self.up1    = nn.Upsample(scale_factor = 2)
        
        
        #===========================   Module 2   ===========================#
        self.down2  = nn.MaxPool3d(4)
        self.conv2a = nn.Conv3d(32 + 1, 32, 3, 1, 1, dtype=torch.float32)
        self.act2a  = nn.PReLU(num_parameters = 32)
        self.conv2b = nn.Conv3d(32, 32, 3, 1, 1, dtype=torch.float32)
        self.act2b  = nn.PReLU(num_parameters = 32)
        self.up2    = nn.Upsample(scale_factor = 2)
        
        
        #===========================   Module 3   ===========================#
        self.down3  = nn.MaxPool3d(2)
        self.conv3a = nn.Conv3d(32 + 1, 32, 3, 1, 1, dtype=torch.float32)
        self.act3a  = nn.PReLU(num_parameters = 32)
        self.conv3b = nn.Conv3d(32, 32, 3, 1, 1, dtype=torch.float32)
        self.act3b  = nn.PReLU(num_parameters = 32)
        self.up3    = nn.Upsample(scale_factor = 2)
        
        
        #===========================   Module 4   ===========================#
        self.conv4a = nn.Conv3d(32 + 1, 32, 3, 1, 1, dtype=torch.float32)
        self.act4a  = nn.PReLU(num_parameters = 32)
        self.conv4b = nn.Conv3d(32, 32, 3, 1, 1, dtype=torch.float32)
        self.act4b  = nn.PReLU(num_parameters = 32)
        

    def forward(self, x_lr):
        
        # Module 1
        x = self.down1(x_lr)
        x = self.conv1a(x)
        x = self.act1a(x)
        x = self.conv1b(x)
        x = self.act1b(x)
        x = self.up1(x)
        
        # Module 2
        x = torch.cat([x, self.down2(x_lr)], dim=1)
        x = self.conv2a(x)
        x = self.act2a(x)
        x = self.conv2b(x)
        x = self.act2b(x)
        x = self.up2(x)
        
        # Module 3
        x = torch.cat([x, self.down3(x_lr)], dim=1)
        x = self.conv3a(x)
        x = self.act3a(x)
        x = self.conv3b(x)
        x = self.act3b(x)
        x = self.up3(x)
        
        # Module 4
        x = torch.cat([x, x_lr], dim=1)
        x = self.conv4a(x)
        x = self.act4a(x)
        x = self.conv4b(x)
        x = self.act4b(x)
        
        return x
    
    
    def learning_rates(self, lr):
        learning_rates = [
            {'params': self.conv1a.parameters(), 'lr': lr},
            {'params': self.act1a.parameters(),  'lr': lr},
            {'params': self.conv1b.parameters(), 'lr': lr},
            {'params': self.act1b.parameters(),  'lr': lr},
            
            {'params': self.conv2a.parameters(), 'lr': lr},
            {'params': self.act2a.parameters(),  'lr': lr},
            {'params': self.conv2b.parameters(), 'lr': lr},
            {'params': self.act2b.parameters(),  'lr': lr},
            
            {'params': self.conv3a.parameters(), 'lr': lr},
            {'params': self.act3a.parameters(),  'lr': lr},
            {'params': self.conv3b.parameters(), 'lr': lr},
            {'params': self.act3b.parameters(),  'lr': lr},
            
            {'params': self.conv4a.parameters(), 'lr': lr},
            {'params': self.act4a.parameters(),  'lr': lr},
            {'params': self.conv4b.parameters(), 'lr': lr},
            {'params': self.act4b.parameters(),  'lr': lr}]
        
        return learning_rates



class MS(torch.nn.Module):
    """
    This class defines the Multi-Scale network.
    
    parameters:
        channels - the number of channels in the volume.
    """
    
    def __init__(self):
        super().__init__()
        
        # Kernel 5 layers
        # in_channels, out_channels, kernel_size, stride, padding
        self.ms5_1_conv  = nn.Conv3d(1, 16, 5, 1, 2, dtype=torch.float32)
        self.ms5_1_prelu = nn.PReLU(num_parameters = 16)
        self.ms5_2_conv  = nn.Conv3d(16, 8, 5, 1, 2, dtype=torch.float32)
        self.ms5_2_prelu = nn.PReLU(num_parameters = 8)
        self.ms5_3_conv  = nn.Conv3d(8, 8, 5, 1, 2, dtype=torch.float32)
        self.ms5_3_prelu = nn.PReLU(num_parameters = 8)
        
        # Kernel 9 layers
        self.ms9_1_conv  = nn.Conv3d(1, 16, 9, 1, 4, dtype=torch.float32)
        self.ms9_1_prelu = nn.PReLU(num_parameters = 16)
        self.ms9_2_conv  = nn.Conv3d(16, 8, 9, 1, 4, dtype=torch.float32)
        self.ms9_2_prelu = nn.PReLU(num_parameters = 8)
        self.ms9_3_conv  = nn.Conv3d(8, 8, 9, 1, 4, dtype=torch.float32)
        self.ms9_3_prelu = nn.PReLU(num_parameters = 8)
        
        # Kernel 13 layers
        self.ms13_1_conv  = nn.Conv3d(1, 16, 13, 1, 6, dtype=torch.float32)
        self.ms13_1_prelu = nn.PReLU(num_parameters = 16)
        self.ms13_2_conv  = nn.Conv3d(16, 8, 13, 1, 6, dtype=torch.float32)
        self.ms13_2_prelu = nn.PReLU(num_parameters = 8)
        self.ms13_3_conv  = nn.Conv3d(8, 8, 13, 1, 6, dtype=torch.float32)
        self.ms13_3_prelu = nn.PReLU(num_parameters = 8)
        
        # Combination layers
        self.comb_1_conv  = nn.Conv3d(1 + 3*8, 8, 7, 1, 3, dtype=torch.float32)
        self.comb_1_prelu = nn.PReLU(num_parameters = 8)
        self.comb_2_conv  = nn.Conv3d(8, 3, 5, 1, 2, dtype=torch.float32)
        self.comb_2_prelu = nn.PReLU(num_parameters = 3)
        
    
    def forward(self, x_lr):
        
        # Kernel 5 layers
        x5 = self.ms5_1_conv(x_lr)
        x5 = self.ms5_1_prelu(x5)
        x5 = self.ms5_2_conv(x5)
        x5 = self.ms5_2_prelu(x5)
        x5 = self.ms5_3_conv(x5)
        x5 = self.ms5_3_prelu(x5)
        
        # Kernel 9 layers
        x9 = self.ms9_1_conv(x_lr)
        x9 = self.ms9_1_prelu(x9)
        x9 = self.ms9_2_conv(x9)
        x9 = self.ms9_2_prelu(x9)
        x9 = self.ms9_3_conv(x9)
        x9 = self.ms9_3_prelu(x9)
        
        # Kernel 13 layers
        x13 = self.ms13_1_conv(x_lr)
        x13 = self.ms13_1_prelu(x13)
        x13 = self.ms13_2_conv(x13)
        x13 = self.ms13_2_prelu(x13)
        x13 = self.ms13_3_conv(x13)
        x13 = self.ms13_3_prelu(x13)
        
        # Combination layers
        x = torch.cat([x_lr, x5, x9, x13], dim=1)
        x = self.comb_1_conv(x)
        x = self.comb_1_prelu(x)
        x = self.comb_2_conv(x)
        x = self.comb_2_prelu(x)
        
        return x
    
    
    def learning_rates(self, lr):
        learning_rates = [
            {'params': self.ms5_1_conv.parameters(),  'lr': lr},
            {'params': self.ms5_1_prelu.parameters(), 'lr': lr},
            {'params': self.ms5_2_conv.parameters(),  'lr': lr},
            {'params': self.ms5_2_prelu.parameters(), 'lr': lr},
            {'params': self.ms5_3_conv.parameters(),  'lr': lr},
            {'params': self.ms5_3_prelu.parameters(), 'lr': lr},
            
            {'params': self.ms9_1_conv.parameters(),  'lr': lr},
            {'params': self.ms9_1_prelu.parameters(), 'lr': lr},
            {'params': self.ms9_2_conv.parameters(),  'lr': lr},
            {'params': self.ms9_2_prelu.parameters(), 'lr': lr},
            {'params': self.ms9_3_conv.parameters(),  'lr': lr},
            {'params': self.ms9_3_prelu.parameters(), 'lr': lr},
            
            {'params': self.ms13_1_conv.parameters(),  'lr': lr},
            {'params': self.ms13_1_prelu.parameters(), 'lr': lr},
            {'params': self.ms13_2_conv.parameters(),  'lr': lr},
            {'params': self.ms13_2_prelu.parameters(), 'lr': lr},
            {'params': self.ms13_3_conv.parameters(),  'lr': lr},
            {'params': self.ms13_3_prelu.parameters(), 'lr': lr},
            
            {'params': self.comb_1_conv.parameters(),  'lr': lr},
            {'params': self.comb_1_prelu.parameters(), 'lr': lr},
            {'params': self.comb_2_conv.parameters(),  'lr': lr},
            {'params': self.comb_2_prelu.parameters(), 'lr': lr}]
        
        return learning_rates