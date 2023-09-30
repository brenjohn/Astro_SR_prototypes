#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 17:38:34 2023

@author: john
"""

import torch
from torch import nn
from torch.nn import init


class FSRCNN(torch.nn.Module):
    """
    This class defines a basic FSRCNN network for upsampling a 3D tensor.
    """
    
    def __init__(self, d = 32, s = 5, m = 1):
        
        super().__init__()
        
        # Feature extraction layer.
        self.extraction = nn.Conv3d(in_channels  = 1, 
                                    out_channels = d, 
                                    kernel_size  = 5, 
                                    stride       = 1,
                                    padding      = 2,
                                    dtype=torch.float32)
        self.extraction_prelu = nn.PReLU(num_parameters=d)
        
        # Shrinking layer.
        self.shrinking = nn.Conv3d(in_channels  = d,
                                   out_channels = s,
                                   kernel_size  = 1,
                                   stride       = 1,
                                   padding      = 0,
                                   dtype=torch.float32)
        self.shrinking_prelu = nn.PReLU(num_parameters=s)
        
        # Mapping layers.
        self.mappings = [
            nn.Conv3d(in_channels  = s,
                      out_channels = s,
                      kernel_size  = 3,
                      stride       = 1,
                      padding      = 1,
                      dtype=torch.float32)
            for mi in range(m)
            ]
        self.mapping_prelus = [nn.PReLU(num_parameters=s) for mi in range(m)]
        
        # Expanding layer.
        self.expanding = nn.Conv3d(in_channels  = s,
                                   out_channels = d,
                                   kernel_size  = 1,
                                   stride       = 1,
                                   padding      = 0,
                                   dtype=torch.float32)
        self.expanding_prelu = nn.PReLU(num_parameters=d)
        
        # Deconvolution layer.
        self.deconv_layer = nn.ConvTranspose3d(in_channels    = d,
                                               out_channels   = 1,
                                               kernel_size    = 16,
                                               stride         = 4,
                                               padding        = 6,
                                               output_padding = 0,
                                               bias=True)
        
        self.init()
        
    def init(self):
        """
        Kaiming He initialization.
        """
        
        init.kaiming_uniform_(self.extraction.weight, 
                              mode='fan_in', 
                              nonlinearity='relu')
        
        init.kaiming_uniform_(self.shrinking.weight, 
                              mode='fan_in', 
                              nonlinearity='relu')
        
        init.kaiming_uniform_(self.expanding.weight, 
                              mode='fan_in', 
                              nonlinearity='relu')
        
        for mapping_layer in self.mappings:
            init.kaiming_uniform_(mapping_layer.weight, 
                                  mode='fan_in', 
                                  nonlinearity='relu')
            
        init.kaiming_uniform_(self.deconv_layer.weight, 
                              mode='fan_in', 
                              nonlinearity='relu')
        
        
    def forward(self, x):
        
        # Feature extraction.
        x = self.extraction(x)
        x = self.extraction_prelu(x)
        
        # Shrinking.
        x = self.shrinking(x)
        x = self.shrinking_prelu(x)
        
        # Mapping.
        for conv, prelu in zip(self.mappings, self.mapping_prelus):
            x = conv(x)
            x = prelu(x)
        
        # Expanding.
        x = self.expanding(x)
        x = self.expanding_prelu(x)
        
        # Upsampling.
        x = self.deconv_layer(x)
        
        return x