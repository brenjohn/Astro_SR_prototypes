#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 23:26:32 2023

@author: john
"""

from torch.utils.data import Dataset


class SRDataset(Dataset):
    """
    This Dataset class will take a list of density fields (represented as 3D 
    tensors) and process it into a dataset of rescaled patches and 
    corresponding high resolution labels.
    
    It can be passed to a pytorch DataLoader object for batching/shuffling.
    """
    
    def __init__(self, Xs, Ys):
        
        self.Xs = Xs
        self.Ys = Ys


    def __len__(self):
        """
        This function is required by the DataLoader interface
        """
        return len(self.Xs)


    def __getitem__(self, idx):
        """
        This function is required by the DataLoader interface
        """
        return self.Xs[idx], self.Ys[idx]