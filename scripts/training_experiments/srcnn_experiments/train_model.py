#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:20:38 2023

@author: john
"""

import sys
sys.path.append("../../..")

import os
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from prototypes.srdataset import SRDataset


#%% Load the datasets
data_dir     = 'data/'
data_path    = data_dir + 'srcnn_train_dataset.pth'
data         = torch.load(data_path)
Xs, Ys       = data['Xs'], data['Ys']
train_data   = SRDataset(Xs, Ys)

data_path    = data_dir + 'srcnn_validation_dataset.pth'
data         = torch.load(data_path)
Xs, Ys       = data['Xs'], data['Ys']
valid_data   = SRDataset(Xs, Ys)


#%% Load the model
model_name         = 'srcnn_9-1-5_64-32'
checkpoints_dir    = 'checkpoints/' + model_name + '/'
checkpoint_num     = 0
checkpoint_name    = model_name + '_{0}.pth'
checkpoint_file    = checkpoints_dir + checkpoint_name.format(checkpoint_num)
checkpoint         = torch.load(checkpoint_file)
model              = checkpoint['model']
optimizer          = checkpoint['optimizer']
loss_train_history = checkpoint['train_history']
loss_valid_history = checkpoint['valid_history']


#%% Training parameters
num_epochs = 7
criterion = nn.MSELoss()

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True)


#%% Training loop
epochs = len(loss_train_history)
for epoch in range(1 + epochs, num_epochs + epochs):
    
    #=============================== Train Model =============================#
    model.train()
    running_loss = 0
    progress_bar = tqdm(train_loader, 
                        desc=f'Epoch {epoch}/{num_epochs}', 
                        leave=False)

    for inputs, labels in progress_bar:
        optimizer.zero_grad()              # Zero the gradients
        outputs = model(inputs)            # Forward pass
        loss = criterion(outputs, labels)  # Compute the loss
        loss.backward()                    # Backpropagation
        optimizer.step()                   # Update weights
        running_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()}, refresh=True)
        
    # Calculate average loss for this epoch
    train_loss = running_loss / len(train_loader)
    loss_train_history.append(train_loss)
    print('epoch =', epoch, 'train loss =', train_loss)
    
    
    #============================ Model Validation ===========================#
    if epoch % 5 == 0:
        model.eval()
        running_loss = 0
        progress_bar = tqdm(valid_loader, 
                            desc='Testing', 
                            leave=False)
        
        for X, Y in progress_bar:
            F = model(X)
            l = criterion(F, Y)
            running_loss += l.item()
            progress_bar.set_postfix({'loss': l.item()}, refresh=True)
            
        test_loss = running_loss / len(valid_loader)
        loss_valid_history.append((len(loss_train_history), test_loss))
        print('epoch =', epoch, 'validation loss =', test_loss)
    
    
    #=============================== Save Model ==============================#
    if epoch % 1 == 0:
        data = {
            'model'         : model,
            'optimizer'     : optimizer,
            'train_history' : loss_train_history,
            'valid_history' : loss_valid_history
        }
        file = checkpoints_dir + checkpoint_name.format(epoch)
        torch.save(data, file)