import os
import argparse
import sys
import time
import cv2 
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset import CheXpert
from simsiam import SimSiam
from lr_scheduler import LR_Scheduler
import config

'author seung-wan.J'

if __name__ == '__main__':
    
    # training parameters
    num_epochs = config.epoch
    warmup_epochs = config.warmup_epochs
    batch_size = config.batch_size
    base_lr = config.base_lr
    warmup_lr = config.warmup_lr
    final_lr = config.final_lr
    num_classes = 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # dataset and loader
    print('dataset...')
    train_dataset = CheXpert(mode='train')
    train_loader = DataLoader(\
        dataset=train_dataset, 
            batch_size=batch_size, shuffle=True, num_workers=8)
    
    print('model setting...')
    # model setting 
    model = SimSiam()
    model = model.to(device)

    # Loss and optimizer
    learning_rate = base_lr*batch_size/256 # page 3 baseline setting 
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
    lr_scheduler = LR_Scheduler(
        optimizer,
        warmup_epochs, warmup_lr*batch_size/256, 
        num_epochs, base_lr*batch_size/256, final_lr*batch_size/256, 
        len(train_loader),
        constant_predictor_lr=True
    )

    print('Training Start...')
    # global_progress = tqdm(range(0, num_epochs), desc=f'Training')
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # local_progress=tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
        for i, (images1, images2) in enumerate(train_loader):
            images1 = images1.to(device)
            images2 = images2.to(device)

            loss = model(images1, images2)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        epoch_loss /= len(train_loader)
        
        print('Epoch [{}/{}], Train_Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        
    
    save_dir = 'model/'
    torch.save(model.state_dict(), save_dir + 'simsiam.pth')