import os
import argparse
import sys
import time
import cv2 
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from tqdm import tqdm

from dataset import CheXpert
from simsiam import SimSiam
from lr_scheduler import LR_Scheduler
import config_linear_eval as config
from metrics import *

'author seung-wan.J'

def linear_eval(simsiam_model):
    # training parameters
    num_epochs = config.epoch
    warmup_epochs = config.warmup_epochs
    batch_size = config.batch_size
    base_lr = config.base_lr
    warmup_lr = config.warmup_lr
    final_lr = 0
    num_classes = 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # dataset and loader
    print('dataset...')
    train_dataset = CheXpert(mode='train', linear_eval=True)
    train_loader = DataLoader(\
        dataset=train_dataset, 
            batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_dataset = CheXpert(mode='valid', linear_eval=True)
    val_loader = DataLoader(\
        dataset=val_dataset, 
            batch_size=batch_size, shuffle=False, num_workers=4)
    
    # model setting
    print('model setting...')
    model = simsiam_model.backbone
    model = model.to(device)
    cls = nn.Linear(in_features=model.out_features, out_features=num_classes, bias=True).to(device)

    # Loss and optimizer
    ce_loss = nn.CrossEntropyLoss()
    learning_rate = base_lr*batch_size/256 # page 3 baseline setting 
    optimizer = torch.optim.SGD(cls.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
    lr_scheduler = LR_Scheduler(
        optimizer,
        warmup_epochs, warmup_lr*batch_size/256, 
        num_epochs, base_lr*batch_size/256, final_lr*batch_size/256, 
        len(train_loader),
    )

    print('Training Start...')
    global_progress = tqdm(range(0, num_epochs), desc=f'Training')
    for epoch in global_progress:
        epoch_loss = 0.0
        tr_acc = 0.0
            
        model.eval()
        cls.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            cls.zero_grad()
            with torch.no_grad():
                feature = model(images)

            outputs = cls(feature)
            loss = ce_loss(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            correct = len(np.where(outputs == labels.cpu().data.numpy())[0])
            tr_acc += correct
            
            lr = lr_scheduler.step()
            
        epoch_loss /= len(train_loader)
        tr_acc /= len(train_loader)*batch_size
            
        va_loss = 0.0
        va_y = []
        va_pred = []
        with torch.no_grad():
            model.eval()
            cls.eval()
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = cls(model(images))
                loss = ce_loss(outputs, labels)
                
                va_y.append(y_batch.cpu().data.numpy())
                va_pred.append(torch.softmax(outputs, 1).data.cpu().numpy()[:, 1])
            
            va_loss /= len(val_loader)
        
        if (epoch + 1) % print_every == 0:
            va_y = np.concatenate(va_y, 0)
            va_pred = np.concatenate(va_pred, 0)
            auc = metrics.roc_auc_score(va_y, va_pred)

            print(get_metrics(va_y, va_pred))
            print('Epoch [{}/{}], T_Loss: {:.4f}, T_Acc: {:.4f}, V_Loss: {:.4f}, V_AUC: {:.4f}'
                    .format(epoch + 1, num_epochs, epoch_loss, tr_acc, va_loss, auc))

        
            
if __name__ == '__main__':

    args = argparse.ArgumentParser()

    