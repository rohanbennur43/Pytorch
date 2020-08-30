# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 20:49:57 2020

@author: Rohan
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Imports
import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Download training dataset
dataset = MNIST(root='data/', download=True,train=True,transform=transforms.ToTensor())
test_dataset = MNIST(root='data/', train=False)
train_ds, val_ds = random_split(dataset, [50000, 10000])
batch_size = 100
input_size=28*28
num_classes=10

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, xb):
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out
    
model = MnistModel()
for images, labels in train_loader:
    outputs = model(images)
    break
loss_fn = F.cross_entropy
# Loss for current batch of data
loss = loss_fn(outputs, labels)
print(loss)
#optimiser
a=[]
learning_rate=0.001
opt=torch.optim.SGD(model.parameters(),lr=learning_rate)
#loss function
def loss_batch(model,loss_func,xb,yb,opt=None,metric=None):
    preds=model(xb)
    loss=loss_func(preds,yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    metric_result=None
    if metric is not None:
        metric_result=metric(preds,yb)
    return loss.item(),len(xb),metric_result   
def evaluate(model,loss_fn,valid_dl,metric=None):
    with torch.no_grad():
        results=[loss_batch(model,loss_fn,xb,yb,metric=metric)
                 for xb,yb in valid_dl]
        losses,nums,metric=zip(*results)
        total=np.sum(nums)
        avg_loss=np.sum(np.multiply(losses,nums))/total
        avg_metric=None
        if metric is not None:
            avg_metric=np.sum(np.multiply(metric,nums))/total
    return avg_loss,total,avg_metric
def accuracy(outputs,labels):
    _,preds=torch.max(outputs,dim=1)
    return torch.sum(preds==labels).item()/len(preds)
val_loss,total,val_metric=evaluate(model,loss_fn,val_loader,metric=accuracy)
print(val_loss,val_metric)
def fit(epochs,model,loss_fn,opt,train_dl,valid_dl,metric=None):
    
    for epoch in range(epochs):
        for xb,yb in train_dl:
            loss,_,_=loss_batch(model,loss_fn,xb,yb,opt)
        result=evaluate(model, loss_fn, valid_dl,metric)
        val_loss,total,val_metric=result
        a.append(val_metric)
      
        if metric is None:
            print(epoch,epochs,val_loss)
        else:
            print('Loss',epoch,epochs,val_loss,metric.__name__,val_metric)
            
fit(10,model,loss_fn,opt,train_loader,val_loader,accuracy)        