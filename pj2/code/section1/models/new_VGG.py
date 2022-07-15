import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
import os
from tqdm import tqdm as tqdm
import time

# from models.my_model import MyNet, train, evaluate, evaluate_test
from torchvision import datasets, transforms
from data.loaders import get_cifar_loader

class testNet(nn.Module):
    def __init__(self,num_features, hidden_size, output_size):
        super(testNet,self).__init__()
        
        self.layer0 = nn.Sequential(
            nn.Conv2d(num_features,64,3,padding=1),
            nn.Conv2d(64,64,3,padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1),
            nn.Conv2d(128, 128, 3,padding=1),
            nn.MaxPool2d(2, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(128,128, 3,padding=1),
            nn.Conv2d(128, 128, 3,padding=1),
            nn.Conv2d(128, 128, 1,padding=1),
            nn.MaxPool2d(2, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3,padding=1),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.Conv2d(256, 256, 1, padding=1),
            nn.MaxPool2d(2, 2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.Conv2d(512, 512, 1, padding=1),
            nn.MaxPool2d(2, 2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.layer5 = nn.Sequential(
            nn.Linear(512*4*4,1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024,hidden_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size,output_size)
        )
        
        
    def forward(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # x = self.avgpool(x)
        x = x.view(-1,512*4*4)
        
        x = self.layer5(x)
        
        return F.log_softmax(x, dim=1)

def train(model, device, train_dataloader, optimizer, epoch, loss_fn):
    model.train()

    for idx, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        preds = model(data) # batch_size * 10
        # loss = loss_fn(preds, target)
        loss = F.nll_loss(preds,target)
        # 前向传播+反向传播+优化
        # loss = F.CrossEntropyLoss(preds, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 1000 == 0:
            print("iteration: {};    Loss: {}".format(idx, loss.item()))

def evaluate(model, device, valid_dataloader,loss_fn, flag):
    model.eval()
    total_loss = 0.
    correct = 0.
    total_len = len(valid_dataloader.dataset)
    with torch.no_grad():
        for idx, (data, target) in enumerate(valid_dataloader):
            
            data, target = data.to(device), target.to(device)
            output = model(data) # batch_size * 1
            # total_loss += loss_fn(output, target).item()
            total_loss += F.nll_loss(output, target, reduction = "sum").item()
            pred = output.argmax(dim = 1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    total_loss = total_loss / total_len
    acc = correct/total_len
    if flag == 1:
        print("Accuracy on test set:{}".format(acc)) 
    else:
        print("valid loss:{}, Accuracy:{}".format(total_loss, acc)) 
    return total_loss, acc