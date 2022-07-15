import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.utils.data as tud
import time
import numpy as np
import matplotlib.pyplot as plt
import os
# ## Models implementation
class MyNet(nn.Module):
    def __init__(self,num_features,hidden_size,output_size):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(num_features, 64, 5, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(1600, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        # x:1 * 28 * 28
        
        x = self.bn1(self.conv1(x))
        x = F.relu(x) # 20 * 28 * 28
        x = F.max_pool2d(x, 2, 2) # 20 * 14 * 14
        
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x) # 20 * 28 * 28
        x = F.max_pool2d(x, 2, 2) # 50 * 5 * 5
        
        x = x.view(-1, 1600) #reshape
        x1 = F.relu(self.fc1(x))
        
        x1 = self.dropout(x1)
        x1 = self.fc2(x1)
        
        return F.log_softmax(x1, dim = 1) # log probability

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
    total_loss =0.
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