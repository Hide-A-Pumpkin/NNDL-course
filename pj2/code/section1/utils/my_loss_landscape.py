import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
from tqdm import tqdm as tqdm
# from IPython import display
import time

from models.my_model import MyNet, train, evaluate, evaluate_test
from torchvision import datasets, transforms
from data.loaders import get_cifar_loader


module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

transform = transforms.Compose( [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# # load dataset from cifar10
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

validset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
valid_dataloader = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=False, num_workers=2)

testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# train_loader = get_cifar_loader(train=True)
# val_loader = get_cifar_loader(train=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# choose device as cuda
# device_id = device_id
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# device = torch.device("cuda:{}".format(3) if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(3))


module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

# hyper parameter
lr = 0.01
momentum = 0.5
num_features = 3
hidden_size = 100
output_size = 10
loss_fn = nn.CrossEntropyLoss()

model = MyNet(num_features,hidden_size,output_size).to(device)
#选择一个optimizer
#SGD/ADAM
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

starttime = time.time()
num_epochs = 50
total_loss = []
acc = []

train(model, device, train_dataloader, optimizer, 0, loss_fn)
total_loss_0, acc_0 = evaluate(model, device, valid_dataloader,loss_fn, 0)
torch.save(model.state_dict(),"CIFAR10_cnn.pth")    
total_loss.append(total_loss_0)
acc.append(acc_0)

for epoch in range(1,num_epochs):
    print("Training epoch:", epoch)
    train(model, device, train_dataloader, optimizer, epoch, loss_fn)
    total_loss_0, acc_0 = evaluate(model, device, valid_dataloader,loss_fn, 0)
    if total_loss_0 < min(total_loss) and acc_0 > max(acc):
        torch.save(model.state_dict(),"CIFAR10_cnn.pth")
    total_loss.append(total_loss_0)
    acc.append(acc_0)

model_ready = MyNet(num_features,hidden_size,output_size).to(device)
model_ready.load_state_dict(torch.load('CIFAR10_cnn.pth'))
evaluate(model_ready, device, test_dataloader,loss_fn, 1)

endtime = time.time()
dtime = endtime - starttime
print("Finish! running time: %.8s s" % dtime)

x1 = range(0, num_epochs)
x2 = range(0, num_epochs)
y1 = acc
y2 = total_loss