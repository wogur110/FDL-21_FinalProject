from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

# Loading and normalizing MNIST
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

trainset = datasets.MNIST('./data', train=True, download=True,
                   transform=transform)
testset = datasets.MNIST('./data', train=False,
                   transform=transform)

# Training settings
batch_size = 64
test_batch_size = 1000
no_cuda = False
seed = 1
input_size = 28*28 # 784
num_classes = 10

use_cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {}
if use_cuda:
    kwargs.update({'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True},
                 )
    
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size, **kwargs)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=test_batch_size, **kwargs)

# Define a network
net = nn.Linear(input_size, num_classes)
    
net = net.to(device)

# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the network
# training settings
epochs = 10
log_interval = 100
save_model = True

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # TODO: training code which optimizes model
        ##### YOUR CODE START #####
        
        # Reshape data and target
        data = data.reshape(data.shape[0], -1) # (batch_size, input_size)        
        
        # Training on GPU
        data = data.to(device)
        target = target.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        output = model(data) # (batch_size, num_classes)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        ##### YOUR CODE END #####
        
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({(100. * batch_idx / len(train_loader)):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # TODO: test code which evaluates model
            ##### YOUR CODE START #####
            # Reshape data and target
            data = data.reshape(data.shape[0], -1) # (batch_size, input_size)
            
            # inference on GPU
            data = data.to(device)
            target = target.to(device)
            
            output = model(data)
            test_loss += criterion(output, target) * len(data)
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()

            ##### YOUR CODE END #####


    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({(100. * correct / len(test_loader.dataset)):.0f}%)\n')

for epoch in range(1, epochs + 1):
    train(net, device, train_loader, optimizer, criterion, epoch)
    test(net, device, test_loader, criterion)