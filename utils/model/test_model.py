import torch
from torch import nn
from torch.nn import functional as F

class LinearNet(nn.Module):
    def __init__(self, num_classes, data_name="MNIST"):
        super().__init__()
        assert data_name == "MNIST" or "CIFAR10"
        if data_name == "MNIST" :
            input_size = 1*28*28
        elif data_name == "CIFAR10" :
            input_size = 3*32*32
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, input):
        x = torch.flatten(input, 1) # Batch x input_size
        out = self.linear(x) # Batch x num_classes

        return out

class LeNet5(nn.Module):
    def __init__(self, num_classes, data_name="CIFAR10"):
        super().__init__()
        assert data_name == "MNIST" or "CIFAR10"
        if data_name == "CIFAR10" :
            self.conv1 = nn.Conv2d(3, 6, 5)
        elif data_name == "MNIST" :
            self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120) # H = W = 5, num_channels = 16
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input):
        x = self.pool1(F.relu(self.conv1(input))) # B x 6 x 14 x 14
        x = self.pool2(F.relu(self.conv2(x))) # B x 16 x 5 x 5
        x = torch.flatten(x, 1) # B x 400
        x = F.relu(self.fc1(x)) # B x 120
        x = F.relu(self.fc2(x)) # B x 84
        out = self.fc3(x) # B x num_classes
        
        return out