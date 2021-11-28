import torch
from torch import nn
from torch.nn import functional as F

class LinearNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, input):
        B = input.shape[0] # Batch
        input = input.view(B, -1) # Batch x input_size
        out = self.linear(input) # Batch x num_classes

        return out