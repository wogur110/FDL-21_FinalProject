import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

def Transform():
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
    
    return transform