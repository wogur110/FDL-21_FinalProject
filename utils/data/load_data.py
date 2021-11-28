import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path  
from torchvision import datasets

from utils.data.transforms import Transform

def create_data_loaders(data_name, data_path, batch_size, num_workers, args, train=True):    
    assert data_name == "MNIST"
    if data_name == "MNIST" :
        data_storage = datasets.MNIST(data_path, train=train, download=True, transform=Transform())

    kwargs = {}
    if torch.cuda.is_available():
        kwargs.update({'num_workers': num_workers,
                   'pin_memory': True,
                   'shuffle': True},
                 )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=batch_size,
        **kwargs
    )

    return data_loader