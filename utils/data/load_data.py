import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path  
from torchvision import datasets

from utils.data.transforms import Transform, ImageNetTransform, TinyImageNetTransform

def create_data_loaders(data_name, data_path, batch_size, num_workers, args, train=True):    
    assert data_name == "MNIST" or data_name == "CIFAR10" or data_name == "CIFAR100" or data_name == "ImageNet" or data_name == "ImageNet32" or data_name == "TinyImageNet"
    if data_name == "MNIST" :
        data_storage = datasets.MNIST(data_path, train=train, download=True, transform=Transform())
    elif data_name == "CIFAR10" :
        data_storage = datasets.CIFAR10(data_path, train=train, download=True, transform=Transform())
    elif data_name == "CIFAR100" :
        data_storage = datasets.CIFAR100(data_path, train=train, download=True, transform=Transform())
    elif data_name == "ImageNet" :
        data_dir = data_path / data_name
        data_storage = datasets.ImageFolder(data_dir, transform=ImageNetTransform(train, resize=False))
    elif data_name == "ImageNet32" :
        data_dir = data_path / "ImageNet"
        data_storage = datasets.ImageFolder(data_dir, transform=ImageNetTransform(train, resize=True))
    elif data_name == "TinyImageNet" :
        data_dir = data_path / data_name
        data_storage = datasets.ImageFolder(data_dir, transform=TinyImageNetTransform(args, train))

    kwargs = {}
    if torch.cuda.is_available():
        kwargs.update({'num_workers': num_workers,
                   'pin_memory': True,
                   'shuffle': train},
                 )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=batch_size,
        **kwargs
    )

    return data_loader