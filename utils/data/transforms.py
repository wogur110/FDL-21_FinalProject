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

def ImageNetTransform(train=True, resize=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if train == True :
        if resize == True :
            transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.Resize(32), # Resized
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
        elif resize == False : 
            transform=transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
    elif train == False :
        if resize == True :
            transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.Resize(32), # Resized
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])
        elif resize == False :
            transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])
    
    return transform

def TinyImageNetTransform(args, train=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.resize_crop :
        transform_resizecrop = transforms.RandomResizedCrop(56)
    else :
        transform_resizecrop = nn.Identity()

    if args.saturation :
        transform_saturation = transforms.ColorJitter(saturation=(0.5, 2.0))
    else :
        transform_saturation = nn.Identity()

    if args.hue : 
        transform_hue = transforms.ColorJitter(hue=0.02)
    else :
        transform_hue = nn.Identity()

    if train == True :
        transform=transforms.Compose([
                transform_resizecrop,
                transforms.RandomHorizontalFlip(),
                transform_saturation,
                transform_hue,
                transforms.ToTensor(),
                normalize,
            ])
    elif train == False :
        transform=transforms.Compose([
                transform_resizecrop,
                transforms.ToTensor(),
                normalize,
            ])
    
    return transform