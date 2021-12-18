import torch
from torch import nn
from torch.nn import functional as F
from math import sqrt

class Conv2d_BEE(nn.Module):

    def __init__(self, in_chans, out_chans, interpolation):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.alpha = interpolation
        self.Conv_layer1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1,2), stride=1, padding=(1,1), bias = False)
        self.Conv_layer2 = nn.Conv2d(in_chans, out_chans, kernel_size=(1,3), stride=1, padding=(0,1))
        self.Conv_layer3 = nn.Conv2d(in_chans, out_chans, kernel_size=(1,2), stride=1, padding=(1,1), bias = False)

    def forward(self, x):
        out1 = self.Conv_layer1(x) * sqrt(2) / sqrt(7)
        out2 = self.Conv_layer2(x) * sqrt(3) / sqrt(7)
        out3 = self.Conv_layer3(x) * sqrt(2) / sqrt(7)

        out = out2 + (out1[:,:,:-2,:-1] * (1 - self.alpha) + out1[:,:,:-2,1:] * self.alpha)/2 + (out3[:,:,2:,:-1] * (1 - self.alpha) + out3[:,:,2:,1:] * self.alpha)/2

        return out

class LinearNet(nn.Module):
    def __init__(self, num_classes, data_name="MNIST"):
        super().__init__()
        assert data_name == "MNIST" or data_name == "CIFAR10" or data_name == "CIFAR100" or data_name == "ImageNet" or data_name == "ImageNet32" or data_name == "TinyImageNet"
        if data_name == "MNIST" :
            input_size = 1*28*28
        elif data_name == "CIFAR10" or data_name == "CIFAR100":
            input_size = 3*32*32
        elif data_name == "ImageNet" :
            input_size = 3*224*224
        elif data_name == "ImageNet32" :
            input_size = 3*32*32
        elif data_name == "TinyImageNet" :
            input_size = 3*64*64
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, input):
        x = torch.flatten(input, 1) # Batch x input_size
        out = self.linear(x) # Batch x num_classes

        return out

class LeNet5(nn.Module):
    def __init__(self, num_classes, data_name="CIFAR10"):
        super().__init__()
        assert data_name == "MNIST" or data_name == "CIFAR10" or data_name == "CIFAR100"
        if data_name == "CIFAR10" or data_name == "CIFAR100":
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

class VGGNet(nn.Module):
    '''
    VGG-like conv-net for TInyImageNet, the input to the network is a 56x56 RGB crop.
    '''
    def __init__(self, model, in_channels=3, num_classes=1000, init_weights=True, data_name="TinyImageNet"):
        super().__init__()
        self.in_channels = in_channels
        self.model = model

        # create conv_layers corresponding to VGG type
        # VGG type dict
        # int : output channels after conv layer
        # 'M' : max pooling layer
        VGG_types = {
            'VGG11' : [64, 'M', 128, 'M', 256, 256, 'M', 512,512, 'M',512,512,'M'],
            'VGG13' : [64,64, 'M', 128, 128, 'M', 256, 256, 'M', 512,512, 'M', 512,512,'M'],
            'VGG16' : [64,64, 'M', 128, 128, 'M', 256, 256,256, 'M', 512,512,512, 'M',512,512,512,'M'],
            'VGG19' : [64,64, 'M', 128, 128, 'M', 256, 256,256,256, 'M', 512,512,512,512, 'M',512,512,512,512,'M'],
            'VGG11_fc' : [64, 'M', 128, 'M', 256, 256, 'M', 512,512, 'M'],
            'VGG13_fc' : [64,64, 'M', 128, 128, 'M', 256, 256, 'M', 512,512, 'M'],
            'VGG16_fc' : [64,64, 'M', 128, 128, 'M', 256, 256,256, 'M', 512,512,512, 'M'],
            'VGG19_fc' : [64,64, 'M', 128, 128, 'M', 256, 256,256,256, 'M', 512,512,512,512, 'M'],
            'VGG11_large' : [64, 'M', 128, 'M', 256, 256, 'M', 512,512],
            'VGG13_large' : [64,64, 'M', 128, 128, 'M', 256, 256, 'M', 512,512],
            'VGG16_large' : [64,64, 'M', 128, 128, 'M', 256, 256,256, 'M', 512,512,512],
            'VGG19_large' : [64,64, 'M', 128, 128, 'M', 256, 256,256,256, 'M', 512,512,512,512],
        }

        for key in VGG_types :
            if key in model :
                self.conv_layers = self.create_conv_laters(VGG_types[key])        

        if "reduced" in model :
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
            # simple version of fc layers
            self.fcs = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, num_classes),
            )

        elif "large" in model :
            self.fcs = nn.Sequential(
                nn.Linear(512 * 8 * 8, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, 2048),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(2048, num_classes),
            )
        
        elif "fc" in model :
            self.fcs = nn.Sequential(
                nn.Linear(512 * 4 * 4, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, 2048),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(2048, num_classes),
            )

        else :
            self.fcs = nn.Sequential(
                nn.Linear(512 * 2 * 2, 1024),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(1024, num_classes),
            )

        # weight initialization
        if init_weights:
            self._initialize_weights()


    def forward(self, x):
        x = self.conv_layers(x)
        if "reduced" in self.model :
            x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fcs(x)
        return x

    # defint weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # define a function to create conv layer taken the key of VGG_type dict 
    def create_conv_laters(self, architecture):
        layers = []
        in_channels = self.in_channels # 3

        for x in architecture:
            if type(x) == int: # int means conv layer
                out_channels = x

                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=(3,3), stride=(1,1), padding=(1,1)),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), ceil_mode=True)]
        
        return nn.Sequential(*layers)

class BEENet(nn.Module):
    '''
    Modified version of VGG-like conv-net for TInyImageNet, the input to the network is a 56x56 RGB crop.
    '''
    def __init__(self, model, in_channels=3, num_classes=1000, init_weights=True, data_name="TinyImageNet", interpolation=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.model = model
        self.interpolation = interpolation

        # create conv_layers corresponding to VGG type
        # VGG type dict
        # int : output channels after conv layer
        # 'M' : max pooling layer
        VGG_types = {
            'BEE11' : [64, 'M', 128, 'M', 256, 256, 'M', 512,512, 'M',512,512,'M'],
            'BEE13' : [64,64, 'M', 128, 128, 'M', 256, 256, 'M', 512,512, 'M', 512,512,'M'],
            'BEE16' : [64,64, 'M', 128, 128, 'M', 256, 256,256, 'M', 512,512,512, 'M',512,512,512,'M'],
            'BEE19' : [64,64, 'M', 128, 128, 'M', 256, 256,256,256, 'M', 512,512,512,512, 'M',512,512,512,512,'M'],
            'BEE11_fc' : [64, 'M', 128, 'M', 256, 256, 'M', 512,512, 'M'],
            'BEE13_fc' : [64,64, 'M', 128, 128, 'M', 256, 256, 'M', 512,512, 'M'],
            'BEE16_fc' : [64,64, 'M', 128, 128, 'M', 256, 256,256, 'M', 512,512,512, 'M'],
            'BEE19_fc' : [64,64, 'M', 128, 128, 'M', 256, 256,256,256, 'M', 512,512,512,512, 'M'],
            'BEE11_large' : [64, 'M', 128, 'M', 256, 256, 'M', 512,512],
            'BEE13_large' : [64,64, 'M', 128, 128, 'M', 256, 256, 'M', 512,512],
            'BEE16_large' : [64,64, 'M', 128, 128, 'M', 256, 256,256, 'M', 512,512,512],
            'BEE19_large' : [64,64, 'M', 128, 128, 'M', 256, 256,256,256, 'M', 512,512,512,512],
        }

        for key in VGG_types :
            if key in model :
                self.conv_layers = self.create_conv_laters(VGG_types[key])      

        if "reduced" in model :
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
            # simple version of fc layers
            self.fcs = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(512, num_classes),
            )
        
        elif "large" in model :
            self.fcs = nn.Sequential(
                nn.Linear(512 * 8 * 8, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, 2048),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(2048, num_classes),
            )
            
        elif "fc" in model :
            self.fcs = nn.Sequential(
                nn.Linear(512 * 4 * 4, 4096),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(4096, 2048),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(2048, num_classes),
            )
        
        else :
            self.fcs = nn.Sequential(
                nn.Linear(512 * 2 * 2, 1024),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(1024, num_classes),
            )

        # weight initialization
        if init_weights:
            self._initialize_weights()


    def forward(self, x):
        x = self.conv_layers(x)
        if "reduced" in self.model :
            x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fcs(x)
        return x

    # defint weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    # define a function to create conv layer taken the key of VGG_type dict 
    def create_conv_laters(self, architecture):
        layers = []
        in_channels = self.in_channels # 3

        for x in architecture:
            if type(x) == int: # int means conv layer
                out_channels = x

                layers += [Conv2d_BEE(in_channels, out_channels, self.interpolation),
                           nn.BatchNorm2d(x),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), ceil_mode=True)]
        
        return nn.Sequential(*layers)