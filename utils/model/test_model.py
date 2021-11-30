import torch
from torch import nn
from torch.nn import functional as F

class LinearNet(nn.Module):
    def __init__(self, num_classes, data_name="MNIST"):
        super().__init__()
        assert data_name == "MNIST" or data_name == "CIFAR10" or data_name == "CIFAR100" or data_name == "ImageNet" or data_name == "ImageNet64"
        if data_name == "MNIST" :
            input_size = 1*28*28
        elif data_name == "CIFAR10" or data_name == "CIFAR100":
            input_size = 3*32*32
        elif data_name == "ImageNet" :
            input_size = 3*224*224
        elif data_name == "ImageNet64" :
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
    def __init__(self, model, in_channels=3, num_classes=1000, init_weights=True, data_name="ImageNet"):
        super().__init__()
        self.in_channels = in_channels

        # create conv_layers corresponding to VGG type
        # VGG type dict
        # int : output channels after conv layer
        # 'M' : max pooling layer
        VGG_types = {
            'VGG11' : [64, 'M', 128, 'M', 256, 256, 'M', 512,512, 'M',512,512,'M'],
            'VGG13' : [64,64, 'M', 128, 128, 'M', 256, 256, 'M', 512,512, 'M', 512,512,'M'],
            'VGG16' : [64,64, 'M', 128, 128, 'M', 256, 256,256, 'M', 512,512,512, 'M',512,512,512,'M'],
            'VGG19' : [64,64, 'M', 128, 128, 'M', 256, 256,256,256, 'M', 512,512,512,512, 'M',512,512,512,512,'M']
        }

        self.conv_layers = self.create_conv_laters(VGG_types[model])

        self.fcs = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        # weight initialization
        if init_weights:
            self._initialize_weights()


    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fcs(x)
        return x

    # defint weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
        
        return nn.Sequential(*layers)


class VGGNet_reduced(nn.Module):
    def __init__(self, model, in_channels=3, num_classes=1000, init_weights=True, data_name="ImageNet"):
        super().__init__()
        self.in_channels = in_channels

        # create conv_layers corresponding to VGG type
        # VGG type dict
        # int : output channels after conv layer
        # 'M' : max pooling layer
        VGG_types = {
            'VGG11_reduced' : [64, 'M', 128, 'M', 256, 256, 'M', 512,512, 'M',512,512,'M'],
            'VGG13_reduced' : [64,64, 'M', 128, 128, 'M', 256, 256, 'M', 512,512, 'M', 512,512,'M'],
            'VGG16_reduced' : [64,64, 'M', 128, 128, 'M', 256, 256,256, 'M', 512,512,512, 'M',512,512,512,'M'],
            'VGG19_reduced' : [64,64, 'M', 128, 128, 'M', 256, 256,256,256, 'M', 512,512,512,512, 'M',512,512,512,512,'M']
        }

        self.conv_layers = self.create_conv_laters(VGG_types[model])
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # simple version of fc layers
        self.fcs_reduced = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

        # weight initialization
        if init_weights:
            self._initialize_weights()


    def forward(self, x):
        x = self.conv_layers(x)
        # simple version of fc layers. In ImageNet, spatial size 7x7 -> 1x1
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fcs_reduced(x)
        return x

    # defint weight initialization function
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
                layers += [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
        
        return nn.Sequential(*layers)