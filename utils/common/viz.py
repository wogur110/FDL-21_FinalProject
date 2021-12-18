import shutil
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import time
import matplotlib.pyplot as plt
from math import ceil
import torchvision

from utils.data.load_data import create_data_loaders
from utils.model.test_model import *

def imshow(args, img):
    """Custom function to display the image using matplotlib"""
  
    #define std correction to be made
    std_correction = np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    
    #define mean correction to be made
    mean_correction = np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    
    #convert the tensor img to numpy img and de normalize 
    npimg = np.multiply(img.numpy(), std_correction) + mean_correction
    
    #plot the numpy image
    plt.figure(figsize = (4, 4))
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    save_path = args.exp_dir / 'image_viz.png'
    plt.savefig(save_path, dpi=100) 

def show_feature(args, dataloader, model):
    """custom function to fetch images from dataloader"""
    images,_ = next(iter(dataloader))
    images = images.cuda(non_blocking=True)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    if "VGG" in args.net_name :
        model.conv_layers[3].register_forward_hook(get_activation('conv1'))
        model.conv_layers[10].register_forward_hook(get_activation('conv2'))
    elif "BEE" in args.net_name :
        model.conv_layers[3].register_forward_hook(get_activation('conv1'))
        model.conv_layers[10].register_forward_hook(get_activation('conv2'))
    
    outputs = model(images) #run the model on the images    
    
    _, pred = torch.max(outputs.data, 1) #get the maximum class     
    
    img = torchvision.utils.make_grid(images.cpu()) #make grid    
    
    imshow(args, img) #call the function

    act1 = activation['conv1'].squeeze().cpu()
    act2 = activation['conv2'].squeeze().cpu()

    #plot the feature of layer 1
    plt.figure(figsize = (50, 20))
    for i, filter in enumerate(act1) :
        if (i == 32) :
            break
        plt.subplot(4, 8, i + 1)
        plt.imshow(filter, cmap = 'gray')
        plt.axis("off")

    save_path = args.exp_dir / 'feature_1_viz.png'
    plt.savefig(save_path, dpi=100)

    #plot the feature of layer 2
    plt.figure(figsize = (50, 20))
    for i, filter in enumerate(act2) :
        if (i == 32) :
            break
        plt.subplot(4, 8, i + 1)
        plt.imshow(filter, cmap = 'gray')
        plt.axis("off")

    save_path = args.exp_dir / 'feature_2_viz.png'
    plt.savefig(save_path, dpi=100) 
    
    return images, pred

def plot_weights(args, model, layer_num):
    if "VGG" in args.net_name :
        #extracting the model features at the particular layer number
        layer = model.conv_layers[layer_num]
        
        #checking whether the layer is convolution layer or not 
        if isinstance(layer, nn.Conv2d):            
            weight_tensor = model.conv_layers[layer_num].weight.data.cpu() #getting the weight tensor data
            num_kernels = weight_tensor.shape[0] #get the number of kernals
            
            #define number of columns and rows for subplots
            num_cols = 12
            num_rows = ceil(num_kernels / float(num_cols))
                        
            fig = plt.figure(figsize=(num_cols, num_rows)) #set the figure size
            
            #looping through all the kernels
            for i in range(num_kernels):
                ax1 = fig.add_subplot(num_rows,num_cols,i+1)
                
                #for each kernel, we convert the tensor to numpy 
                npimg = np.array(weight_tensor[i].numpy(), np.float32)
                #standardize the numpy image
                npimg = (npimg - np.mean(npimg)) / np.std(npimg)
                npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
                npimg = npimg.transpose((1, 2, 0))[:,:,:3]
                ax1.imshow(npimg)
                ax1.axis('off')
                ax1.set_title(str(i))
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])

            save_path = args.exp_dir / f'kernel_viz_{layer_num}.png'            
            plt.savefig(save_path, dpi=100)    
            plt.tight_layout()
                
        else:
            print("Can only visualize layers which are convolutional")

    if "BEE" in args.net_name :
        #extracting the model features at the particular layer number

        #checking whether the layer is convolution layer or not 
        try : 
            layer1 = model.conv_layers[layer_num].Conv_layer1
            layer2 = model.conv_layers[layer_num].Conv_layer2
            layer3 = model.conv_layers[layer_num].Conv_layer3
        except : 
            print("Can only visualize layers which are convolutional")
            return

        weight_ts1 = layer1.weight.data.cpu()
        weight_ts2 = layer2.weight.data.cpu()
        weight_ts3 = layer3.weight.data.cpu()

        weight_tensor = torch.zeros((weight_ts2.shape[0], weight_ts2.shape[1], 3, 3)) # visualize real kernel
        weight_tensor_params = torch.zeros((weight_ts2.shape[0], weight_ts2.shape[1], 7)) # visualize with 7 parameters in BEENet kernel
        weight_tensor[:, :, :1, :2] += weight_ts1 / 2
        weight_tensor[:, :, :1, 1:] += weight_ts1 / 2
        weight_tensor[:, :, 1:2, :] += weight_ts2
        weight_tensor[:, :, 2:3, :2] += weight_ts3 / 2
        weight_tensor[:, :, 2:3, 1:] += weight_ts3 / 2

        weight_tensor_params[:, :, :2] += weight_ts1.squeeze()
        weight_tensor_params[:, :, 2:5] += weight_ts2.squeeze()
        weight_tensor_params[:, :, 5:7] += weight_ts3.squeeze()
      
        num_kernels = weight_tensor.shape[0] #get the number of kernals
        
        #define number of columns and rows for subplots
        num_cols = 12
        num_rows = ceil(num_kernels / float(num_cols))

        # visualize weight_tensor            
        fig = plt.figure(figsize=(num_cols, num_rows)) #set the figure size
        
        #looping through all the kernels
        for i in range(num_kernels):
            ax1 = fig.add_subplot(num_rows,num_cols,i+1)
            
            #for each kernel, we convert the tensor to numpy
            npimg = np.array(weight_tensor[i].numpy(), np.float32)
            #standardize the numpy image
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
            npimg = npimg.transpose((1, 2, 0))[:,:,:3]
            ax1.imshow(npimg)
            ax1.axis('off')
            ax1.set_title(str(i))
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

        save_path = args.exp_dir / f'kernel_viz_{layer_num}.png'            
        plt.savefig(save_path, dpi=100)    
        plt.tight_layout()

        # visualize weight_tensor_params
        fig = plt.figure(figsize=(num_cols, num_rows)) #set the figure size
        
        #looping through all the kernels
        for i in range(num_kernels):
            ax1 = fig.add_subplot(num_rows,num_cols,i+1)
            
            #for each kernel, we convert the tensor to numpy
            npimg = np.array(weight_tensor_params[i].numpy(), np.float32)
            #standardize the numpy image
            npimg = (npimg - np.mean(npimg)) / np.std(npimg)
            npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))

            npimg_plot = np.ones(shape = (npimg.shape[0], 3, 3))
            npimg_plot[:,0,:2] = npimg[:,:2]
            npimg_plot[:,1,:3] = npimg[:,2:5]
            npimg_plot[:,2,:2] = npimg[:,5:7]

            npimg_plot = npimg_plot.transpose((1, 2, 0))[:,:,:3]
            ax1.imshow(npimg_plot)
            ax1.axis('off')
            ax1.set_title(str(i))
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])

        save_path = args.exp_dir / f'kernel_viz_params_{layer_num}.png'            
        plt.savefig(save_path, dpi=100)    
        plt.tight_layout()
            
def kernel_viz(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current cuda device: ', torch.cuda.current_device())

    val_loader = create_data_loaders(
        data_name=args.data_name,
        data_path=args.data_path_val, 
        batch_size=args.batch_size,
        num_workers=0,
        args=args,
        train=False
    )

    images,_ = next(iter(val_loader))
    images = images.cuda(non_blocking=True)

    assert args.net_name == "LinearNet" or args.net_name == "LeNet5" or "VGG" in args.net_name or "BEE" in args.net_name
    if args.net_name == "LinearNet":
        model = LinearNet(
            num_classes = args.num_classes,
            data_name = args.data_name
        )
    if args.net_name == "LeNet5":
        model = LeNet5(
            num_classes = args.num_classes,
            data_name = args.data_name
        )
    if "VGG" in args.net_name:
        # VGGNet is for TinyImageNet (spatial size : 64x64)
        model = VGGNet(
            model = args.net_name,
            in_channels = args.in_channels,
            num_classes = args.num_classes,
            init_weights = True,
            data_name = args.data_name
        )            
    if "BEE" in args.net_name:
        # BEENet is for TinyImageNet (spatial size : 64x64)
        model = BEENet(
            model = args.net_name,
            in_channels = args.in_channels,
            num_classes = args.num_classes,
            init_weights = True,
            data_name = args.data_name
        )
    model.to(device=device)

    checkpoints_dir = args.exp_dir / 'checkpoints'
    checkpoint = torch.load(checkpoints_dir / 'best_model.pt', map_location = 'cpu')
    print(checkpoint['epoch'], checkpoint['best_val_loss'])

    model.load_state_dict(checkpoint['model'])
    model.eval()
    summary(model, input_size=(3, 64, 64), device=device.type)

    plot_weights(args, model, 0)

    images, pred = show_feature(args, val_loader, model)