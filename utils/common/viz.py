import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import matplotlib.pyplot as plt
from math import ceil
import torchvision

from utils.data.load_data import create_data_loaders
from utils.model.test_model import *
from pathlib import Path

def imshow(args, img):
    """Custom function to display the image using matplotlib"""
      
    std_correction = np.asarray([0.229, 0.224, 0.225]).reshape(3, 1, 1) #define std correction to be made
        
    mean_correction = np.asarray([0.485, 0.456, 0.406]).reshape(3, 1, 1) #define mean correction to be made
        
    npimg = np.multiply(img.numpy(), std_correction) + mean_correction #convert the tensor img to numpy img and de normalize 
    
    #plot the numpy image
    plt.figure(figsize = (4, 4))
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    save_path = args.exp_dir / 'image_viz.png'
    plt.savefig(save_path, dpi=100) 

def show_feature(args, dataloader, index, model):
    """
    custom function to show feature maps of input image from dataloader
    dataloader : dataloader of validation dataset
    index : index of image from validation dataset (seed fix)
    """

    # Find image which has correct index of validation dataset
    for idx, (images, _) in enumerate(dataloader) :
        if idx == index :
            break

    images = images.cuda(non_blocking=True)

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.conv_layers[3].register_forward_hook(get_activation('conv1'))
    model.conv_layers[10].register_forward_hook(get_activation('conv2'))
    model.conv_layers[20].register_forward_hook(get_activation('conv3'))
    model.conv_layers[30].register_forward_hook(get_activation('conv4'))
    
    outputs = model(images) #run the model on the images    
    
    _, pred = torch.max(outputs.data, 1) #get the maximum class     
    
    img = torchvision.utils.make_grid(images.cpu()) #make grid    
    
    imshow(args, img) #call the function

    act1 = activation['conv1'].squeeze().cpu()
    act2 = activation['conv2'].squeeze().cpu()
    act3 = activation['conv3'].squeeze().cpu()
    act4 = activation['conv4'].squeeze().cpu()

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

    #plot the feature of layer 3
    plt.figure(figsize = (50, 20))
    for i, filter in enumerate(act3) :
        if (i == 32) :
            break
        plt.subplot(4, 8, i + 1)
        plt.imshow(filter, cmap = 'gray')
        plt.axis("off")

    save_path = args.exp_dir / 'feature_3_viz.png'
    plt.savefig(save_path, dpi=100) 

    #plot the feature of layer 4
    plt.figure(figsize = (50, 20))
    for i, filter in enumerate(act4) :
        if (i == 32) :
            break
        plt.subplot(4, 8, i + 1)
        plt.imshow(filter, cmap = 'gray')
        plt.axis("off")

    save_path = args.exp_dir / 'feature_4_viz.png'
    plt.savefig(save_path, dpi=100) 
    
    return images, pred

def plot_weights(args, model, layer_num):
    """
    custom function to plot weight of convolution layer
    """

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
            
            # looping through all the kernels
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

    if "Hexa" in args.net_name :
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

        weight_tensor = torch.zeros((weight_ts2.shape[0], weight_ts2.shape[1], 3, 3)) # visualize weight of real kernel
        weight_tensor_params = torch.zeros((weight_ts2.shape[0], weight_ts2.shape[1], 7)) # visualize with 7 parameters in kernel
        weight_tensor[:, :, :1, :2] += weight_ts1 / 2
        weight_tensor[:, :, :1, 1:] += weight_ts1 / 2
        weight_tensor[:, :, 1:2, :] += weight_ts2
        weight_tensor[:, :, 2:3, :2] += weight_ts3 / 2
        weight_tensor[:, :, 2:3, 1:] += weight_ts3 / 2

        weight_tensor_params[:, :, :2] += weight_ts1.squeeze()
        weight_tensor_params[:, :, 2:5] += weight_ts2.squeeze()
        weight_tensor_params[:, :, 5:7] += weight_ts3.squeeze()
      
        num_kernels = weight_tensor.shape[0] #get the number of kernals
        
        # define number of columns and rows for subplots
        num_cols = 12
        num_rows = ceil(num_kernels / float(num_cols))

        ## visualize weight_tensor : weight of real kernel     
        fig = plt.figure(figsize=(num_cols, num_rows)) #set the figure size
        
        # looping through all the kernels
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

        ## visualize weight_tensor_params : 7 parameters in kernel
        fig = plt.figure(figsize=(num_cols, num_rows)) #set the figure size
        
        # looping through all the kernels
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
    """
    Visualize weight of kernel and feature map 
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current cuda device: ', torch.cuda.current_device())

    # Make val_loader
    val_loader = create_data_loaders(
        data_name=args.data_name,
        data_path=args.data_path_val, 
        batch_size=args.batch_size,
        num_workers=0,
        args=args,
        train=False
    )

    # Assign model wrt net_name
    assert args.net_name == "LinearNet" or args.net_name == "LeNet5" or "VGG" in args.net_name or "Hexa" in args.net_name
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
    if "Hexa" in args.net_name:
        # HexaNet is for TinyImageNet (spatial size : 64x64)
        model = HexaNet(
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
    summary(model, input_size=(3, 64, 64), device=device.type) # for TinyImageNet, the input to the network is a 64x64 RGB image.

    plot_weights(args, model, 0)  # custom function to plot weight of convolution layer

    images, pred = show_feature(args, val_loader, args.image_index, model) # custom function to show feature maps of input image from dataloader

def test_validate(args) :
    """
    print validate input image list which has correct prediction with Hexa16_fc_hue but wrong prediction with VGG16_fc
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current cuda device: ', torch.cuda.current_device())

    # Make val_loader
    val_loader = create_data_loaders(
        data_name=args.data_name,
        data_path=args.data_path_val, 
        batch_size=args.batch_size,
        num_workers=0,
        args=args,
        train=False
    )

    # Assign model_VGG with VGG16_fc and mode_Hexa with Hexa16_fc_hue
    model_VGG = VGGNet(
        model = 'VGG16_fc',
        in_channels = args.in_channels,
        num_classes = args.num_classes,
        init_weights = True,
        data_name = args.data_name
    )
    model_VGG.to(device=device)

    model_Hexa = HexaNet(
        model = 'Hexa16_fc_hue',
        in_channels = args.in_channels,
        num_classes = args.num_classes,
        init_weights = True,
        data_name = args.data_name
    )
    model_Hexa.to(device=device)

    images,_ = next(iter(val_loader))
    images = images.cuda(non_blocking=True)

    exp_dir1 = Path('./result/AdamW') / Path('VGG16_fc_TinyImageNet')
    exp_dir2 = Path('./result/AdamW') / Path('Hexa16_fc_hue_TinyImageNet')

    checkpoints_dir1 = exp_dir1 / 'checkpoints'
    checkpoints_dir2 = exp_dir2 / 'checkpoints'
    checkpoint1 = torch.load(checkpoints_dir1 / 'best_model.pt', map_location = 'cpu')
    checkpoint2 = torch.load(checkpoints_dir2 / 'best_model.pt', map_location = 'cpu')

    model_VGG.load_state_dict(checkpoint1['model'])
    model_VGG.eval()
    summary(model_VGG, input_size=(3, 64, 64), device=device.type)
    model_Hexa.load_state_dict(checkpoint2['model'])
    model_Hexa.eval()
    summary(model_Hexa, input_size=(3, 64, 64), device=device.type)

    # print validate input image list which has correct prediction with Hexa16_fc_hue but wrong prediction with VGG16_fc
    for idx, (images, targets) in enumerate(val_loader) :
        images = images.cuda(non_blocking=True)
        output_VGG = model_VGG(images)
        _, pred_VGG = torch.max(output_VGG.data, 1)
        output_Hexa = model_Hexa(images)
        _, pred_Hexa = torch.max(output_Hexa.data, 1)

        if targets == pred_Hexa.cpu() and targets != pred_VGG.cpu() :
            print(idx, targets, pred_Hexa.cpu(), pred_VGG.cpu())

        