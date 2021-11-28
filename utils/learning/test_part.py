import numpy as np
import torch
import torch.nn as nn

from utils.data.load_data import create_data_loaders
from utils.model.test_model import *

def test_calc(args, model, data_loader, loss_type):
    model.eval()
    metric_loss = 0.0

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target = data
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(input)
            loss = loss_type(output, target)
            metric_loss += loss.item()

    metric_loss /= len(data_loader)
    return metric_loss

def test(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    model = LeNet(
        in_chans = args.in_chans, 
        out_chans = args.out_chans
    )
    model.to(device=device)

    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu')
    print(checkpoint['epoch'], checkpoint['best_val_loss'])
    model.load_state_dict(checkpoint['model'])

    loss_type = nn.CrossEntropyLoss().to(device=device)

    test_loader = create_data_loaders(
        data_path=args.data_path, 
        batch_size=args.batch_size,
        num_workers=0,
        args=args,
        train=False
    )

    test_loss = test_calc(args, model, test_loader, loss_type)
    print(f'TestLoss = {test_loss:.4g}')