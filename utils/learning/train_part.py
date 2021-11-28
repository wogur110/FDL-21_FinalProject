import shutil
import numpy as np
import torch
import torch.nn as nn
import time

from utils.data.load_data import create_data_loaders
from utils.model.test_model import *

def train_epoch(args, epoch, model, data_loader, optimizer, scheduler, loss_type):
    model.train()
    time0 = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        input, target = data
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(input)
        loss = loss_type(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()

    scheduler.step()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - time0

def validate(args, model, data_loader, loss_type):
    model.eval()
    start = time.perf_counter()
    metric_loss = 0.0
    correct = 0

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target = data
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(input)
            loss = loss_type(output, target)
            metric_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()

    metric_loss /= len(data_loader)
    accuracy = correct / len(data_loader.dataset) * 100
    return metric_loss, accuracy, time.perf_counter() - start

def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Current cuda device: ', torch.cuda.current_device())

    train_loader = create_data_loaders(
        data_name=args.data_name,
        data_path=args.data_path_train, 
        batch_size=args.batch_size,
        num_workers=0,
        args=args,
        train=True
    )
    val_loader = create_data_loaders(
        data_name=args.data_name,
        data_path=args.data_path_val, 
        batch_size=args.batch_size,
        num_workers=0,
        args=args,
        train=False
    )

    assert str(args.net_name) == "LinearNet" or "basicCNN"
    if str(args.net_name) == "LinearNet":
        model = LinearNet(
            input_size = args.input_size, 
            num_classes = args.num_classes
        )
    model.to(device=device)

    loss_type = nn.CrossEntropyLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)

    best_val_loss = 1.
    start_epoch = 0

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')

        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, scheduler, loss_type)
        val_loss, val_accuracy, val_time = validate(args, model, val_loader, loss_type)

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} ValAccuracy = {val_accuracy:.4g}% TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )
