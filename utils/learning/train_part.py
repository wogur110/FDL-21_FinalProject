import shutil
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
import time
import matplotlib.pyplot as plt

from utils.data.load_data import create_data_loaders
from utils.model.test_model import *

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    time0 = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    correct_1 = 0
    correct_5 = 0

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
        _, predicted = torch.topk(output, 5, 1, True, True)
        predicted = predicted.T
        correct_mat = predicted.eq(target.view(1, -1).expand_as(predicted))
        correct_1 += (correct_mat[:1].reshape(-1).float().sum(0)).item()
        correct_5 += (correct_mat[:5].reshape(-1).float().sum(0)).item()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()

    total_loss = total_loss / len_loader
    top1accuracy = correct_1 / len(data_loader.dataset) * 100
    top5accuracy = correct_5 / len(data_loader.dataset) * 100
    return total_loss, top1accuracy, top5accuracy, time.perf_counter() - time0

def validate(args, model, data_loader, loss_type):
    model.eval()
    start = time.perf_counter()
    metric_loss = 0.0
    correct = 0
    correct_1 = 0
    correct_5 = 0

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target = data
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(input)
            loss = loss_type(output, target)
            metric_loss += loss.item()
            _, predicted = torch.topk(output, 5, 1, True, True)
            predicted = predicted.T
            correct_mat = predicted.eq(target.view(1, -1).expand_as(predicted))
            correct_1 += (correct_mat[:1].reshape(-1).float().sum(0)).item()
            correct_5 += (correct_mat[:5].reshape(-1).float().sum(0)).item()

    metric_loss /= len(data_loader)
    top1accuracy = correct_1 / len(data_loader.dataset) * 100
    top5accuracy = correct_5 / len(data_loader.dataset) * 100
    return metric_loss, top1accuracy, top5accuracy, time.perf_counter() - start

def update_lr(metric_history, scheduler):
    """
    Decreases learning rate (lr), terminates training after 3 lr decreases
    Track validation accuracy, decrease lr by 0.2 when:
        1. validation accuracy worsens
        2. less than 0.2% absolute improvement last 3 iterations
    """
    val_accs = metric_history["valtop1"]
    if len(val_accs) < 20 : 
        return False
    decrease = False
    # decrease LR if validation acc worsens
    if val_accs[-1] < max(val_accs):
        decrease = True
    avg_2 = (val_accs[-2] + val_accs[-3]) / 2
    # decrease LR if validation accuracy doesn't improve by 0.2% (absolute)
    if abs(val_accs[-1] - avg_2) < 0.2:
        decrease = True
    if decrease :
        scheduler.step()

    return decrease

def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    """
    Save model parameter and hyperparameter for each epoch, and convert best_model if it is best
    """
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

def plot_result(args, loss_history, metric_history):
    """
    Plot loss history vs epochs and Accuracy history vs epochs
    """
    num_epochs = len(loss_history["train"])
    save_dir = args.exp_dir

    # plot loss progress
    plt.figure(1)
    plt.title("Train-Val Loss")
    plt.plot(range(1,num_epochs+1),loss_history["train"],label="train")
    plt.plot(range(1,num_epochs+1),loss_history["val"],label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    #plt.show()
    plt.savefig(save_dir / 'loss.png')

    # plot accuracy progress
    plt.figure(2)
    plt.title("Train-Val Accuracy")
    plt.plot(range(1,num_epochs+1),metric_history["traintop1"],label="traintop1")
    plt.plot(range(1,num_epochs+1),metric_history["valtop1"],label="valtop1")
    plt.plot(range(1,num_epochs+1),metric_history["traintop5"],label="traintop5")
    plt.plot(range(1,num_epochs+1),metric_history["valtop5"],label="valtop5")
    plt.ylabel("Accuracy")
    plt.xlabel("Training Epochs")
    plt.legend()
    #plt.show()
    plt.savefig(save_dir / 'accuracy.png')

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
    print(model)
    summary(model, input_size=(3, 64, 64), device=device.type) # for TinyImageNet, the input to the network is a 56x56 RGB crop.

    loss_type = nn.CrossEntropyLoss().to(device=device)
    #optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2) # track validation accuracy and decrease lr by 0.2

    best_val_loss = 10.
    start_epoch = 0

    # history of loss values in each epoch
    loss_history={
        "train": [],
        "val": [],
    }
    
    # histroy of metric values in each epoch
    metric_history={
        "traintop1": [],
        "valtop1": [],
        "traintop5": [],
        "valtop5": [],
    }

    decrease_lr_count = 0
    report_result = ""

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')

        train_loss, train_top1accuracy, train_top5accuracy, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
        val_loss, val_top1accuracy, val_top5accuracy, val_time = validate(args, model, val_loader, loss_type)

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir / 'checkpoints', epoch + 1, model, optimizer, best_val_loss, is_new_best)

        # collect loss and metric for training dataset
        loss_history["train"].append(train_loss)
        metric_history["traintop1"].append(train_top1accuracy)
        metric_history["traintop5"].append(train_top5accuracy)

        # collect loss and metric for validation dataset
        loss_history["val"].append(val_loss)
        metric_history["valtop1"].append(val_top1accuracy)
        metric_history["valtop5"].append(val_top5accuracy)

        result = f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} ' + \
            f'ValLoss = {val_loss:.4g} Valtop1Accuracy = {val_top1accuracy:.4g}% Valtop5Accuracy = {val_top5accuracy:.4g}% TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s\n'
        print(result)

        report_result += result

        if update_lr(metric_history, scheduler) :
            decrease_lr_count += 1
            result = '*** New learning rate: {}\n'.format(scheduler.get_last_lr())
            print(result)
            report_result += result
        
        # Decreases learning rate (lr), terminates training after 3 lr decreases
        if decrease_lr_count > 3 :
            break

    if not args.no_plot_result:
        plot_result(args, loss_history, metric_history)
    
    save_result_path = args.exp_dir / "result.txt"
    f = open(save_result_path, 'w')
    f.write(report_result)
    f.close()