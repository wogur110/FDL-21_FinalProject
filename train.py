import argparse
import torch
from utils.learning.train_part import train
from utils.common.utils import seed_fix
from pathlib import Path

def parse():
    parser = argparse.ArgumentParser(description='Final Project for FDL-21',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('-w', '--workers', type=int, default=4, help='# of workers for data loader')
    parser.add_argument('-e', '--num-epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-s', '--step', type=int, default=20, help='Step size for StepLR scheduler')
    parser.add_argument('-r', '--report-interval', type=int, default=100, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='LinearNet', help='Name of network, LinearNet or basicCNN')

    parser.add_argument('-d', '--data-name', type=str, default='MNIST', help='Name of dataset')
    parser.add_argument('-t', '--data-path-train', type=Path, default='./Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='./Data/val/', help='Directory of validation data')

    parser.add_argument('--seed', type=int, default=None, help='Fix random seed')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    args.exp_dir = './result' / args.net_name / 'checkpoints'
    args.exp_dir.mkdir(parents=True, exist_ok=True)

    assert args.data_name == "MNIST"
    if args.data_name == "MNIST" :
        args.input_size = 1*28*28
        args.num_classes = 10

    if args.seed is not None:
        seed_fix(args.seed)

    train(args)