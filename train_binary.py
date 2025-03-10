##########################
# Imports
##########################
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from binary_dataloader import ValDataLoader, TrainDataLoader
from config_binary import ClsCerConfig_binary
from models.untils import get_net
from train_binary_model import train_binary_model

##########################
# Local Imports
##########################
current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))


def run(config):
    cuda_dev = torch.device('cuda')
    net = get_net(config)
    train_data = TrainDataLoader(config)
    val_data = ValDataLoader(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.Log_folder, timestamp + '_binary')
    os.makedirs(log_dir, exist_ok=True)

    optimizer = optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # scheduler = None
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0)
    # scheduler = LambdaLR(optimizer, lr_lambda=google_lr_lambda)
    # scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    with open(r"{}/log_info.txt".format(log_dir), 'w') as f:
        f.writelines('mode_type = {}\n'.format(config.net))
        f.writelines('epoch = {}\n'.format(config.epochs))
        f.writelines('learning_rate = {}\n'.format(config.lr))
        f.writelines('batch size = {}\n'.format(config.batch_size))
        f.writelines('optmizer = {}\n'.format(type(optimizer).__name__))
        f.writelines('scheduler = {}\n'.format(type(scheduler).__name__) if scheduler else 'None')

    net = train_binary_model(net, optimizer, scheduler, train_data, val_data, log_dir, config, device=cuda_dev)


############################
# MAIN
############################
if __name__ == "__main__":
    config = ClsCerConfig_binary()
    parser = argparse.ArgumentParser(description="Run Training on Classification")
    parser.add_argument(
        "-e",
        "--epochs",
        default=config.epochs, type=int,
        help="Specify the number of epochs required for training"
    )
    parser.add_argument(
        "-b",
        "--batch",
        default=config.batch_size, type=int,
        help="Specify the batch size"
    )
    parser.add_argument(
        "-w",
        "--workers",
        default=config.num_workers, type=int,
        help="Specify the number of workers"
    )
    parser.add_argument(
        "--net",
        default=config.net,
        help="please input true net"
    )
    parser.add_argument(
        "--lr",
        default=config.lr, type=float,
        help="Learning Rate"
    )

    args = parser.parse_args()
    config.net = args.net
    config.epochs = args.epochs
    config.batch_size = args.batch
    config.num_workers = args.workers
    config.lr = args.lr

    run(config)
