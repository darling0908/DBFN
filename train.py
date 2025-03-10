import argparse
import os
import sys
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from CrossValidationDataLoader import CrossValidationDataLoader
from config import ClsCerConfig
from models.untils import get_net
from train_model import train_model

current_path_abs = os.path.abspath('.')
sys.path.append(current_path_abs)
print('{} appended to sys!'.format(current_path_abs))


def run(config):

    cuda_dev = torch.device('cuda')
    net = get_net(config)
    cross_validation_loader = CrossValidationDataLoader(config, config.num_folders)
    validation_accuracies, validation_precisions, validation_precisions, validation_recalls, validation_f1s, all_auc = [
    ], [], [], [], [], []
    overall_confusion_matrix = np.zeros((4, 4))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.Log_folder, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    optimizer = optim.AdamW(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    scheduler = None
    # scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=0)
    # scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    for fold in range(config.num_folders):
        train_data_loader = cross_validation_loader.get_train_loaders()[fold]
        val_data_loader = cross_validation_loader.get_val_loaders()[fold]
        folder_dir = os.path.join(log_dir, 'Fold {}'.format(fold + 1))
        os.makedirs(folder_dir, exist_ok=True)
        with open(r"{}/log_info.txt".format(log_dir), 'w') as f:
            f.writelines('mode_type = {}\n'.format(config.net))
            f.writelines('epoch = {}\n'.format(config.epochs))
            f.writelines('learning_rate = {}\n'.format(config.lr))
            f.writelines('batch size = {}\n'.format(config.batch_size))
            f.writelines('weight_decay = {}\n'.format(config.weight_decay))
            f.writelines('optmizer = {}\n'.format(type(optimizer).__name__))
            f.writelines('scheduler = {}\n'.format(type(scheduler).__name__) if scheduler else 'None')
        net, val_accuracy, avg_pre, avg_recall, avg_f1, cm, auc = train_model(net, optimizer, scheduler,
                                                                              train_data_loader,
                                                                              val_data_loader, folder_dir,
                                                                              config, device=cuda_dev)
        validation_accuracies.append(val_accuracy)
        validation_precisions.append(avg_pre)
        validation_recalls.append(avg_recall)
        validation_f1s.append(avg_f1)
        all_auc.append(auc)
        overall_confusion_matrix += cm
        if fold != config.num_folders - 1:
            print('Fold {} is done!'.format(fold + 1))
        else:
            print('All train is done!')
    # 计算平均验证指标
    average_val_accuracy = sum(validation_accuracies) / len(validation_accuracies)
    average_val_pre = sum(validation_precisions) / len(validation_precisions)
    average_val_recall = sum(validation_recalls) / len(validation_recalls)
    average_val_f1 = sum(validation_f1s) / len(validation_f1s)
    average_auc = sum(all_auc) / len(all_auc)
    confusion_matrix = overall_confusion_matrix / config.num_folders
    class_totals = confusion_matrix.sum(axis=1)

    # 归一化混淆矩阵
    normalized_confusion_matrix = confusion_matrix / class_totals[:, None]

    with open(r"{}/Average_val_metrics.txt".format(log_dir), 'w') as f:
        f.writelines("Average Validation Accuracy: {}\n".format(average_val_accuracy))
        f.writelines("Average Validation Precision: {}\n".format(average_val_pre))
        f.writelines("Average Validation Recall: {}\n".format(average_val_recall))
        f.writelines("Average Validation F1 Score: {}\n".format(average_val_f1))
        f.writelines("Average Validation AUC: {}\n".format(average_auc))
        f.write("-" * 37 + "\n")
        for row in normalized_confusion_matrix:
            f.write("| ")
            f.write(" | ".join([f"{val:.4f}" for val in row]))
            f.write(" |\n")
        f.write("-" * 37 + "\n")


if __name__ == "__main__":
    config = ClsCerConfig()
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
