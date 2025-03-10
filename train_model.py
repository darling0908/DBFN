import gzip
import os
import time

import torch
from tqdm import tqdm

from untils import cross_entropy_loss, MetricsEvaluator, plt_loss, plt_single


def train_model(net, optimizer, scheduler, train_data, val_data, folder_dir, config, device=None):
    print('Start training...')
    net = net.to(device)
    # train loop
    output_file = '{}/{}'.format(folder_dir, 'metrics_results.txt')
    best_accuracy = 0.0
    val_accuracies = []  # 用于存储每个 epoch 的验证准确率
    labels_all = []
    predicted_labels_all = []
    plt_train_loss = []
    plt_val_loss = []

    for epoch in range(config.epochs):
        torch.cuda.empty_cache()
        epoch_start_time = time.time()
        training_loss = 0.0
        train_criterion = cross_entropy_loss()

        # switch to train mode
        net.train()
        train_loop = tqdm(enumerate(train_data), total=len(train_data), colour='blue')
        for i, data in train_loop:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = train_criterion(outputs, labels)

            # GoogLeNet forward pass
            # outputs, aux_logits2, aux_logits1 = net(inputs)
            # loss0 = train_criterion(outputs, labels.to(device))
            # loss1 = train_criterion(aux_logits1, labels.to(device))
            # loss2 = train_criterion(aux_logits2, labels.to(device))
            # loss = loss0 + loss1 * 0.3 + loss2 * 0.3

            if torch.isnan(loss):
                print("Loss became NaN. Stopping training.")
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save and print statistics
            training_loss += loss.item()
            epoch_end_time = time.time()
            epoch_elapsed_time = epoch_end_time - epoch_start_time
            train_loop.set_description(f'Epoch [{epoch + 1}/{config.epochs}]')
            train_loop.set_postfix(CE_loss=loss.item(), time=epoch_elapsed_time)

        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        # 切换到评估模式
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_criterion = cross_entropy_loss()

        with torch.no_grad():  # 不需要计算梯度，加速推理过程
            val_loop = tqdm(enumerate(val_data), total=len(val_data))
            for i, data in val_loop:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                loss = val_criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted_labels = torch.max(outputs, dim=1)
                labels_all.extend(labels.cpu().numpy().tolist())
                predicted_labels_all.extend(predicted_labels.cpu().numpy().tolist())

                total += labels.size(0)
                correct += torch.eq(predicted_labels, labels).sum().item()
                val_loop.set_description("Validate")
                val_loop.set_postfix(Val_CE_loss=loss.item(), Val_running_loss=val_loss / (i + 1))

        if torch.isnan(loss):
            break  # 如果损失变成 NaN，停止训练

        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)  # 记录当前 epoch 的验证指标
        plt_train_loss.append(training_loss / len(train_data))
        plt_val_loss.append(val_loss / len(val_data))
        print('[Epoch %d] Train Loss: %.3f | Validation Loss: %.3f | Validation Accuracy: %.3f' %
              (epoch + 1, training_loss / len(train_data), val_loss / len(val_data), val_accuracy))

        if val_accuracy > best_accuracy and epoch >= 10:
            best_accuracy = val_accuracy
            best_accuracy_path = os.path.join(folder_dir, 'models_best.pth')
            torch.save(net.state_dict(), best_accuracy_path)

    acc = sum(val_accuracies) / len(val_accuracies)
    metrics = MetricsEvaluator(config.num_outs)
    metrics.evaluate_metrics(labels_all, predicted_labels_all, acc=acc,
                             output_file=output_file)
    avg_pre, avg_recall, avg_f1, auc, cm = metrics.get_metrics()
    plt_loss(config, plt_train_loss, plt_val_loss, folder_dir)
    plt_single(config, plt_train_loss, folder_dir)
    print('This Training\'s val accuracy is {:.4f}'.format(acc))
    print('Training ended!')

    return net, acc, avg_pre, avg_recall, avg_f1, cm, auc
