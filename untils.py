import os

import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, auc, recall_score, precision_score, accuracy_score, \
    confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from torch import nn


def cross_entropy_loss():
    criterion = nn.CrossEntropyLoss()
    return criterion


def compute_four_class_auc(y_true, y_pred):
    """
    计算四分类问题的AUC

    参数:
    y_true: 真实的标签，类别的数字表示形式
    y_pred: 预测的标签

    返回值:
    auc: 四分类AUC值
    """
    # 将标签转换为二进制矩阵
    lb = LabelBinarizer()
    y_true_binary = lb.fit_transform(y_true)

    # 初始化AUC值列表
    auc_list = []
    y_pred = np.array(y_pred)
    # 计算每个类别的AUC
    for i in range(y_true_binary.shape[1]):
        auc = roc_auc_score(y_true_binary[:, i], (y_pred == i).astype(int))
        auc_list.append(auc)

    # 计算平均AUC
    auc_mean = np.mean(auc_list)

    return auc_mean

class MetricsEvaluator:
    def __init__(self, num_classes):
        self.cm = None
        self.auc = None
        self.avg_f1 = None
        self.avg_recall = None
        self.avg_precision = None
        self.num_classes = num_classes
        self.class_labels = range(num_classes)

    def evaluate_metrics(self, labels_all, predicted_labels_all, acc, output_file):
        precision, recall, f1, macro_prec, macro_recall, macro_f1, self.cm = calculate_metrics(labels_all,
                                                                                               predicted_labels_all)
        self.auc = compute_four_class_auc(labels_all, predicted_labels_all)
        # 计算四类指标的平均值
        self.avg_precision = sum(precision) / len(precision)
        self.avg_recall = sum(recall) / len(recall)
        self.avg_f1 = sum(f1) / len(f1)

        # 创建一个 PrettyTable 对象
        table = PrettyTable()
        # 设置表头
        table.field_names = ["Class", "Pre", "Recall", "F1"]
        # 设置单元格边框样式
        table.horizontal_char = '-'
        table.vertical_char = '|'
        table.junction_char = '+'

        # 设置每列的对齐方式为居中
        table.align["Class"] = "c"
        table.align["Pre"] = "c"
        table.align["Recall"] = "c"
        table.align["F1"] = "c"

        # 计算每列的最大宽度
        max_label_len = max(len("Class"), max(len(str(label)) for label in self.class_labels))
        max_precision_len = max(len("Pre"), max(len(f"{prec:.4f}") for prec in precision))
        max_recall_len = max(len("Recall"), max(len(f"{rec:.4f}") for rec in recall))
        max_f1_len = max(len("F1"), max(len(f"{f1score:.4f}") for f1score in f1))

        # 添加每个类别的指标
        for label, prec, rec, f1score in zip(self.class_labels, precision, recall, f1):
            table.add_row([f"{label}".ljust(max_label_len), f"{prec:.4f}".ljust(max_precision_len),
                           f"{rec:.4f}".ljust(max_recall_len), f"{f1score:.4f}".ljust(max_f1_len)])

        # 添加四类指标的平均值
        table.add_row(["Average".ljust(max_label_len), f"{self.avg_precision:.4f}".ljust(max_precision_len),
                       f"{self.avg_recall:.4f}".ljust(max_recall_len), f"{self.avg_f1:.4f}".ljust(max_f1_len)])

        # 将表格内容写入到文件
        with open(output_file, 'w') as f:
            f.write(str(table))
            f.write('\n')
            f.write('This Valid Average Acc = {}\n'.format(acc))
            f.writelines('This Valid Maco Presicon = {}\n'.format(macro_prec))
            f.writelines('This Valid Maco Recall = {}\n'.format(macro_recall))
            f.writelines('This Valid Maco F1-Score = {}\n'.format(macro_f1))
            f.writelines('This Valid  AUC = {}\n'.format(self.auc))
            f.writelines('The confusion matrix:\n{}'.format(np.array2string(self.cm)))

    def get_metrics(self):
        return self.avg_precision, self.avg_recall, self.avg_f1, self.auc, self.cm



def plt_loss(config, train_loss, val_loss, log_dir):
    plt.clf()  # 清除当前图形
    plt.plot(range(1, config.epochs + 1), train_loss, label='Training Loss', color='blue')
    plt.plot(range(1, config.epochs + 1), val_loss, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # 保存图形到指定位置
    plt.savefig('{}/TrainAndVal_loss_plot.png'.format(log_dir))


def plt_single(config, train_loss, log_dir):
    plt.clf()  # 清除当前图形
    plt.plot(range(1, config.epochs + 1), train_loss, label='Training Loss', color='blue')
    plt.title('Training  Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # 保存图形到指定位置
    plt.savefig('{}/Training_loss_plot.png'.format(log_dir))


class BinaryClassificationMetrics:
    def __init__(self, y_true, y_pred, y_pred_prob):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_pred_prob = np.array(y_pred_prob)
        self.conf_matrix = confusion_matrix(self.y_true, self.y_pred)
        self.tn, self.fp, self.fn, self.tp = self.conf_matrix.ravel()

    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def precision(self):
        return precision_score(self.y_true, self.y_pred)

    def recall(self):
        return recall_score(self.y_true, self.y_pred)

    def sensitivity(self):
        return recall_score(self.y_true, self.y_pred)

    def specificity(self):
        # actual_negatives = np.sum(self.y_true == 0)
        # # print(actual_negatives)
        # true_negatives = np.sum((self.y_true == 0) & (self.y_pred == 0))
        # # print(true_negatives)
        # return true_negatives / actual_negatives if actual_negatives else 0
        return self.tn / (self.tn + self.fp) if (self.tn + self.fp) != 0 else 0

    def f1_score(self):
        return f1_score(self.y_true, self.y_pred)

    def roc_auc(self):
        return roc_auc_score(self.y_true, self.y_pred_prob)

    def plot_roc_curve(self, config, save_path=None):
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_prob)
        save_roc_data(fpr, tpr, config, save_dir='{}/roc_data'.format(config.test_folder))
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        if save_path:
            plt.savefig(save_path)

    def save_metrics(self, file_path):
        with open(file_path, 'w') as f:
            f.write("Accuracy: {}\n".format(self.accuracy()))
            f.write("Precision: {}\n".format(self.precision()))
            f.write("Recall: {}\n".format(self.recall()))
            f.write("Sensitivity: {}\n".format(self.sensitivity()))
            f.write("Specificity: {}\n".format(self.specificity()))
            f.write("F1 Score: {}\n".format(self.f1_score()))
            f.write("ROC AUC: {}\n".format(self.roc_auc()))


def save_roc_data(fpr, tpr, config, save_dir=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, f'{config.net}_fpr.npy'), fpr)
    np.save(os.path.join(save_dir, f'{config.net}_tpr.npy'), tpr)