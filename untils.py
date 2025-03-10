import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score
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
