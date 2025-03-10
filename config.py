from .augm import train_transforms_dict, val_transforms_dict


class ClsCerConfig:
    batch_size = 8
    num_workers = 0
    Log_folder = 'Log/'
    lr = 1e-4  #
    epochs = 500
    weight_decay = 5e-2
    num_outs = 4
    num_folders = 5  # 交叉验证
    transform_train = train_transforms_dict()
    transform_val = val_transforms_dict()
    net = "DB-FN"
