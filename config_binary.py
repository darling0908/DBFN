from .augm import train_transforms_dict_tianchi, val_transforms_dict_tianchi, test_transforms_dict_tianchi


class ClsCerConfig_binary:
    batch_size = 8   # baseline = 32 , new_net=8
    num_workers = 0
    Log_folder = 'Log/'
    test_folder = 'Log/tests'
    lr = 0.0001    #
    epochs = 60
    weight_path = None
    weight_decay = 3e-4
    num_outs = 2
    num_folders = 5  # 交叉验证

    transform_train = train_transforms_dict_tianchi()
    transform_val = val_transforms_dict_tianchi()
    transform_test = test_transforms_dict_tianchi()
    net = "v2-b"

