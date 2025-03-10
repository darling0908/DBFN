from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, \
    ToTensor, GaussianBlur, RandomAffine, Normalize, Resize, RandomRotation

img_norm_cfg = {
    'mean': [0.612, 0.649, 0.726],
    'std': [0.142, 0.154, 0.099]
}


def train_transforms_dict():
    return Compose([
        Resize((224, 224)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(15),  # 随机旋转
        RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        GaussianBlur(3, sigma=(0.1, 2.0)),
        ToTensor(),
        Normalize(mean=img_norm_cfg['mean'], std=img_norm_cfg['std']),
    ])


def val_transforms_dict():
    return Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=img_norm_cfg['mean'], std=img_norm_cfg['std']),
    ])


def train_transforms_dict_tianchi():
    return Compose([
        Resize((224, 224)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(15),  # 随机旋转
        RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        GaussianBlur(3, sigma=(0.1, 2.0)),
        ToTensor(),
        Normalize(mean=(0.558, 0.570, 0.594), std=(0.123, 0.109, 0.082)),
    ])


def val_transforms_dict_tianchi():
    return Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=(0.559, 0.570, 0.596), std=(0.124, 0.111, 0.082)),
    ])


def test_transforms_dict_tianchi():
    return Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=(0.553, 0.566, 0.591), std=(0.122, 0.108, 0.081)),
    ])
