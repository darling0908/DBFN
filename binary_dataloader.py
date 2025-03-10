import json

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class TrainDataLoader:
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.train_transform = config.transform_train
        train_dataset = ImageFolder('tianchi/train', transform=self.train_transform)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                       shuffle=True)

        data_list = train_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in data_list.items())
        # write dict into json file
        json_str = json.dumps(cla_dict, indent=4)
        with open('tianchi/train/class_indices.json', 'w') as json_file:
            json_file.write(json_str)

    def __iter__(self):
        return iter(self.train_loader)

    def __len__(self):
        return len(self.train_loader)


class ValDataLoader:
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.val_transform = config.transform_val
        val_dataset = ImageFolder('tianchi/val', transform=self.val_transform)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                     collate_fn=None)
        data_list = val_dataset.class_to_idx
        cla_dict = dict((val, key) for key, val in data_list.items())
        # write dict into json file
        json_str = json.dumps(cla_dict, indent=4)
        with open('tianchi/val/class_indices.json', 'w') as json_file:
            json_file.write(json_str)

        self.dataset_length = len(val_dataset)

    def __iter__(self):
        return iter(self.val_loader)

    def __len__(self):
        return len(self.val_loader)

    def get_dataset_length(self):
        return self.dataset_length