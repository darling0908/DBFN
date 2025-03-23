import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.data[index]  # Assuming data is a list of tuples (image, label)
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)


def custom_collate_fn(batch):
    images = []
    labels = []
    for img, _ in batch:
        images.append(torch.from_numpy(np.array(img)))
    for _, label in batch:
        labels.append(label)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels)
    return images, labels


class CrossValidationDataLoader:
    def __init__(self, config, k_fold):
        self.k_fold = k_fold
        self.config = config
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.train_transform = config.transform_train
        self.val_transform = config.transform_val

        # Load entire dataset
        full_dataset = ImageFolder('E:/PycharmProjects/Cer_Classification/CrossValidate_Data/train')
        num_samples = len(full_dataset)

        # Divide dataset by class
        class_to_indices = {}
        for idx, (image, label) in enumerate(full_dataset):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)

        train_transforms = [self.train_transform, self.train_transform, self.train_transform, self.train_transform,
                            self.train_transform, ]  # list of transforms for each fold
        val_transforms = [self.val_transform, self.val_transform, self.val_transform, self.val_transform,
                          self.val_transform, ]

        # Initialize data loaders for each fold
        self.train_data = []
        self.val_data = []
        self.train_loaders = []
        self.val_loaders = []

        for fold_idx in range(k_fold):
            val_indices = []
            for class_indices in class_to_indices.values():
                num_samples_per_fold = len(class_indices) // k_fold
                fold_indices = class_indices[fold_idx * num_samples_per_fold: (fold_idx + 1) * num_samples_per_fold]
                if fold_idx == k_fold - 1:  # For the last fold, take remaining samples
                    fold_indices = class_indices[fold_idx * num_samples_per_fold:]

                val_indices.extend(fold_indices)

            # Get train indices (exclude val indices)
            all_indices = list(range(num_samples))
            train_indices = [idx for idx in all_indices if idx not in val_indices]

            # Shuffle training indices
            np.random.shuffle(train_indices)

            # Create subset datasets for training and validation
            train_dataset = Subset(full_dataset, train_indices)
            val_dataset = Subset(full_dataset, val_indices)

            self.train_data.append(train_dataset)
            self.val_data.append(val_dataset)

        for fold in range(k_fold):
            # Create training and validation datasets with appropriate transforms
            train_dataset = CustomDataset(self.train_data[fold], transform=train_transforms[fold])
            val_dataset = CustomDataset(self.val_data[fold], transform=val_transforms[fold])

            # Create data loaders for training and validation
            train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                      shuffle=True)
            val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                    collate_fn=custom_collate_fn)

            # Add data loaders to the respective lists
            self.train_loaders.append(train_loader)
            self.val_loaders.append(val_loader)

    def get_train_loaders(self):
        return self.train_loaders

    def get_val_loaders(self):
        return self.val_loaders

