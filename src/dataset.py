import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TinyImageNetDataset(datasets.ImageFolder):
    """
    Wraps ImageFolder to support Albumentations.
    standard torchvision transforms are slower and less flexible.
    """
    def __init__(self, root, transform=None):
        super().__init__(root, transform=None)
        self.albumentations_transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.albumentations_transform:
            augmented = self.albumentations_transform(image=image)
            image = augmented['image']

        return image, target


def get_transforms(mean, std):
    """
    Returns training and validation transforms.
    Args:
        mean: Calculated in EDA (e.g., [0.485, 0.456, 0.406])
        std: Calculated in EDA (e.g., [0.229, 0.224, 0.225])
    """
    train_transform = A.Compose([
        # Spatial augmentations (Geometric)
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        # Pixel-level augmentations (Regularization)
        # CoarseDropout
        A.CoarseDropout(
            max_holes=1,
            max_height=16,
            max_width=16,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=mean,
            p=0.5
        ),
        # Normalization & Tensor conversion
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    return train_transform, val_transform


def get_dataloaders(data_dir, batch_size, mean, std, num_workers=4):
    train_tf, val_tf = get_transforms(mean, std)

    train_set = TinyImageNetDataset(root=os.path.join(data_dir, 'train'), transform=train_tf)
    val_set = TinyImageNetDataset(root=os.path.join(data_dir, 'val'), transform=val_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader