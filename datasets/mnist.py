"""MNIST dataset loader.

This module wraps the torchvision MNIST dataset and returns PyTorch
dataloaders.  A normalisation transform is applied by default.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple

def get_dataloaders(train_batch_size: int = 64, test_batch_size: int = 1000,
                    root: str = './data', download: bool = True, num_workers: int = 0,
                    shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
    """Return training and testing dataloaders for the MNIST dataset.

    Parameters
    ----------
    train_batch_size: int
        Batch size for the training set.
    test_batch_size: int
        Batch size for the test set.
    root: str
        Directory to store the dataset.
    download: bool
        If ``True``, downloads the dataset if not present.
    num_workers: int
        Number of worker processes used by the dataloaders.
    shuffle: bool
        Whether to shuffle the training dataset.

    Returns
    -------
    (DataLoader, DataLoader)
        A tuple containing the training and testing dataloaders.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=transform)
    testset = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=transform)
    train_loader = DataLoader(trainset, batch_size=train_batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def get_calibration_loader(root: str = './data', batch_size: int = 1,
                           num_samples: int = 200, num_workers: int = 0) -> DataLoader:
    """Return a small subset of the training data for quantisation calibration.

    This helper constructs a DataLoader over a subset of the MNIST training set
    limited to ``num_samples`` examples.  During calibration only the input
    tensors are used.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
    subset = torch.utils.data.Subset(dataset, range(num_samples))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader