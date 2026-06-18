"""Synthetic sine wave dataset for sequence prediction.

This dataset generates a continuous sine wave and slices it into input
sequences and target values.  It can be used to train and evaluate LSTM
models on a simple regression task.
"""

from __future__ import annotations

import math
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional

class SineWaveDataset(Dataset):
    """PyTorch dataset for a synthetic sine wave time series.

    The dataset generates a sine wave and constructs pairs of
    `(input_sequence, target_value)` where the target is the value one
    step after the end of the sequence.

    Parameters
    ----------
    seq_len: int
        Length of the input sequences.
    total_length: int
        Total number of time steps to generate.  The number of samples
        produced by the dataset will be `total_length - seq_len`.
    noise_std: float
        Standard deviation of Gaussian noise added to the signal; set to
        zero for a clean sine wave.
    """
    def __init__(self, seq_len: int = 12, total_length: int = 2000, noise_std: float = 0.0) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.total_length = total_length
        self.noise_std = noise_std
        # Precompute the sine wave
        t = torch.linspace(0, 4 * math.pi, total_length + 1)
        signal = torch.sin(t)
        if noise_std > 0:
            signal += noise_std * torch.randn_like(signal)
        # Normalise to roughly [-1,1]
        self.data = signal

    def __len__(self) -> int:
        return self.total_length - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.data[idx : idx + self.seq_len]
        target = self.data[idx + self.seq_len]
        # Reshape to (seq_len, 1) as required by LSTM (batch_first=False)
        return seq.unsqueeze(-1), target.unsqueeze(-1)

def get_dataloaders(seq_len: int = 12, total_length: int = 2400000, train_ratio: float = 0.8,
                    batch_size: int = 16, noise_std: float = 0.05, shuffle: bool = True,
                    num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    """Return training and testing dataloaders for the sine wave dataset.

    The time series is split into a training and testing portion
    according to ``train_ratio``.
    """
    dataset = SineWaveDataset(seq_len=seq_len, total_length=total_length, noise_std=noise_std)
    n_train = int(len(dataset) * train_ratio)
    n_test = len(dataset) - n_train
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

def get_calibration_loader(seq_len: int = 12, total_length: int = 2400000, num_samples: int = 200000,
                           batch_size: int = 1, noise_std: float = 0.05, num_workers: int = 0) -> DataLoader:
    """Return a calibration loader for static quantisation.

    A subset of the sine wave sequence is used for calibration.  Note that
    calibration only uses input sequences and ignores targets.
    """
    dataset = SineWaveDataset(seq_len=seq_len, total_length=total_length, noise_std=noise_std)
    subset = torch.utils.data.Subset(dataset, range(min(num_samples, len(dataset))))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader