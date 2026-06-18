"""Evaluation helpers for trained models."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple, Optional

def evaluate_classification(model: nn.Module, data_loader: DataLoader,
                            device: Optional[str] = None) -> float:
    """Compute classification accuracy over a dataset.

    Parameters
    ----------
    model: nn.Module
        The trained classifier.
    data_loader: DataLoader
        Dataloader providing `(inputs, targets)` pairs.
    device: str, optional
        Device on which to run the evaluation.  If omitted, will use CUDA
        if available.

    Returns
    -------
    float
        The accuracy in the range [0,1].
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total if total > 0 else 0.0

def evaluate_regression(model: nn.Module, data_loader: DataLoader,
                        device: Optional[str] = None) -> float:
    """Compute mean squared error over a regression dataset.

    Parameters
    ----------
    model: nn.Module
        The trained regression model.
    data_loader: DataLoader
        Dataloader providing `(inputs, targets)` pairs.
    device: str, optional
        Device on which to run the evaluation.  If omitted, will use CUDA
        if available.

    Returns
    -------
    float
        The mean squared error.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)
    mse_loss = nn.MSELoss(reduction='mean')
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            # If targets have trailing singleton dimension, remove it
            if targets.dim() > 1 and targets.size(-1) == 1:
                targets = targets.squeeze(-1)
            if outputs.dim() > 1 and outputs.size(-1) == 1:
                outputs = outputs.squeeze(-1)
            loss = mse_loss(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            count += targets.size(0)
    return total_loss / count if count > 0 else float('nan')