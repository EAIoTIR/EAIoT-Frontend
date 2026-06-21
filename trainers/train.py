"""Generic training loop implementation.

This module provides a helper function for running a standard training loop
over a PyTorch dataloader.  It is intentionally simple and intended to
illustrate the basic pattern of training a model.  Users with more
sophisticated requirements should feel free to adapt it.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Callable, Optional


def train_model(model: nn.Module, train_loader: DataLoader, criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                optimizer: Optimizer, epochs: int = 1, device: Optional[str] = None,
                log_interval: int = 100,
                epoch_end_callback: Optional[Callable[[nn.Module, int], None]] = None) -> nn.Module:
    """Train a model for a number of epochs.

    Parameters
    ----------
    model: nn.Module
        The model to train.
    train_loader: DataLoader
        A dataloader providing batches of `(inputs, targets)`.
    criterion: callable
        Loss function.
    optimizer: Optimizer
        Optimiser used to update model parameters.
    epochs: int
        Number of full passes over the training data.
    device: str, optional
        Device on which to run training (e.g. ``'cuda'`` or ``'cpu'``).  If
        omitted, uses CUDA if available.
    log_interval: int
        How frequently (in batches) to print progress information.
    epoch_end_callback: callable, optional
        Function called as ``epoch_end_callback(model, epoch_number)`` after
        each epoch. This is used by the CLI to apply pruning during training.

    Returns
    -------
    nn.Module
        The trained model (returned for convenience).
    """
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # If targets have an extra dimension (e.g. (batch,1)), squeeze it
            if not (outputs.dim() > 1 and outputs.size(-1) == 1) and (targets.dim() > 1 and targets.size(-1) == 1):
                targets = targets.squeeze(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if log_interval > 0 and batch_idx % log_interval == 0:
                print(f"Epoch {epoch+1}/{epochs} Batch {batch_idx}/{len(train_loader)}\tLoss: {loss.item():.4f}")
        epoch_number = epoch + 1
        if epoch_end_callback is not None:
            epoch_end_callback(model, epoch_number)
        print(f"Finished epoch {epoch_number}/{epochs}")
    return model
