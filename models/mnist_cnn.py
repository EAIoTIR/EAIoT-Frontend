"""Convolutional neural network (CNN) model definitions.

This module provides a simple CNN suitable for MNIST or other small
grayscale image classification tasks.  The architecture consists of two
convolutional layers followed by fully connected layers.  You can
customise the number of output classes via the ``num_classes`` argument.
"""

from __future__ import annotations

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """A small convolutional network for 28×28 single‑channel images.

    Parameters
    ----------
    in_channels: int, optional
        Number of input channels; defaults to 1 for grayscale images.
    num_classes: int, optional
        Number of output classes; defaults to 10.
    hidden_channels: tuple[int, int], optional
        Number of output channels for the first and second convolutional
        layers.  Defaults to (16, 32).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        # The output of the second conv + pooling layer has size (c2, 7, 7) if input is (1,28,28)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_model(in_channels: int = 1, num_classes: int = 10) -> nn.Module:
    """Return a new instance of ``SimpleCNN``.

    This function is called by the CLI to create the model.  See
    ``SimpleCNN`` for parameter documentation.
    """
    return SimpleCNN(in_channels=in_channels, num_classes=num_classes)

def get_dummy_input(batch_size: int = 1, in_channels: int = 1, height: int = 28, width: int = 28) -> torch.Tensor:
    """Return a dummy input tensor for ONNX export.

    Parameters
    ----------
    batch_size: int, optional
        Batch dimension to use for the dummy input.  Defaults to 1.
    in_channels: int, optional
        Number of channels.  Defaults to 1.
    height: int, optional
        Image height.  Defaults to 28.
    width: int, optional
        Image width.  Defaults to 28.

    Returns
    -------
    torch.Tensor
        A random tensor with the specified shape on CPU.
    """
    return torch.randn(batch_size, in_channels, height, width)