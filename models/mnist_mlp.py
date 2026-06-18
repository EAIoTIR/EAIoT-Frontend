"""Multi‑layer perceptron (MLP) model definitions.

This module defines a simple fully connected network for classification
tasks on flattened image inputs.  The architecture consists of a
series of linear layers separated by ReLU activations.  Both the
number of hidden layers and their sizes can be configured.
"""

import torch
import torch.nn as nn
from typing import Sequence

class MNC (nn.Module):
    def __init__(self, input_size):
        super(MNC, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.l1 = nn.Linear(input_size, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.l1(x)
        x = self.relu(x)

        x = self.l2(x)
        x = self.relu(x)

        x = self.l3(x)

        return x

def get_model(input_size: int = 28 * 28) -> nn.Module:
    """Return a new instance of ``MLP``.

    See ``MLP`` for parameter documentation.
    """
    return MNC(input_size=input_size)

def get_dummy_input(batch_size: int = 1, height: int = 28, width: int = 28) -> torch.Tensor:
    """Return a dummy input tensor for ONNX export.

    The MLP flattens its input internally, so a 4D tensor with shape
    ``(batch_size, 1, height, width)`` is returned for compatibility with
    the CNN/vision datasets.
    """
    return torch.randn(batch_size, 1, height, width)