"""Long short‑term memory (LSTM) model definitions.

This module defines a simple LSTM for univariate time‑series prediction.  By
default the network uses a single LSTM layer followed by a linear layer
to produce the final output.
"""

from __future__ import annotations

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """A basic LSTM model for sequence regression.

    Parameters
    ----------
    input_size: int
        Number of expected features in the input (e.g. 1 for a scalar time
        series).
    hidden_size: int
        Number of features in the hidden state.
    num_layers: int
        Number of recurrent layers.  Defaults to 1.
    output_size: int
        Size of the output; typically 1 for univariate prediction.
    dropout: float
        Dropout probability applied between LSTM layers; ignored if
        ``num_layers`` is 1.
    batch_first: bool
        If ``True``, the input and output tensors are provided as
        ``(batch, seq, feature)``.  Otherwise, ``(seq, batch, feature)``.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 100, num_layers: int = 1,
                 output_size: int = 1, dropout: float = 0.0, batch_first: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=batch_first, dropout=dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size) if batch_first else (seq_len, batch, input_size)
        out, _ = self.lstm(x)
        # Use the last time step's hidden state for prediction
        if self.batch_first:
            last_hidden = out[:, -1, :]
        else:
            last_hidden = out[-1, :, :]
        return self.linear(last_hidden)

def get_model(input_size: int = 1, hidden_size: int = 100, num_layers: int = 1,
              output_size: int = 1, dropout: float = 0.0, batch_first: bool = True) -> nn.Module:
    """Return a new instance of ``LSTMModel``.

    See ``LSTMModel`` for parameter documentation.
    """
    return LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                     output_size=output_size, dropout=dropout, batch_first=batch_first)

def get_dummy_input(batch_size: int = 1, seq_len: int = 12, input_size: int = 1, batch_first: bool = True) -> torch.Tensor:
    """Return a dummy sequence for ONNX export.

    Parameters
    ----------
    batch_size: int
        Number of sequences in the batch.
    seq_len: int
        Length of each input sequence.
    input_size: int
        Number of features per time step.
    batch_first: bool
        If ``True``, returns input with shape (batch, seq, input_size).  Otherwise
        (seq, batch, input_size).
    """
    import torch
    shape = (batch_size, seq_len, input_size) if batch_first else (seq_len, batch_size, input_size)
    return torch.randn(*shape)