"""Utilities to convert nn.LSTM layers into stacks of linear layers.

This module is adapted from the original `lstmtolinear.py` example.  It
provides classes and functions that replace standard PyTorch LSTM
implementations with an equivalent collection of linear layers.  This
transformation is useful when exporting an LSTM to ONNX or when
applying quantisation, as it avoids unsupported operations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLinearLSTMCell(nn.Module):
    """Handles a single time‑step for a single direction in an LSTM.

    This class performs the equivalent computation of the standard
    PyTorch ``nn.LSTM`` cell but using linear layers only.  It is used
    internally by :class:`CustomLinearLSTM` and is not intended to be
    used directly.
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, proj_size: int = 0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        real_hidden_size = proj_size if proj_size > 0 else hidden_size
        self.linear_ih = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.linear_hh = nn.Linear(real_hidden_size, 4 * hidden_size, bias=bias)
        if proj_size > 0:
            self.linear_hr = nn.Linear(hidden_size, proj_size, bias=False)
        else:
            self.linear_hr = None

    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor, c_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gates = self.linear_ih(x_t) + self.linear_hh(h_t)
        i_t, f_t, g_t, o_t = gates.chunk(4, dim=1)
        i_t = torch.sigmoid(i_t)
        f_t = torch.sigmoid(f_t)
        g_t = torch.tanh(g_t)
        o_t = torch.sigmoid(o_t)
        c_t = (f_t * c_t) + (i_t * g_t)
        h_prime = o_t * torch.tanh(c_t)
        if self.proj_size > 0:
            h_t = self.linear_hr(h_prime)
        else:
            h_t = h_prime
        return h_t, c_t


class CustomLinearLSTM(nn.Module):
    """Replicates the behaviour of ``nn.LSTM`` using linear layers only.

    Parameters
    ----------
    input_size: int
        Number of input features per time step.
    hidden_size: int
        Number of features in the hidden state.
    num_layers: int
        Number of recurrent layers.
    bias: bool
        If ``True``, uses bias terms in the linear layers.
    batch_first: bool
        If ``True``, expects inputs as ``(batch, seq_len, input_size)``.
    dropout: float
        Dropout probability between LSTM layers.
    bidirectional: bool
        If ``True``, implements a bidirectional LSTM.
    proj_size: int
        If > 0, use a projection layer after the cell outputs.
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True,
                 batch_first: bool = False, dropout: float = 0.0, bidirectional: bool = False,
                 proj_size: int = 0) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.num_directions = 2 if bidirectional else 1
        self.real_hidden_size = proj_size if proj_size > 0 else hidden_size
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else self.real_hidden_size * self.num_directions
            self.cells.append(CustomLinearLSTMCell(layer_input_size, hidden_size, bias, proj_size))
            if bidirectional:
                self.cells.append(CustomLinearLSTMCell(layer_input_size, hidden_size, bias, proj_size))

    def forward(self, x: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()
        if hidden is None:
            num_states = self.num_layers * self.num_directions
            h_0 = torch.zeros(num_states, batch_size, self.real_hidden_size, device=x.device)
            c_0 = torch.zeros(num_states, batch_size, self.hidden_size, device=x.device)
        else:
            h_0, c_0 = hidden
        current_input = x
        h_n = []
        c_n = []
        for layer in range(self.num_layers):
            idx_fw = layer * self.num_directions
            cell_fw = self.cells[idx_fw]
            h_t_fw, c_t_fw = h_0[idx_fw], c_0[idx_fw]
            out_fw = []
            for t in range(seq_len):
                h_t_fw, c_t_fw = cell_fw(current_input[t], h_t_fw, c_t_fw)
                out_fw.append(h_t_fw)
            h_n_layer = [h_t_fw]
            c_n_layer = [c_t_fw]
            if self.bidirectional:
                idx_bw = idx_fw + 1
                cell_bw = self.cells[idx_bw]
                h_t_bw, c_t_bw = h_0[idx_bw], c_0[idx_bw]
                out_bw = []
                for t in range(seq_len - 1, -1, -1):
                    h_t_bw, c_t_bw = cell_bw(current_input[t], h_t_bw, c_t_bw)
                    out_bw.insert(0, h_t_bw)
                h_n_layer.append(h_t_bw)
                c_n_layer.append(c_t_bw)
                layer_output = [torch.cat([f, b], dim=-1) for f, b in zip(out_fw, out_bw)]
            else:
                layer_output = out_fw
            if self.dropout > 0 and layer < self.num_layers - 1:
                layer_output = [F.dropout(out, p=self.dropout, training=self.training) for out in layer_output]
            current_input = torch.stack(layer_output, dim=0)
            h_n.extend(h_n_layer)
            c_n.extend(c_n_layer)
        h_n = torch.stack(h_n, dim=0)
        c_n = torch.stack(c_n, dim=0)
        if self.batch_first:
            current_input = current_input.transpose(0, 1)
        return current_input, (h_n, c_n)

    def load_from_pytorch_lstm(self, pt_lstm: nn.LSTM) -> None:
        """Copy weights and biases from a standard ``nn.LSTM`` into this module."""
        state_dict = pt_lstm.state_dict()
        with torch.no_grad():
            for layer in range(self.num_layers):
                for direction in range(self.num_directions):
                    suffix = "_reverse" if direction == 1 else ""
                    cell_idx = layer * self.num_directions + direction
                    cell = self.cells[cell_idx]
                    cell.linear_ih.weight.copy_(state_dict[f'weight_ih_l{layer}{suffix}'])
                    if self.bias:
                        cell.linear_ih.bias.copy_(state_dict[f'bias_ih_l{layer}{suffix}'])
                    cell.linear_hh.weight.copy_(state_dict[f'weight_hh_l{layer}{suffix}'])
                    if self.bias:
                        cell.linear_hh.bias.copy_(state_dict[f'bias_hh_l{layer}{suffix}'])
                    if self.proj_size > 0:
                        cell.linear_hr.weight.copy_(state_dict[f'weight_hr_l{layer}{suffix}'])


def replace_lstm_with_linear(model: nn.Module) -> nn.Module:
    """Recursively replace all ``nn.LSTM`` modules in ``model`` with linear equivalents.

    After training a model that contains ``nn.LSTM`` layers it is sometimes
    advantageous to substitute them with stacks of linear layers before
    exporting to ONNX.  This function walks the module hierarchy and
    performs the substitution in place.

    Parameters
    ----------
    model: nn.Module
        A PyTorch model which may contain LSTM layers.

    Returns
    -------
    nn.Module
        The original model with LSTM layers replaced.  The operation is
        performed in place, so the return value is provided for convenience.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.LSTM):
            replacement = CustomLinearLSTM(
                input_size=module.input_size,
                hidden_size=module.hidden_size,
                num_layers=module.num_layers,
                bias=module.bias,
                batch_first=module.batch_first,
                dropout=module.dropout,
                bidirectional=module.bidirectional,
                proj_size=module.proj_size,
            )
            replacement.load_from_pytorch_lstm(module)
            setattr(model, name, replacement)
        else:
            replace_lstm_with_linear(module)
    return model