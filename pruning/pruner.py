"""PyTorch pruning utilities for MLP and CNN layers.

The functions in this module wrap :mod:`torch.nn.utils.prune` so the main
pipeline can prune ``nn.Linear`` layers in MLPs and ``nn.Conv2d``/``nn.Linear``
layers in CNNs either during training or after training/loading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

PruneMode = Literal["none", "during", "after"]
PruneMethod = Literal[
    "l1_unstructured",
    "random_unstructured",
    "ln_structured",
    "random_structured",
]
PruneLayerSelection = Literal["linear", "conv", "both"]
PruneScope = Literal["local", "global"]


@dataclass
class PruningConfig:
    """Configuration for pruning supported layers in a model.

    Attributes
    ----------
    mode:
        ``"none"`` disables pruning. ``"during"`` applies pruning from the
        training loop. ``"after"`` applies pruning once after training or after
        loading an existing checkpoint.
    amount:
        Fraction of weights/channels to prune. Values between 0 and 1 are
        interpreted as a proportion by PyTorch's pruning API.
    method:
        PyTorch pruning method to apply.
    layers:
        Which layer classes to prune: fully connected ``Linear`` layers,
        convolutional ``Conv2d`` layers, or both.
    scope:
        ``"local"`` prunes each layer independently. ``"global"`` computes a
        single threshold across all selected parameters. Global pruning is only
        supported for unstructured methods.
    prune_bias:
        Whether to prune layer biases in addition to weights.
    structured_dim:
        Dimension used by structured pruning. For ``Conv2d`` weights, ``0``
        prunes output channels; for ``Linear`` weights, ``0`` prunes output
        neurons.
    structured_norm:
        Norm used by ``ln_structured`` pruning.
    start_epoch:
        First epoch number at which during-training pruning may run.
    frequency:
        Epoch interval for during-training pruning.
    finalize:
        Whether to remove pruning re-parameterisations after the selected
        pruning phase so zeroed weights are stored as normal parameters.
    exclude:
        Optional list of exact module names to skip.
    """

    mode: PruneMode = "none"
    amount: float = 0.0
    method: PruneMethod = "l1_unstructured"
    layers: PruneLayerSelection = "both"
    scope: PruneScope = "local"
    prune_bias: bool = False
    structured_dim: int = 0
    structured_norm: float = 2.0
    start_epoch: int = 1
    frequency: int = 1
    finalize: bool = True
    exclude: Sequence[str] = ()

    @property
    def enabled(self) -> bool:
        return self.mode != "none" and self.amount > 0


def parse_excluded_layers(raw_value: str | None) -> tuple[str, ...]:
    """Parse a comma-separated list of module names to exclude."""
    if not raw_value:
        return ()
    return tuple(item.strip() for item in raw_value.split(",") if item.strip())


def _selected_module_types(layers: PruneLayerSelection) -> tuple[type[nn.Module], ...]:
    if layers == "linear":
        return (nn.Linear,)
    if layers == "conv":
        return (nn.Conv2d,)
    if layers == "both":
        return (nn.Linear, nn.Conv2d)
    raise ValueError(f"Unsupported pruning layer selection: {layers}")


def _iter_prunable_modules(
    model: nn.Module,
    layers: PruneLayerSelection,
    exclude: Sequence[str] = (),
) -> Iterable[tuple[str, nn.Module]]:
    module_types = _selected_module_types(layers)
    excluded = set(exclude)
    for name, module in model.named_modules():
        if not name or name in excluded:
            continue
        if isinstance(module, module_types):
            yield name, module


def _parameters_to_prune(
    model: nn.Module,
    config: PruningConfig,
) -> list[tuple[nn.Module, str]]:
    parameters: list[tuple[nn.Module, str]] = []
    for _, module in _iter_prunable_modules(model, config.layers, config.exclude):
        if getattr(module, "weight", None) is not None:
            parameters.append((module, "weight"))
        if config.prune_bias and getattr(module, "bias", None) is not None:
            parameters.append((module, "bias"))
    return parameters


def validate_pruning_config(config: PruningConfig) -> None:
    """Validate pruning options early and fail with actionable messages."""
    if config.mode not in ("none", "during", "after"):
        raise ValueError("--prune must be one of: none, during, after")
    if config.amount < 0:
        raise ValueError("--prune-amount must be non-negative")
    if 0 < config.amount < 1:
        pass
    elif config.amount >= 1:
        raise ValueError(
            "--prune-amount currently expects a fraction between 0 and 1 "
            "for predictable CLI behaviour"
        )
    if config.frequency < 1:
        raise ValueError("--prune-frequency must be at least 1")
    if config.start_epoch < 1:
        raise ValueError("--prune-start-epoch must be at least 1")
    if config.scope == "global" and config.method in ("ln_structured", "random_structured"):
        raise ValueError("Global pruning is only supported for unstructured pruning methods")
    if config.prune_bias and config.method in ("ln_structured", "random_structured"):
        raise ValueError("Structured pruning of bias vectors is not supported by this helper")


def should_prune_epoch(epoch: int, config: PruningConfig) -> bool:
    """Return ``True`` when during-training pruning should run for ``epoch``."""
    if not config.enabled or config.mode != "during":
        return False
    return epoch >= config.start_epoch and (epoch - config.start_epoch) % config.frequency == 0


def apply_pruning(model: nn.Module, config: PruningConfig) -> nn.Module:
    """Apply the requested pruning method in place and return ``model``."""
    validate_pruning_config(config)
    if not config.enabled:
        return model

    parameters = _parameters_to_prune(model, config)
    if not parameters:
        print("Pruning requested, but no matching Linear/Conv2d layers were found.")
        return model

    if config.scope == "global":
        pruning_method = (
            prune.L1Unstructured
            if config.method == "l1_unstructured"
            else prune.RandomUnstructured
        )
        prune.global_unstructured(parameters, pruning_method=pruning_method, amount=config.amount)
    else:
        for module, parameter_name in parameters:
            if config.method == "l1_unstructured":
                prune.l1_unstructured(module, name=parameter_name, amount=config.amount)
            elif config.method == "random_unstructured":
                prune.random_unstructured(module, name=parameter_name, amount=config.amount)
            elif config.method == "ln_structured":
                prune.ln_structured(
                    module,
                    name=parameter_name,
                    amount=config.amount,
                    n=config.structured_norm,
                    dim=config.structured_dim,
                )
            elif config.method == "random_structured":
                prune.random_structured(
                    module,
                    name=parameter_name,
                    amount=config.amount,
                    dim=config.structured_dim,
                )
            else:
                raise ValueError(f"Unsupported pruning method: {config.method}")
    return model


def finalize_pruning(model: nn.Module, config: PruningConfig) -> nn.Module:
    """Make active pruning masks permanent by removing re-parameterisations."""
    for _, module in _iter_prunable_modules(model, config.layers, config.exclude):
        for parameter_name in ("weight", "bias"):
            if hasattr(module, f"{parameter_name}_orig"):
                prune.remove(module, parameter_name)
    return model


def pruning_sparsity(model: nn.Module, config: PruningConfig) -> tuple[int, int, float]:
    """Count zero values in selected prunable parameters.

    Returns
    -------
    (zero_count, total_count, sparsity)
        Sparsity is returned as a value in ``[0, 1]``. If there are no selected
        parameters, all values are zero.
    """
    zeros = 0
    total = 0
    with torch.no_grad():
        for _, module in _iter_prunable_modules(model, config.layers, config.exclude):
            names = ["weight"]
            if config.prune_bias:
                names.append("bias")
            for parameter_name in names:
                tensor = getattr(module, parameter_name, None)
                if tensor is None:
                    continue
                data = tensor.detach()
                zeros += int(torch.count_nonzero(data == 0).item())
                total += data.numel()
    sparsity = (zeros / total) if total else 0.0
    return zeros, total, sparsity
