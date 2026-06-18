"""Command line interface.

This script ties together the model definitions, dataset loaders, training
code, ONNX export, quantisation and C code generation into a single
entry point. 
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
from dataclasses import replace
from types import ModuleType

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from converters import convert_to_c
from converters.lstm_linear import replace_lstm_with_linear
from pruning import (
    PruningConfig,
    apply_pruning,
    finalize_pruning,
    parse_excluded_layers,
    pruning_sparsity,
    should_prune_epoch,
    validate_pruning_config,
)
from trainers import train_model, evaluate_classification, evaluate_regression
from converters.export_onnx import export_to_onnx


def str2bool(v):
    """Convert a string to boolean for CLI arguments."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def save_one_test_sample_to_bin(test_loader, output_path: str, dtype: str = "float32"):
    batch = next(iter(test_loader))

    if isinstance(batch, (tuple, list)):
        inputs = batch[0]
    elif isinstance(batch, dict):
        for key in ("input", "inputs", "x", "data"):
            if key in batch:
                inputs = batch[key]
                break
        else:
            inputs = next(iter(batch.values()))
    else:
        inputs = batch

    if isinstance(inputs, torch.Tensor):
        sample = inputs[:1].detach().cpu().contiguous()
        array = sample.numpy()
    else:
        array = np.asarray(inputs)
        if array.shape:
            array = array[:1]

    array = np.ascontiguousarray(array.astype(dtype, copy=False))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    array.tofile(output_path)
    return output_path, tuple(array.shape), str(array.dtype)


def _load_module_from_file(path: str, module_name: str) -> ModuleType:
    if not path:
        raise ValueError(f"No path was provided for {module_name}.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Python file does not exist: {path}")

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python file {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call_with_supported_kwargs(function, **candidate_kwargs):
    """Call ``function`` with only the keyword arguments it accepts."""
    signature = inspect.signature(function)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return function(**candidate_kwargs)

    supported_kwargs = {
        name: value
        for name, value in candidate_kwargs.items()
        if name in signature.parameters
    }
    return function(**supported_kwargs)


def _first_input_batch(data_loader) -> torch.Tensor:
    batch = next(iter(data_loader))
    if isinstance(batch, (tuple, list)):
        inputs = batch[0]
    elif isinstance(batch, dict):
        for key in ("input", "inputs", "x", "data"):
            if key in batch:
                inputs = batch[key]
                break
        else:
            inputs = next(iter(batch.values()))
    else:
        inputs = batch

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.as_tensor(inputs)
    return inputs[:1].detach().cpu()


def _load_model_and_dummy_input(args: argparse.Namespace, fallback_loader=None) -> tuple[nn.Module, torch.Tensor | None, str]:
    if args.model_file:
        model_module = _load_module_from_file(args.model_file, 'custom_model')
        model_name = os.path.splitext(os.path.basename(args.model_file))[0]
    else:
        raise ValueError("You must specify --model-file unless --onnx-path is used.")

    if not hasattr(model_module, 'get_model'):
        raise AttributeError(f"Model source for {model_name} must define a get_model() function")

    model = _call_with_supported_kwargs(model_module.get_model, seq_len=args.seq_len)
    dummy_input = None
    if hasattr(model_module, 'get_dummy_input'):
        dummy_input = _call_with_supported_kwargs(
            model_module.get_dummy_input,
            seq_len=args.seq_len,
        )
    elif fallback_loader is not None:
        dummy_input = _first_input_batch(fallback_loader)

    if dummy_input is not None and not isinstance(dummy_input, torch.Tensor):
        dummy_input = torch.as_tensor(dummy_input)

    return model, dummy_input, model_name


def _load_dataloaders(args: argparse.Namespace):
    if not args.dataset_file:
        raise ValueError("You must specify --dataset-file for the PyTorch pipeline or for static ONNX quantization.")

    dataset_module = _load_module_from_file(args.dataset_file, 'custom_dataset')
    dataset_name = os.path.splitext(os.path.basename(args.dataset_file))[0]

    if not hasattr(dataset_module, 'get_dataloaders'):
        raise AttributeError(f"Dataset source for {dataset_name} must define a get_dataloaders() function")

    common_kwargs = {
        'root': args.data_path,
        'train_batch_size': args.batch_size,
        'test_batch_size': args.test_batch_size,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
    }
    train_loader, test_loader = _call_with_supported_kwargs(dataset_module.get_dataloaders, **common_kwargs)

    if hasattr(dataset_module, 'get_calibration_loader'):
        calib_loader = _call_with_supported_kwargs(
            dataset_module.get_calibration_loader,
            batch_size=1,
            seq_len=args.seq_len,
        )
    else:
        calib_loader = train_loader

    return train_loader, test_loader, calib_loader, dataset_name


def _build_pruning_config(args: argparse.Namespace) -> PruningConfig:
    return PruningConfig(
        mode=args.prune,
        amount=args.prune_amount,
        method=args.prune_method,
        layers=args.prune_layers,
        scope=args.prune_scope,
        prune_bias=args.prune_bias,
        structured_dim=args.prune_structured_dim,
        structured_norm=args.prune_structured_norm,
        start_epoch=args.prune_start_epoch,
        frequency=args.prune_frequency,
        finalize=args.prune_finalize,
        exclude=parse_excluded_layers(args.prune_exclude_layers),
    )


def _print_pruning_summary(model: nn.Module, config: PruningConfig, label: str) -> None:
    zeros, total, sparsity = pruning_sparsity(model, config)
    if total:
        print(f"{label}: {zeros}/{total} selected parameters are zero ({sparsity * 100:.2f}% sparsity).")
    else:
        print(f"{label}: no selected prunable parameters found.")


def _select_device(args: argparse.Namespace) -> str:
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device, but CUDA is not available.")
    return device


def _resolve_checkpoint_path(args: argparse.Namespace, model_name: str) -> str:
    raw_value = args.weights_path
    if raw_value is None or str(raw_value).strip() == '' or str(raw_value).lower() in {'auto', 'true', 'yes', '1'}:
        return os.path.join(args.output_dir, f"{model_name}.pt")
    if str(raw_value).lower() in {'false', 'no', '0', 'none'}:
        return os.path.join(args.output_dir, f"{model_name}.pt")

    candidate = str(raw_value)
    if not candidate.endswith('.pt'):
        candidate = f"{candidate}.pt"
    if not os.path.isabs(candidate) and os.path.dirname(candidate) == '':
        candidate = os.path.join(args.output_dir, candidate)
    return candidate


def _checkpoint_stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _append_suffix_once(stem: str, suffix: str) -> str:
    return stem if stem.endswith(suffix) else f"{stem}{suffix}"


def _make_during_pruning_config(args: argparse.Namespace, config: PruningConfig) -> PruningConfig:
    """Optionally convert a desired final sparsity into a per-step amount."""
    if config.mode != 'during' or not config.enabled:
        return config

    pruning_epochs = [
        epoch
        for epoch in range(1, args.epochs + 1)
        if should_prune_epoch(epoch, config)
    ]
    num_pruning_steps = len(pruning_epochs)
    if num_pruning_steps == 0:
        print(
            "Pruning was requested during training, but no epoch matches the pruning schedule. "
            "Check --epochs, --prune-start-epoch, and --prune-frequency."
        )
        return config

    if num_pruning_steps == 1:
        print(
            f"During-training pruning will run {num_pruning_steps} time(s) "
            f"with amount {config.amount * 100:.2f}% each time."
        )
        return config

    # Treat --prune-amount as the desired final sparsity over all scheduled pruning steps.
    target_sparsity = config.amount
    per_step_amount = 1.0 - ((1.0 - target_sparsity) ** (1.0 / num_pruning_steps))
    print(
        f"Target final pruning sparsity: {target_sparsity * 100:.2f}% "
        f"over {num_pruning_steps} pruning steps. "
        f"Using per-step amount: {per_step_amount * 100:.2f}%."
    )
    return replace(config, amount=per_step_amount)


def _quantize_onnx_if_requested(
    args: argparse.Namespace,
    onnx_path: str,
    artifact_name: str,
    calib_loader=None,
) -> str:
    if args.quantize == 'none':
        return onnx_path

    from quantization import quantize_model

    quantized_path = os.path.join(args.output_dir, f"{artifact_name}_quant.onnx")
    if args.quantize == 'static':
        if calib_loader is None:
            raise ValueError("--quantize static requires --dataset-file so calibration data can be loaded.")
        quantize_model(
            onnx_path,
            quantized_path,
            calibration_loader=calib_loader,
            activation_type=args.act_type,
            weight_type=args.weight_type,
        )
    else:
        quantize_model(
            onnx_path,
            quantized_path,
            calibration_loader=None,
            activation_type=args.act_type,
            weight_type=args.weight_type,
        )
    print(f"Quantized model saved to {quantized_path}")
    return quantized_path


def _convert_onnx_to_c(args: argparse.Namespace, onnx_to_use: str, test_loader=None, dataset_name: str | None = None) -> None:
    onnx_name = os.path.splitext(os.path.basename(onnx_to_use))[0]
    c_output_path = args.c_output_path or os.path.join(args.output_dir, f"{onnx_name}.c")
    os.makedirs(os.path.dirname(c_output_path) or '.', exist_ok=True)

    convert_to_c(
        onnx_to_use,
        c_output_path,
        onnx2c_executable=args.onnx2c_executable,
        quant=args.quant_c,
    )

    if test_loader is None or dataset_name is None:
        print("Skipping input.bin generation because --dataset-file was not provided.")
        return

    input_bin_path = os.path.join(args.output_dir, f"{dataset_name}_input.bin")
    written_path, input_shape, input_dtype = save_one_test_sample_to_bin(test_loader, input_bin_path)
    print(
        f"Saved one test input sample to {written_path} "
        f"with shape {input_shape} and dtype {input_dtype}"
    )
    return input_bin_path


def _validate_args(args: argparse.Namespace) -> None:
    if args.prune == 'during' and not args.train:
        raise ValueError("--prune during requires --train true. Use --prune after when loading existing weights.")
    if args.epochs < 1 and not args.onnx_path:
        raise ValueError("--epochs must be at least 1")
    if getattr(args, 'batch_size', 1) < 1:
        raise ValueError("--batch-size must be at least 1")
    if getattr(args, 'test_batch_size', 1) < 1:
        raise ValueError("--test-batch-size must be at least 1")

    if args.onnx_path:
        if not os.path.exists(args.onnx_path):
            raise FileNotFoundError(f"ONNX file does not exist: {args.onnx_path}")
        if args.prune != 'none':
            raise ValueError("--onnx-path cannot be combined with --prune. Pruning must happen before ONNX export.")
        if args.quantize == 'none' and not args.export_c:
            raise ValueError("--onnx-path was provided, but no action was requested. Use --export-c and/or --quantize.")
        if args.quantize == 'static' and not getattr(args, 'dataset_file', None):
            raise ValueError("--quantize static with --onnx-path requires --dataset-file for calibration data.")
    else:
        if not args.model_file:
            raise ValueError("You must specify --model-file unless --onnx-path is used.")
        if not args.dataset_file:
            raise ValueError("You must specify --dataset-file unless --onnx-path is used.")

# ==========================================
# GUI Integration APIs
# ==========================================

def run_training(args: argparse.Namespace) -> str:
    """Trains a PyTorch model and saves the .pt file."""
    os.makedirs(args.output_dir, exist_ok=True)
    device = _select_device(args)

    # Legacy GUI backwards compatibility
    if hasattr(args, 'load_model') and args.load_model:
        args.weights_path = args.load_model
    if not hasattr(args, 'weights_path'):
        args.weights_path = None

    train_loader, test_loader, _, _ = _load_dataloaders(args)
    model, _, model_name = _load_model_and_dummy_input(args, fallback_loader=test_loader)

    if args.classification:
        criterion: nn.Module = nn.CrossEntropyLoss()
        eval_fn = evaluate_classification
    else:
        criterion = nn.MSELoss()
        eval_fn = evaluate_regression

    # Load existing weights instead of training if requested
    if not args.train and args.weights_path:
        if not os.path.exists(args.weights_path):
            raise FileNotFoundError(f"Weights file {args.weights_path} does not exist.")
        model.load_state_dict(torch.load(args.weights_path, map_location='cpu'))
        print(f"Loaded existing weights from {args.weights_path}")
        return args.weights_path

    print(f"Starting training with model: {args.model_file}")
    pruning_config = _build_pruning_config(args)
    pruning_config_for_step = _make_during_pruning_config(args, pruning_config)
    pruning_was_applied = False

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    pruning_config_for_step = _make_during_pruning_config(args, pruning_config)

    def pruning_callback(current_model: nn.Module, epoch_number: int) -> None:
        nonlocal pruning_was_applied
        if should_prune_epoch(epoch_number, pruning_config):
            apply_pruning(current_model, pruning_config_for_step)
            pruning_was_applied = True
            _print_pruning_summary(current_model, pruning_config, f"Pruning after epoch {epoch_number}")

    callback = pruning_callback if pruning_config.mode == 'during' else None
    model = train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        epochs=args.epochs,
        device=device,
        log_interval=100,
        epoch_end_callback=callback,
    )

    if pruning_was_applied and pruning_config.finalize:
        finalize_pruning(model, pruning_config)
        _print_pruning_summary(model, pruning_config, "Finalized pruning")

    artifact_name = _append_suffix_once(model_name, '_pruned') if pruning_was_applied else model_name
    output_path = os.path.join(args.output_dir, f"{artifact_name}.pt")
    
    torch.save(model.state_dict(), output_path)
    print(f"Training complete. Saved to {output_path}")

    metric = eval_fn(model, test_loader, device=device)
    metric_str = f"Test accuracy: {metric * 100:.2f}%" if args.classification else f"Test MSE: {metric:.6f}"
    print(metric_str)

    return output_path


def run_pruning(pt_path: str, model_file: str, dataset_file: str, args: argparse.Namespace) -> str:
    """Applies pruning to an existing .pt file and saves a new .pt file."""
    args.model_file = model_file
    args.dataset_file = dataset_file
    args.weights_path = pt_path
    
    os.makedirs(args.output_dir, exist_ok=True)

    _, test_loader, _, _ = _load_dataloaders(args)
    model, _, _ = _load_model_and_dummy_input(args, fallback_loader=test_loader)

    print(f"Starting pruning on weights: {pt_path}")
    model.load_state_dict(torch.load(pt_path, map_location='cpu'))

    pruning_config = _build_pruning_config(args)
    if pruning_config.mode == 'none':
        print("Warning: Pruning mode was set to 'none'. Overriding to 'after' for the pruning stage.")
        pruning_config.mode = 'after'
        pruning_config.enabled = True

    apply_pruning(model, pruning_config)
    _print_pruning_summary(model, pruning_config, "Pruning applied")

    if pruning_config.finalize:
        finalize_pruning(model, pruning_config)
        _print_pruning_summary(model, pruning_config, "Finalized pruning")

    checkpoint_stem = _checkpoint_stem(pt_path)
    artifact_name = _append_suffix_once(checkpoint_stem, '_pruned')
    output_path = os.path.join(args.output_dir, f"{artifact_name}.pt")
    
    torch.save(model.state_dict(), output_path)
    print(f"Pruning complete. Saved to {output_path}")
    return output_path


def run_export_onnx(pt_path: str, model_file: str, dataset_file: str, args: argparse.Namespace) -> str:
    """Converts a trained/pruned .pt model into ONNX format."""
    args.model_file = model_file
    args.dataset_file = dataset_file
    args.weights_path = pt_path
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = _select_device(args)

    test_loader = None
    if dataset_file:
        _, test_loader, _, _ = _load_dataloaders(args)

    model, dummy_input, _ = _load_model_and_dummy_input(args, fallback_loader=test_loader)
    
    print(f"Exporting {pt_path} to ONNX format...")
    model.load_state_dict(torch.load(pt_path, map_location='cpu'))
    model = replace_lstm_with_linear(model).to(device)

    if dummy_input is None:
        raise ValueError("Could not infer dummy input for ONNX export. Provide get_dummy_input() or a dataset.")

    checkpoint_stem = _checkpoint_stem(pt_path)
    output_path = os.path.join(args.output_dir, f"{checkpoint_stem}.onnx")
    export_to_onnx(model, dummy_input.to(device), output_path, opset_version=14)
    
    print(f"ONNX export complete. Saved to {output_path}")
    return output_path


def run_quantization(onnx_path: str, quant_type: str, dataset_file: str, args: argparse.Namespace) -> str:
    """Quantizes an ONNX model to either static or dynamic int8 format."""
    args.onnx_path = onnx_path
    args.quantize = quant_type
    args.dataset_file = dataset_file

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Quantizing {onnx_path} using {quant_type} method...")

    calib_loader = None
    if quant_type == 'static':
        if not dataset_file:
            raise ValueError("--quantize static requires a dataset file for calibration data.")
        _, _, calib_loader, _ = _load_dataloaders(args)

    artifact_name = _checkpoint_stem(onnx_path)
    if artifact_name.endswith('_quant'):
        artifact_name = artifact_name[:-6]

    output_path = _quantize_onnx_if_requested(args, onnx_path, artifact_name, calib_loader)
    return output_path

def run_build(c_path, model_name, input_path):
    return f"source <vivado path> \n make add-model MODEL_C={c_path} MODEL_NAME={model_name.upper()} INPUT_BIN={input_path} \n make all \n make deploy \n make host"

def run_convert_c(onnx_path: str, model_file: str, dataset_file: str, args: argparse.Namespace) -> str:
    """Converts an ONNX representation into C code utilizing onnx2c."""
    args.onnx_path = onnx_path
    args.model_file = model_file
    args.dataset_file = dataset_file

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Converting {onnx_path} to C implementation...")

    test_loader = None
    dataset_name = None
    if dataset_file:
        _, test_loader, _, dataset_name = _load_dataloaders(args)

    input_bin_path =_convert_onnx_to_c(args, onnx_path, test_loader=test_loader, dataset_name=dataset_name)

    onnx_name = os.path.splitext(os.path.basename(onnx_path))[0]
    output_path = getattr(args, 'c_output_path', None) or os.path.join(args.output_dir, f"{onnx_name}.c")
    
    print(f"C conversion complete. Saved to {output_path}")
    output_path = os.path.abspath(output_path)
    input_bin_path = os.path.abspath(input_bin_path)
    commands = run_build(output_path, onnx_name, input_bin_path)
    print("Follow README of backend repository or run below commands in command line or terminal in the specified order to run hardware codes, inside backend directory:")
    print(commands)
    return {'out': output_path, 'name': onnx_name.upper(), 'input': input_bin_path}


def run_pipeline(args: argparse.Namespace) -> None:
    """
    Executes the entire end-to-end pipeline at once. 
    This is used by the legacy CLI and the original single-page GUI worker.
    """
    _validate_args(args)
    os.makedirs(args.output_dir, exist_ok=True)

    device = _select_device(args)

    train_loader = test_loader = calib_loader = None
    dataset_name = None
    if args.dataset_file:
        train_loader, test_loader, calib_loader, dataset_name = _load_dataloaders(args)

    # ONNX-only path
    if args.onnx_path:
        print(f"Using existing ONNX model: {args.onnx_path}")
        artifact_name = os.path.splitext(os.path.basename(args.onnx_path))[0]
        onnx_to_use = _quantize_onnx_if_requested(args, args.onnx_path, artifact_name, calib_loader)
        if args.export_c:
            _convert_onnx_to_c(args, onnx_to_use, test_loader=test_loader, dataset_name=dataset_name)
        print("Pipeline completed successfully.")
        return

    pruning_config = _build_pruning_config(args)
    validate_pruning_config(pruning_config)
    if pruning_config.mode == 'during' and not args.train:
        raise ValueError("--prune during requires --train true. Use --prune after when loading existing weights.")

    model, dummy_input, model_name = _load_model_and_dummy_input(args, fallback_loader=test_loader)
    if dummy_input is None:
        if test_loader is None:
            raise ValueError("Could not infer dummy input. Add get_dummy_input() to the model file or provide --dataset-file.")
        dummy_input = _first_input_batch(test_loader)

    if args.classification:
        criterion: nn.Module = nn.CrossEntropyLoss()
        eval_fn = evaluate_classification
    else:
        criterion = nn.MSELoss()
        eval_fn = evaluate_regression

    loaded_checkpoint_stem = None
    pruning_was_applied = False
    
    if getattr(args, 'weights_path', None) or getattr(args, 'load_model', None):
        target_path = getattr(args, 'weights_path', None) or args.load_model
        if target_path and target_path.lower() not in ('auto', 'true', 'yes', '1', 'false', 'no', '0', 'none'):
            if not os.path.exists(target_path):
                raise FileNotFoundError(
                    f"Weights file {target_path} does not exist. "
                    "Train the model first, pass --weights-path"
                )
            state_dict = torch.load(target_path, map_location='cpu')
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Error loading state dict from {target_path}: {e}\n"
                    "If this checkpoint was saved with --prune-finalize false, save/load a finalized checkpoint instead."
                ) from e
            loaded_checkpoint_stem = _checkpoint_stem(target_path)
            print(f"Loaded model weights from {target_path}")
        
    if args.train:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        pruning_config_for_step = _make_during_pruning_config(args, pruning_config)

        def pruning_callback(current_model: nn.Module, epoch_number: int) -> None:
            nonlocal pruning_was_applied
            if should_prune_epoch(epoch_number, pruning_config):
                apply_pruning(current_model, pruning_config_for_step)
                pruning_was_applied = True
                _print_pruning_summary(current_model, pruning_config, f"Pruning after epoch {epoch_number}")

        callback = pruning_callback if pruning_config.mode == 'during' else None
        model = train_model(
            model,
            train_loader,
            criterion,
            optimizer,
            epochs=args.epochs,
            device=device,
            log_interval=100,
            epoch_end_callback=callback,
        )

    if pruning_config.mode == 'after':
        apply_pruning(model, pruning_config)
        pruning_was_applied = pruning_config.enabled
        _print_pruning_summary(model, pruning_config, "Pruning after training/loading")

    if pruning_was_applied and pruning_config.finalize:
        finalize_pruning(model, pruning_config)
        _print_pruning_summary(model, pruning_config, "Finalized pruning")
    elif pruning_was_applied:
        print(
            "Warning: --prune-finalize false leaves pruning masks/re-parameterisations in the state_dict. "
            "A fresh model may not load that checkpoint unless the same pruning hooks are recreated."
        )

    base_artifact_stem = loaded_checkpoint_stem or model_name
    artifact_name = _append_suffix_once(base_artifact_stem, '_pruned') if pruning_was_applied else base_artifact_stem

    if args.train or pruning_was_applied:
        final_weights_path = os.path.join(args.output_dir, f"{artifact_name}.pt")
        torch.save(model.state_dict(), final_weights_path)
        print(f"Saved PyTorch weights to {final_weights_path}")

    metric = eval_fn(model, test_loader, device=device)
    if args.classification:
        print(f"Test accuracy: {metric * 100:.2f}%")
    else:
        print(f"Test mean squared error: {metric:.6f}")

    model = replace_lstm_with_linear(model).to(device)

    onnx_path = os.path.join(args.output_dir, f"{artifact_name}.onnx")
    export_to_onnx(model, dummy_input.to(device), onnx_path)
    print(f"Exported ONNX model to {onnx_path}")

    onnx_to_use = _quantize_onnx_if_requested(args, onnx_path, artifact_name, calib_loader)

    if args.export_c:
        _convert_onnx_to_c(args, onnx_to_use, test_loader=test_loader, dataset_name=dataset_name)

    print("Pipeline completed successfully.")


# ==========================================
# CLI Entry Point
# ==========================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, evaluate, quantise and convert models.")

    # Custom model/dataset selection.
    parser.add_argument('--model-file', type=str,
                        help='Path to a Python file containing a custom model with a get_model() function.')
    parser.add_argument('--dataset-file', type=str,
                        help='Path to a Python file containing a custom dataset loader with a get_dataloaders() function.')
    parser.add_argument('--data-path', type=str, default='./data', help='Absolute path to the data directory')
    # Training control.
    parser.add_argument('--train', type=str2bool, default=True,
                        help='If false, skip training and load existing model weights.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size.')
    parser.add_argument('--test-batch-size', type=int, default=1000, help='Testing batch size.')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for Adam optimiser.')
    parser.add_argument('--output-dir', type=str, default='build', help='Directory to store outputs.')
    parser.add_argument('--device', type=str, default=None, help='Device on which to run training/evaluation/export, e.g. cpu or cuda.')
    parser.add_argument('--seq-len', type=int, default=12,
                        help='Sequence length for synthetic sine dataset and dummy input for LSTM.')
    parser.add_argument('--classification', type=str2bool, default=True,
                        help='Set true for classification, false for regression.')
    parser.add_argument('--weights-path', '--load-model', dest='weights_path', type=str, nargs='?', const='auto', default=None,
                        help='Checkpoint path to load. Alias: --load-model.')

    # PyTorch pruning controls.
    parser.add_argument('--prune', type=str, choices=['none', 'during', 'after'], default='none',
                        help='Apply PyTorch pruning: none, during training, or once after training/loading.')
    parser.add_argument('--prune-amount', type=float, default=0.0,
                        help='Fraction of selected weights/channels to prune, e.g. 0.3 for 30%%.')
    parser.add_argument('--prune-method', type=str,
                        choices=['l1_unstructured', 'random_unstructured', 'ln_structured', 'random_structured'],
                        default='l1_unstructured', help='PyTorch pruning method to use.')
    parser.add_argument('--prune-layers', type=str, choices=['linear', 'conv', 'both'], default='both',
                        help='Layer types to prune: Linear, Conv2d, or both.')
    parser.add_argument('--prune-scope', type=str, choices=['local', 'global'], default='local',
                        help='Local layer-wise pruning or global pruning across selected parameters. Global supports unstructured methods only.')
    parser.add_argument('--prune-bias', type=str2bool, default=False,
                        help='Whether to prune layer bias tensors as well as weights.')
    parser.add_argument('--prune-structured-dim', type=int, default=0,
                        help='Dimension for structured pruning. 0 prunes output channels/neurons.')
    parser.add_argument('--prune-structured-norm', type=float, default=2.0,
                        help='Norm used by ln_structured pruning.')
    parser.add_argument('--prune-start-epoch', type=int, default=1,
                        help='First epoch at which --prune during is applied.')
    parser.add_argument('--prune-frequency', type=int, default=1,
                        help='Epoch interval for --prune during.')
    parser.add_argument('--prune-finalize', type=str2bool, default=True,
                        help='Remove pruning re-parameterisations after pruning so zeros are stored as normal weights.')
    parser.add_argument('--prune-exclude-layers', type=str, default='',
                        help='Comma-separated exact module names to skip, e.g. fc2,l3.')

    # ONNX/quantization/C export controls.
    parser.add_argument('--quantize', type=str, choices=['none', 'static', 'dynamic'], default='none',
                        help='Apply quantisation to the ONNX model.')
    parser.add_argument('--export-c', action='store_true', default=False,
                        help='Convert the selected ONNX model to C using onnx2c.')
    parser.add_argument('--quant-c', type=str2bool, default=True,
                        help='Whether to request quantized helper generation in onnx2c.')
    parser.add_argument('--act-type', type=str, default='quint8', help='Activation quantization type.')
    parser.add_argument('--weight-type', type=str, default='qint8', help='Weight quantization type.')
    parser.add_argument('--onnx-path', type=str, default=None,
                        help='Path to an already exported ONNX model. If set, skip PyTorch model loading, training, evaluation, and ONNX export.')
    parser.add_argument('--c-output-path', type=str, default=None,
                        help='Optional output path for the generated C file. Defaults to <output-dir>/<onnx_stem>.c.')
    parser.add_argument('--onnx2c-executable', type=str, default='./onnx2c/build/onnx2c',
                        help='Path/name of the onnx2c executable.')

    return parser.parse_args()


def main() -> None:
    """Execution triggered by command line."""
    args = parse_args()
    run_pipeline(args)


if __name__ == '__main__':
    main()

