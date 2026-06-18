#test file to check fp32 and quantized models outputs

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Iterator

import numpy as np
import onnxruntime as ort
import torch


_ONNX_NUMPY_DTYPES: dict[str, np.dtype] = {
    "tensor(float)": np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(double)": np.dtype(np.float64),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(uint64)": np.dtype(np.uint64),
    "tensor(uint32)": np.dtype(np.uint32),
    "tensor(uint16)": np.dtype(np.uint16),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(bool)": np.dtype(np.bool_),
}


def _load_module_from_file(path: str, module_name: str) -> ModuleType:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Python file does not exist: {source}")

    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Python file: {source}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call_with_supported_kwargs(function, **candidate_kwargs):
    """Call a dataset helper with only the keyword arguments it accepts."""
    signature = inspect.signature(function)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return function(**candidate_kwargs)

    supported = {
        name: value
        for name, value in candidate_kwargs.items()
        if name in signature.parameters
    }
    return function(**supported)


def _parse_dataset_arg(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise argparse.ArgumentTypeError(
            f"Dataset argument must use KEY=VALUE syntax, received: {text!r}"
        )
    key, raw_value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Dataset argument key cannot be empty.")

    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return key, value


def _load_test_loader(args: argparse.Namespace) -> Iterable[Any]:
    module = _load_module_from_file(args.dataset_file, "comparison_dataset")
    if not hasattr(module, "get_dataloaders"):
        raise AttributeError(
            f"{args.dataset_file} must define get_dataloaders()."
        )

    custom_kwargs = dict(args.dataset_arg)
    loader_kwargs = {
        "train_batch_size": 1,
        "test_batch_size": 1,
        "batch_size": 1,
        "seq_len": args.seq_len,
        **custom_kwargs,
    }
    loaders = _call_with_supported_kwargs(module.get_dataloaders, **loader_kwargs)
    if not isinstance(loaders, (tuple, list)) or len(loaders) < 2:
        raise TypeError(
            "get_dataloaders() must return at least (train_loader, test_loader)."
        )
    return loaders[1]


def _extract_inputs_and_targets(batch: Any) -> tuple[Any, Any | None]:
    if isinstance(batch, (tuple, list)):
        if not batch:
            raise ValueError("The dataset yielded an empty tuple/list.")
        inputs = batch[0]
        targets = batch[1] if len(batch) > 1 else None
        return inputs, targets

    if isinstance(batch, dict):
        input_keys = ("input", "inputs", "x", "data")
        target_keys = ("target", "targets", "y", "label", "labels")

        input_key = next((key for key in input_keys if key in batch), None)
        if input_key is None:
            input_key = next(iter(batch), None)
        if input_key is None:
            raise ValueError("The dataset yielded an empty dictionary.")

        target_key = next((key for key in target_keys if key in batch), None)
        return batch[input_key], batch[target_key] if target_key else None

    return batch, None


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _iter_samples(loader: Iterable[Any]) -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
    """Yield individual samples while retaining a batch dimension of one."""
    for batch in loader:
        inputs, targets = _extract_inputs_and_targets(batch)
        input_array = _to_numpy(inputs)

        if input_array.ndim == 0:
            input_array = input_array.reshape(1, 1)
        elif input_array.shape[0] == 0:
            continue

        target_array = None if targets is None else _to_numpy(targets)
        batch_size = int(input_array.shape[0])

        for index in range(batch_size):
            sample = np.ascontiguousarray(input_array[index : index + 1])
            target = None
            if target_array is not None:
                if target_array.ndim == 0:
                    target = target_array.reshape(1)
                elif target_array.shape[0] == batch_size:
                    target = np.asarray(target_array[index])
                else:
                    target = np.asarray(target_array)
            yield sample, target


def _session(path: str) -> ort.InferenceSession:
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {model_path}")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _validate_single_input(session: ort.InferenceSession, label: str):
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(
            f"{label} model has {len(inputs)} inputs. This tester currently supports "
            "the framework's single-input models."
        )
    return inputs[0]


def _cast_for_onnx(sample: np.ndarray, onnx_type: str) -> np.ndarray:
    dtype = _ONNX_NUMPY_DTYPES.get(onnx_type)
    if dtype is None:
        raise TypeError(f"Unsupported ONNX input type: {onnx_type}")
    return np.ascontiguousarray(sample.astype(dtype, copy=False))


def _shape_compatible(sample_shape: tuple[int, ...], model_shape: list[Any]) -> bool:
    if len(sample_shape) != len(model_shape):
        return False
    for actual, expected in zip(sample_shape, model_shape):
        if isinstance(expected, int) and expected > 0 and actual != expected:
            return False
    return True


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponents = np.exp(shifted)
    denominator = np.sum(exponents)
    if denominator == 0 or not np.isfinite(denominator):
        return np.full_like(values, np.nan, dtype=np.float64)
    return exponents / denominator


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.astype(np.float64, copy=False).reshape(-1)
    right_flat = right.astype(np.float64, copy=False).reshape(-1)
    denominator = np.linalg.norm(left_flat) * np.linalg.norm(right_flat)
    if denominator == 0:
        return 1.0 if np.array_equal(left_flat, right_flat) else 0.0
    return float(np.dot(left_flat, right_flat) / denominator)


def _format_array(values: np.ndarray, limit: int) -> str:
    flat = np.asarray(values).reshape(-1)
    shown = flat[:limit]
    text = np.array2string(
        shown,
        precision=6,
        separator=", ",
        suppress_small=False,
        max_line_width=160,
    )
    if flat.size > limit:
        text = text[:-1] + f", ... ({flat.size} values)]"
    return text


def _target_scalar(target: np.ndarray | None) -> int | float | None:
    if target is None:
        return None
    flat = np.asarray(target).reshape(-1)
    if flat.size != 1:
        return None
    value = flat[0]
    if np.issubdtype(flat.dtype, np.integer):
        return int(value)
    return float(value)


def _infer_task(
    requested_task: str,
    output: np.ndarray,
    target: np.ndarray | None,
) -> str:
    if requested_task != "auto":
        return requested_task

    flat_output = np.asarray(output).reshape(-1)
    if target is not None:
        target_array = np.asarray(target)
        if (
            flat_output.size > 1
            and target_array.size == 1
            and np.issubdtype(target_array.dtype, np.integer)
        ):
            return "classification"
    return "regression"


def _file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024.0 * 1024.0)


def compare_models(args: argparse.Namespace) -> dict[str, Any]:
    fp32_session = _session(args.fp32_model)
    quant_session = _session(args.quantized_model)

    fp32_input = _validate_single_input(fp32_session, "FP32")
    quant_input = _validate_single_input(quant_session, "Quantized")

    if fp32_input.type != quant_input.type:
        raise TypeError(
            "The models expect different input types: "
            f"FP32={fp32_input.type}, quantized={quant_input.type}."
        )

    loader = _load_test_loader(args)

    all_fp32: list[np.ndarray] = []
    all_quant: list[np.ndarray] = []
    sample_metrics: list[dict[str, Any]] = []
    cosine_scores: list[float] = []

    classification_records: list[tuple[int | None, int, int]] = []
    regression_records: list[tuple[np.ndarray | None, np.ndarray, np.ndarray]] = []
    resolved_task: str | None = None

    for sample_index, (sample, target) in enumerate(_iter_samples(loader)):
        if sample_index >= args.num_samples:
            break

        sample = _cast_for_onnx(sample, fp32_input.type)
        if not _shape_compatible(tuple(sample.shape), fp32_input.shape):
            raise ValueError(
                f"Sample shape {tuple(sample.shape)} is incompatible with the FP32 "
                f"model input shape {fp32_input.shape}."
            )
        if not _shape_compatible(tuple(sample.shape), quant_input.shape):
            raise ValueError(
                f"Sample shape {tuple(sample.shape)} is incompatible with the quantized "
                f"model input shape {quant_input.shape}."
            )

        fp32_outputs = fp32_session.run(None, {fp32_input.name: sample})
        quant_outputs = quant_session.run(None, {quant_input.name: sample})
        if len(fp32_outputs) != len(quant_outputs):
            raise ValueError(
                "The models expose different output counts: "
                f"FP32={len(fp32_outputs)}, quantized={len(quant_outputs)}."
            )
        if not fp32_outputs:
            raise ValueError("The ONNX models produced no outputs.")

        # The current framework exports one output. Concatenate if a custom model
        # happens to expose several outputs so the numerical comparison remains useful.
        fp32_output = np.concatenate([np.asarray(x).reshape(-1) for x in fp32_outputs])
        quant_output = np.concatenate([np.asarray(x).reshape(-1) for x in quant_outputs])
        if fp32_output.shape != quant_output.shape:
            raise ValueError(
                "The models produced different flattened output shapes: "
                f"FP32={fp32_output.shape}, quantized={quant_output.shape}."
            )

        fp32_float = fp32_output.astype(np.float64, copy=False)
        quant_float = quant_output.astype(np.float64, copy=False)
        difference = quant_float - fp32_float
        absolute_difference = np.abs(difference)
        cosine = _cosine_similarity(fp32_float, quant_float)

        if resolved_task is None:
            resolved_task = _infer_task(args.task, fp32_output, target)

        within_tolerance = bool(
            np.allclose(fp32_float, quant_float, rtol=args.rtol, atol=args.atol)
        )

        record: dict[str, Any] = {
            "index": sample_index,
            "target": None if target is None else np.asarray(target).reshape(-1).tolist(),
            "fp32_output": fp32_float.tolist(),
            "quantized_output": quant_float.tolist(),
            "mean_absolute_difference": float(np.mean(absolute_difference)),
            "max_absolute_difference": float(np.max(absolute_difference)),
            "cosine_similarity": cosine,
            "within_tolerance": within_tolerance,
        }

        if resolved_task == "classification":
            fp32_class = int(np.argmax(fp32_float))
            quant_class = int(np.argmax(quant_float))
            target_value = _target_scalar(target)
            label = int(target_value) if isinstance(target_value, int) else None
            classification_records.append((label, fp32_class, quant_class))

            fp32_probability = float(np.max(_softmax(fp32_float)))
            quant_probability = float(np.max(_softmax(quant_float)))
            record.update(
                {
                    "fp32_prediction": fp32_class,
                    "quantized_prediction": quant_class,
                    "prediction_agreement": fp32_class == quant_class,
                    "fp32_top_probability": fp32_probability,
                    "quantized_top_probability": quant_probability,
                }
            )
        else:
            target_float = None
            if target is not None:
                target_flat = np.asarray(target, dtype=np.float64).reshape(-1)
                if target_flat.size == fp32_float.size:
                    target_float = target_flat
                elif target_flat.size == 1 and fp32_float.size == 1:
                    target_float = target_flat
            regression_records.append((target_float, fp32_float, quant_float))

        all_fp32.append(fp32_float)
        all_quant.append(quant_float)
        cosine_scores.append(cosine)
        sample_metrics.append(record)

        if sample_index < args.show_samples:
            target_text = "none" if target is None else _format_array(target, args.max_output_values)
            print(f"\nSample {sample_index}")
            print(f"  target:           {target_text}")
            print(
                "  fp32 output:      "
                f"{_format_array(fp32_float, args.max_output_values)}"
            )
            print(
                "  quantized output: "
                f"{_format_array(quant_float, args.max_output_values)}"
            )
            if resolved_task == "classification":
                print(
                    "  prediction:       "
                    f"fp32={record['fp32_prediction']}, "
                    f"quantized={record['quantized_prediction']}, "
                    f"agree={record['prediction_agreement']}"
                )
            print(
                "  difference:       "
                f"mean_abs={record['mean_absolute_difference']:.8g}, "
                f"max_abs={record['max_absolute_difference']:.8g}, "
                f"cosine={record['cosine_similarity']:.8f}, "
                f"within_tolerance={record['within_tolerance']}"
            )

    if not all_fp32:
        raise RuntimeError("The test loader did not yield any samples.")

    fp32_all = np.concatenate(all_fp32)
    quant_all = np.concatenate(all_quant)
    diff_all = quant_all - fp32_all
    abs_diff_all = np.abs(diff_all)
    denominator = float(np.linalg.norm(fp32_all))
    relative_l2 = float(np.linalg.norm(diff_all) / (denominator + np.finfo(float).eps))

    summary: dict[str, Any] = {
        "task": resolved_task,
        "samples": len(all_fp32),
        "fp32_model": str(Path(args.fp32_model).resolve()),
        "quantized_model": str(Path(args.quantized_model).resolve()),
        "fp32_size_mb": _file_size_mb(args.fp32_model),
        "quantized_size_mb": _file_size_mb(args.quantized_model),
        "size_reduction_percent": 100.0
        * (1.0 - os.path.getsize(args.quantized_model) / os.path.getsize(args.fp32_model)),
        "output_mae": float(np.mean(abs_diff_all)),
        "output_rmse": float(math.sqrt(np.mean(np.square(diff_all)))),
        "output_max_absolute_difference": float(np.max(abs_diff_all)),
        "output_relative_l2_error": relative_l2,
        "mean_cosine_similarity": float(np.mean(cosine_scores)),
        "minimum_cosine_similarity": float(np.min(cosine_scores)),
        "atol": args.atol,
        "rtol": args.rtol,
        "samples_within_tolerance": sum(
            bool(record["within_tolerance"]) for record in sample_metrics
        ) / len(sample_metrics),
        "all_outputs_within_tolerance": all(
            bool(record["within_tolerance"]) for record in sample_metrics
        ),
    }

    if resolved_task == "classification":
        agreement_count = sum(fp32 == quant for _, fp32, quant in classification_records)
        summary["prediction_agreement"] = agreement_count / len(classification_records)

        records_with_labels = [record for record in classification_records if record[0] is not None]
        if records_with_labels:
            summary["fp32_accuracy"] = sum(
                label == fp32 for label, fp32, _ in records_with_labels
            ) / len(records_with_labels)
            summary["quantized_accuracy"] = sum(
                label == quant for label, _, quant in records_with_labels
            ) / len(records_with_labels)
    else:
        records_with_targets = [record for record in regression_records if record[0] is not None]
        if records_with_targets:
            targets = np.concatenate([record[0] for record in records_with_targets])
            fp32_predictions = np.concatenate([record[1] for record in records_with_targets])
            quant_predictions = np.concatenate([record[2] for record in records_with_targets])
            summary["fp32_target_mae"] = float(np.mean(np.abs(fp32_predictions - targets)))
            summary["quantized_target_mae"] = float(np.mean(np.abs(quant_predictions - targets)))
            summary["fp32_target_mse"] = float(np.mean(np.square(fp32_predictions - targets)))
            summary["quantized_target_mse"] = float(np.mean(np.square(quant_predictions - targets)))

    results = {"summary": summary, "samples": sample_metrics}

    print("\n=== Comparison summary ===")
    print(f"Task:                         {summary['task']}")
    print(f"Samples:                      {summary['samples']}")
    print(f"FP32 model size:              {summary['fp32_size_mb']:.4f} MB")
    print(f"Quantized model size:         {summary['quantized_size_mb']:.4f} MB")
    print(f"Size reduction:               {summary['size_reduction_percent']:.2f}%")
    print(f"Output MAE:                   {summary['output_mae']:.8g}")
    print(f"Output RMSE:                  {summary['output_rmse']:.8g}")
    print(
        "Maximum absolute difference:   "
        f"{summary['output_max_absolute_difference']:.8g}"
    )
    print(f"Relative L2 error:            {summary['output_relative_l2_error']:.8g}")
    print(f"Mean cosine similarity:       {summary['mean_cosine_similarity']:.8f}")
    print(f"Minimum cosine similarity:    {summary['minimum_cosine_similarity']:.8f}")
    print(
        f"Samples within tolerance:      {summary['samples_within_tolerance'] * 100:.2f}% "
        f"(atol={summary['atol']:g}, rtol={summary['rtol']:g})"
    )

    if resolved_task == "classification":
        print(f"Prediction agreement:         {summary['prediction_agreement'] * 100:.2f}%")
        if "fp32_accuracy" in summary:
            print(f"FP32 accuracy:                {summary['fp32_accuracy'] * 100:.2f}%")
            print(f"Quantized accuracy:           {summary['quantized_accuracy'] * 100:.2f}%")
    else:
        if "fp32_target_mae" in summary:
            print(f"FP32 target MAE:              {summary['fp32_target_mae']:.8g}")
            print(f"Quantized target MAE:         {summary['quantized_target_mae']:.8g}")
            print(f"FP32 target MSE:              {summary['fp32_target_mse']:.8g}")
            print(f"Quantized target MSE:         {summary['quantized_target_mse']:.8g}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved detailed results to {output_path}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run shared dataset samples through FP32 and quantized ONNX models."
    )
    parser.add_argument(
        "--fp32-model",
        required=True,
        help="Path to the FP32 ONNX model, for example build/mnist_cnn.onnx.",
    )
    parser.add_argument(
        "--quantized-model",
        "--quant-model",
        dest="quantized_model",
        required=True,
        help="Path to the quantized ONNX model, for example build/mnist_cnn_quant.onnx.",
    )
    parser.add_argument(
        "--dataset-file",
        required=True,
        help="Dataset Python file exposing get_dataloaders().",
    )
    parser.add_argument(
        "--task",
        choices=("auto", "classification", "regression"),
        default="auto",
        help="Comparison mode. Auto detects classification from integer labels and vector outputs.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of test samples to run through both models.",
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=10,
        help="Number of per-sample comparisons to print.",
    )
    parser.add_argument(
        "--max-output-values",
        type=int,
        default=10,
        help="Maximum output values shown for each model per sample.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=12,
        help="Sequence length passed to compatible dataset loaders.",
    )
    parser.add_argument(
        "--dataset-arg",
        action="append",
        type=_parse_dataset_arg,
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra argument for get_dataloaders(); repeat as needed. Values use JSON "
            "syntax when possible, e.g. --dataset-arg download=false."
        ),
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance used by the per-sample numpy.allclose check.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance used by the per-sample numpy.allclose check.",
    )
    parser.add_argument(
        "--fail-on-output-mismatch",
        action="store_true",
        help="Exit with status 1 unless every compared output is within tolerance.",
    )
    parser.add_argument(
        "--fail-on-prediction-change",
        action="store_true",
        help="For classification, exit with status 1 if any predicted class changes.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for detailed machine-readable comparison results.",
    )

    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be at least 1.")
    if args.show_samples < 0:
        parser.error("--show-samples cannot be negative.")
    if args.max_output_values < 1:
        parser.error("--max-output-values must be at least 1.")
    return args


def main() -> None:
    args = parse_args()
    results = compare_models(args)
    summary = results["summary"]

    failed = False
    if args.fail_on_output_mismatch and not summary["all_outputs_within_tolerance"]:
        print("\nFAILED: at least one output exceeded the configured tolerance.")
        failed = True
    if (
        args.fail_on_prediction_change
        and summary["task"] == "classification"
        and summary.get("prediction_agreement", 1.0) < 1.0
    ):
        print("\nFAILED: at least one quantized prediction differs from FP32.")
        failed = True

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()