"""Helper functions for exporting PyTorch models to ONNX and applying quantisation."""

from __future__ import annotations

import os
import tempfile
import torch
import onnx
from onnxsim import simplify
from typing import Optional, Iterable, Any

try:
    from onnxruntime.quantization import (
        quantize_static as ort_quantize_static,
        quantize_dynamic as ort_quantize_dynamic,
        quant_pre_process,
        CalibrationDataReader,
        QuantFormat,
        QuantType,
    )
except ImportError:
    # Provide fallbacks if onnxruntime is not available
    ort_quantize_static = None
    ort_quantize_dynamic = None
    quant_pre_process = None
    CalibrationDataReader = object  # type: ignore
    class QuantFormat:  # type: ignore
        QOperator = None
    class QuantType:  # type: ignore
        QInt8 = None
        QUInt8 = None

class GenericCalibrationDataReader(CalibrationDataReader):
    """A simple calibration data reader for ONNX Runtime quantisation.

    This reader iterates over a PyTorch dataloader and yields input batches
    keyed by the provided input name.  It works with both classification
    datasets (where each batch is a tuple of ``(inputs, targets)``) and
    regression/time‑series datasets.

    Parameters
    ----------
    dataloader: iterable
        An iterable yielding input batches; can be a PyTorch DataLoader.
    input_name: str
        Name of the ONNX model input node.
    max_samples: int, optional
        Maximum number of batches to use for calibration.  If ``None``, the
        entire dataloader is consumed.
    """
    def __init__(self, dataloader: Iterable[Any], input_name: str, max_samples: Optional[int] = None) -> None:
        self._iter = iter(dataloader)
        self.input_name = input_name
        self.max_samples = max_samples
        self._count = 0

    def get_next(self) -> Optional[dict]:
        if self.max_samples is not None and self._count >= self.max_samples:
            return None
        try:
            batch = next(self._iter)
        except StopIteration:
            return None
        self._count += 1
        # Extract inputs; batch may be (inputs, targets) or just inputs
        if isinstance(batch, (tuple, list)):
            inputs = batch[0]
        else:
            inputs = batch
        # Convert PyTorch tensor to numpy
        if isinstance(inputs, torch.Tensor):
            arr = inputs.detach().cpu().numpy()
        else:
            # Assume it is already numpy or list
            arr = inputs
        return {self.input_name: arr}

def quantize_model(onnx_model_path: str, output_model_path: str, calibration_loader: Optional[Iterable[Any]] = None,
                   input_name: str = 'input', quant_format: Optional[str] = 'qoperator',
                   activation_type: Optional[str] = 'quint8', weight_type: Optional[str] = 'qint8',
                   max_calibration_batches: int = 200) -> None:
    """Quantise an ONNX model using ONNX Runtime.

    This helper performs either static or dynamic quantisation depending on
    whether a calibration loader is provided.  For static quantisation
    ``calibration_loader`` must be an iterable yielding only the input tensors.

    Parameters
    ----------
    onnx_model_path: str
        Path to the FP32 ONNX model.
    output_model_path: str
        Path where the quantised model will be saved.
    calibration_loader: iterable, optional
        Dataloader used to collect activation statistics for static
        quantisation.  If ``None``, dynamic quantisation is applied.
    input_name: str
        Name of the input node in the ONNX model.
    quant_format: str
        Quantisation format; currently only 'qoperator' is supported.
    activation_type: str
        Data type for activations; 'quint8' for unsigned 8‑bit.
    weight_type: str
        Data type for weights; 'qint8' for signed 8‑bit.
    max_calibration_batches: int
        Maximum number of batches to use during static calibration.
    """
    if ort_quantize_static is None or ort_quantize_dynamic is None:
        raise RuntimeError("onnxruntime.quantization is not available; please install onnxruntime to use quantisation features.")
    if calibration_loader is None:
        # Use dynamic quantisation
        wt = QuantType.QInt8 if weight_type.lower() == 'qint8' else QuantType.QUInt8
        ort_quantize_dynamic(
            onnx_model_path,
            output_model_path,
            weight_type=wt,
            op_types_to_quantize=["MatMul", "Gemm"],
            per_channel=False
        )
    else:
        # Perform static quantisation; first prepare the model
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessed_path = os.path.join(tmpdir, 'preprocessed.onnx')
            # Preprocess may insert quantisation nodes and fold constants
            quant_pre_process(
                input_model_path=onnx_model_path,
                output_model_path=preprocessed_path,
                skip_optimization=False,
            )
            # Setup calibration reader
            reader = GenericCalibrationDataReader(calibration_loader, input_name=input_name, max_samples=max_calibration_batches)
            fmt = QuantFormat.QOperator
            act_type = QuantType.QUInt8 if activation_type.lower() == 'quint8' else QuantType.QInt8
            wt_type = QuantType.QInt8 if weight_type.lower() == 'qint8' else QuantType.QUInt8
            ort_quantize_static(
                model_input=preprocessed_path,
                model_output=output_model_path,
                calibration_data_reader=reader,
                quant_format=fmt,
                activation_type=act_type,
                weight_type=wt_type,
                op_types_to_quantize=['MatMul', 'Gemm', 'Conv'],
                extra_options={'unify_op_type_gen_bool_device': True},
            )
    # Simplify the quantized model
    onnx_model = onnx.load(output_model_path)
    onnx_model, _ = simplify(onnx_model, dynamic_input_shape=False, skip_fuse_bn=True)
    onnx.save(onnx_model, output_model_path)