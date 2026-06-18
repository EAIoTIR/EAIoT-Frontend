import torch
import onnx
from onnxsim import simplify

def export_to_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, output_path: str,
                   opset_version: int = 14, simplify_model: bool = True) -> None:
    """Export a PyTorch model to ONNX and optionally simplify it.

    Parameters
    ----------
    model: nn.Module
        Trained (or untrained) PyTorch model to export.
    dummy_input: torch.Tensor
        Example input with the correct shape.  Used by ``torch.onnx.export``.
    output_path: str
        Path to write the ONNX file.
    opset_version: int
        ONNX opset version to target.
    simplify_model: bool
        If ``True``, run the ONNX simplifier on the exported graph.
    """
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=None,
            dynamo=False
        )
    if simplify_model:
        onnx_model = onnx.load(output_path)
        onnx_model, _ = simplify(onnx_model, dynamic_input_shape=False, skip_fuse_bn=True)
        for n in onnx_model.graph.node:
            if n.op_type == "Reshape":
                for a in n.attribute:
                    if a.name == "allowzero":
                        a.i = 0   # zero in shape means "copy input dimension"
        onnx.save(onnx_model, output_path)
