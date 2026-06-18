"""Wrapper around the onnx2c CLI tool.

This module exposes a function to invoke the `onnx2c` command line tool
to convert a quantised ONNX model into C source code.  The tool must
already be installed and available on the system PATH.  See
<https://github.com/sixty-north/onnx2c> for installation instructions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional

def convert_to_c(onnx_path: str, output_dir: str, onnx2c_executable: str = 'onnx2c',
                 overwrite: bool = True, quant=False) -> None:
    """Convert a quantised ONNX model to C using onnx2c.

    Parameters
    ----------
    onnx_path: str
        Path to the quantised ONNX model.
    output_dir: str
        Directory where the generated C source files will be written.  If the
        directory does not exist it will be created.
    onnx2c_executable: str
        Name or path of the onnx2c executable.  Defaults to 'onnx2c'.
    overwrite: bool
        If ``True``, existing output files will be overwritten.  Otherwise
        the function will raise an error if the output directory is not empty.
    """
    if shutil.which(onnx2c_executable) is None:
        raise RuntimeError(
            f"The onnx2c executable '{onnx2c_executable}' was not found. "
            "Please install onnx2c and ensure it is on your PATH."
        )
    if not overwrite and os.listdir(output_dir):
        raise FileExistsError(f"Output directory '{output_dir}' is not empty.")
    cmd = [onnx2c_executable, '-l', '4', onnx_path]
    if quant:
        cmd = [onnx2c_executable, '-l', '4', '-q', onnx_path]
    print(f"Running command: {' '.join(cmd)}")
    # result = subprocess.run(cmd, capture_output=True, text=True)
    with open(output_dir, "w") as f:
        result = subprocess.run(cmd, stdout=f)
    if result.returncode != 0:
        raise RuntimeError(f"onnx2c failed with exit code {result.returncode}:\n{result.stderr}")
    print(f"C source generated in {output_dir}")