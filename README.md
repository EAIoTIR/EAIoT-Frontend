# EAIoT

EAIoT is a compact Python framework for embedded-AI model workflows. It helps you take a PyTorch model through training, evaluation, optional pruning, ONNX export, ONNX Runtime quantization, and optional C code generation with `onnx2c`.

The project is intentionally small and script-oriented, so it is easy to adapt for experiments, coursework, prototypes, and embedded deployment pipelines.

---

## Features

- PyTorch training and evaluation helpers for classification and regression.
- Built-in MNIST CNN, MNIST MLP, and LSTM-style regression examples.
- Built-in MNIST and synthetic sine-wave dataset loaders.
- PyTorch pruning for `nn.Linear` and `nn.Conv2d` layers.
- Pruning can run once after training/loading, or gradually during training.
- During-training pruning treats `--prune-amount` as the desired final sparsity over all scheduled pruning steps.
- ONNX export from PyTorch models.
- Direct use of an already exported ONNX model through `--onnx-path`.
- ONNX Runtime static and dynamic quantization.
- Optional LSTM-to-linear replacement before ONNX export.
- ONNX-to-C conversion through the external `onnx2c` command-line tool.
- Optional export of one test input sample as a `.bin` file for generated-C testing when a dataset file is provided.

---

## Project structure

```text
.
├── main.py                    # Command-line training/export/quantization/C pipeline
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── __init__.py                # Small package-level helpers
├── converters/
│   └── onnx_to_c.py           # Wrapper around onnx2c
├── datasets/
│   ├── mnist.py               # MNIST dataloaders
│   └── sine.py                # Synthetic sine-wave dataloaders
├── models/
│   ├── mnist_cnn.py           # Simple CNN for MNIST-style images
│   ├── mnist_mlp.py           # MLP for MNIST-style images
│   ├── lstm.py                # LSTM regression model
│   └── lstm_linear.py         # LSTM-to-linear replacement utilities
├── pruning/
│   └── pruner.py              # PyTorch pruning helpers
├── quantization/
│   └── quantizer.py           # ONNX export and quantization helpers
└── trainers/
    ├── train.py               # Training loop
    └── evaluation.py          # Classification/regression evaluation
```

---

## Requirements

Recommended Python version: **Python 3.11+**.

Python packages are listed in `requirements.txt`:

- `torch`
- `torchvision`
- `numpy`
- `onnx`
- `onnxruntime`
- `onnxsim`

---

## Installation

Clone the repository:
```bash
git clone https://github.com/EAIoTIR/EAIoT-Frontend.git
cd EAIoT-Frontend
git submodule update --init
```

The C export step also requires the external `onnx2c` executable. This is **not** installed by `pip install -r requirements.txt`. Make sure `onnx2c` is included by submodule update --init or clone it separately:
```bash
git clone https://github.com/EAIoTIR/onnx2c.git
```
inside onnx2c, follow its README.md to clone its submodules completely.

The default expected path is:

```text
./onnx2c/build/onnx2c
```

then create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows PowerShell/CMD
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU/CUDA support, install the correct PyTorch and TorchVision build for your system before installing the rest of the dependencies.

---

## Quick start

Command line interface

### Train the built-in MNIST CNN and export ONNX

```bash
python main.py \
  --model-file models/mnist_cnn.py \
  --dataset-file datasets/mnist.py \
  --train true \
  --epochs 1 \
  --classification true \
  --output-dir build
```

This trains the model, evaluates classification accuracy, saves PyTorch weights, and exports an ONNX model.

### Train, prune, and quantize the MNIST CNN

```bash
python main.py \
  --model-file models/mnist_cnn.py \
  --dataset-file datasets/mnist.py \
  --train true \
  --epochs 3 \
  --classification true \
  --prune after \
  --prune-amount 0.3 \
  --prune-method l1_unstructured \
  --quantize static \
  --act-type quint8 \
  --weight-type qint8 \
  --output-dir build
```

This applies 30% L1 unstructured pruning after training, saves pruned weights, exports ONNX, and writes a statically quantized ONNX model.

### Train the built-in LSTM on synthetic sine data

```bash
python main.py \
  --model-file models/lstm.py \
  --dataset-file datasets/sine.py \
  --train true \
  --epochs 5 \
  --classification false \
  --seq-len 12 \
  --quantize dynamic \
  --output-dir build
```

Use `--classification false` for regression datasets so the pipeline uses MSE loss and regression evaluation.

### Convert an existing ONNX model to C

```bash
python main.py \
  --onnx-path build/mnist_cnn_quant.onnx \
  --export-c \
  --quant-c true \
  --output-dir build
```

When `--onnx-path` is supplied, the pipeline skips PyTorch model loading, training, evaluation, pruning, and ONNX export. It uses the supplied ONNX file for optional quantization and/or C generation.

### Convert an existing ONNX model to a custom C output path

```bash
python main.py \
  --onnx-path build/mnist_cnn_quant.onnx \
  --export-c \
  --c-output-path build/generated/mnist_cnn.c \
  --onnx2c-executable ./onnx2c/build/onnx2c
```

---

## Command-line interface

Run:

```bash
python main.py --help
```

### Main PyTorch workflow options

| Option | Default | Description |
| --- | --- | --- |
| `--model-file` | required unless `--onnx-path` is used | Python file exposing `get_model()` and preferably `get_dummy_input()`. |
| `--dataset-file` | required unless only using `--onnx-path` without static quantization | Python file exposing `get_dataloaders()` and optionally `get_calibration_loader()`. |
| `--train` | `true` | Train the model. Use `false` to load existing PyTorch weights. |
| `--epochs` | `1` | Number of training epochs. Must be at least 1 for the PyTorch pipeline. |
| `--batch-size` | `64` | Training batch size. |
| `--test-batch-size` | `1000` | Test/evaluation batch size. |
| `--learning-rate` | `0.001` | Adam optimizer learning rate. |
| `--device` | auto | Device to use, such as `cpu` or `cuda`. If CUDA is requested but unavailable, the script raises an error. |
| `--output-dir` | `build` | Directory for generated weights, ONNX files, C files, and input `.bin` samples. |
| `--classification` | `true` | Set `true` for classification loss/evaluation and `false` for regression loss/evaluation. |
| `--seq-len` | `12` | Sequence length forwarded to compatible model/dataset helpers. |
| `--weights-path` | auto | Checkpoint path to load. Alias: `--load-model`. |

### Checkpoint loading

Default behavior:

```bash
python main.py \
  --model-file models/mnist_mlp.py \
  --dataset-file datasets/mnist.py \
  --train false \
  --output-dir build
```

This loads:

```text
build/mnist_mlp.pt
```

To load a specific checkpoint path:

```bash
python main.py \
  --model-file models/mnist_cnn.py \
  --dataset-file datasets/mnist.py \
  --train false \
  --weights-path build/mnist_cnn_pruned.pt \
  --output-dir build
```

To load by checkpoint stem from `--output-dir`:

```bash
python main.py \
  --model-file models/mnist_cnn.py \
  --dataset-file datasets/mnist.py \
  --train false \
  --weights-path mnist_cnn_pruned \
  --output-dir build
```

`--load-model` is kept as a backwards-compatible alias for `--weights-path`:

```bash
python main.py \
  --model-file models/mnist_cnn.py \
  --dataset-file datasets/mnist.py \
  --train false \
  --load-model mnist_cnn_pruned \
  --output-dir build
```

### ONNX, quantization, and C export options

| Option | Default | Description |
| --- | --- | --- |
| `--onnx-path` | `None` | Existing ONNX model to use directly, skipping PyTorch model loading, training, evaluation, pruning, and ONNX export. |
| `--quantize` | `none` | Quantization mode: `none`, `static`, or `dynamic`. |
| `--act-type` | `quint8` | Activation quantization type for static quantization. |
| `--weight-type` | `qint8` | Weight quantization type. |
| `--export-c` | `false` | Convert the selected ONNX model to C with `onnx2c`. |
| `--quant-c` | `true` | Whether to request quantized helper generation in `onnx2c`. |
| `--c-output-path` | auto | Optional output path for the generated C file. Defaults to `<output-dir>/<onnx_stem>.c`. |
| `--onnx2c-executable` | `./onnx2c/build/onnx2c` | Path or executable name for `onnx2c`. |

Rules for `--onnx-path`:

- `--onnx-path` cannot be combined with `--prune`; pruning must happen before ONNX export.
- If `--onnx-path` is provided, at least one action must be requested: `--export-c` and/or `--quantize`.
- `--quantize static` with `--onnx-path` requires `--dataset-file` so calibration data can be loaded.
- If `--dataset-file` is not supplied in ONNX-only mode, C generation still works, but `dataset_name_input.bin` is skipped.

### Pruning options

| Option | Default | Description |
| --- | --- | --- |
| `--prune` | `none` | Pruning mode: `none`, `during`, or `after`. |
| `--prune-amount` | `0.0` | Fraction of selected weights/channels to prune, for example `0.3`. |
| `--prune-method` | `l1_unstructured` | One of `l1_unstructured`, `random_unstructured`, `ln_structured`, `random_structured`. |
| `--prune-layers` | `both` | Target layer types: `linear`, `conv`, or `both`. |
| `--prune-scope` | `local` | `local` layer-wise pruning or `global` pruning across selected parameters. Global pruning supports unstructured methods only. |
| `--prune-bias` | `false` | Also prune bias tensors. Structured bias pruning is not supported by the helper. |
| `--prune-structured-dim` | `0` | Dimension for structured pruning. `0` prunes output channels/neurons. |
| `--prune-structured-norm` | `2.0` | Norm used by `ln_structured` pruning. |
| `--prune-start-epoch` | `1` | First epoch at which `--prune during` may run. |
| `--prune-frequency` | `1` | Epoch interval for `--prune during`. |
| `--prune-finalize` | `true` | Remove pruning reparameterizations so zeros are saved as normal weights. |
| `--prune-exclude-layers` | empty | Comma-separated exact module names to skip, such as `fc2,l3`. |

During-training pruning note:

When `--prune during` is used, the current `main.py` treats `--prune-amount` as the target final sparsity across all scheduled pruning events. For example:

```bash
python main.py \
  --model-file models/mnist_cnn.py \
  --dataset-file datasets/mnist.py \
  --train true \
  --epochs 5 \
  --prune during \
  --prune-amount 0.2 \
  --prune-start-epoch 2 \
  --prune-frequency 1 \
  --prune-layers both
```

The pruning schedule runs after epochs 2, 3, 4, and 5. Instead of pruning 20% four separate times, the script internally computes a smaller per-step pruning amount so the final sparsity is close to 20%.

---

## Workflow examples

### Train once, then reload weights later

First train and save weights:

```bash
python main.py \
  --model-file models/mnist_mlp.py \
  --dataset-file datasets/mnist.py \
  --train true \
  --epochs 2 \
  --output-dir build
```

Then reload the saved weights and export again without retraining:

```bash
python main.py \
  --model-file models/mnist_mlp.py \
  --dataset-file datasets/mnist.py \
  --train false \
  --output-dir build
```

### Prune after training

```bash
python main.py \
  --model-file models/mnist_cnn.py \
  --dataset-file datasets/mnist.py \
  --train true \
  --epochs 3 \
  --prune after \
  --prune-amount 0.3 \
  --prune-method l1_unstructured \
  --prune-layers both \
  --output-dir build
```

This saves:

```text
build/mnist_cnn_pruned.pt
build/mnist_cnn_pruned.onnx
```

### Prune during training

```bash
python main.py \
  --model-file models/mnist_cnn.py \
  --dataset-file datasets/mnist.py \
  --train true \
  --epochs 5 \
  --prune during \
  --prune-amount 0.2 \
  --prune-start-epoch 2 \
  --prune-frequency 1 \
  --prune-layers both \
  --output-dir build
```

This gradually prunes during training and saves pruned weights/ONNX using the `_pruned` suffix.

### Load previously pruned weights

```bash
python main.py \
  --model-file models/mnist_cnn.py \
  --dataset-file datasets/mnist.py \
  --train false \
  --weights-path build/mnist_cnn_pruned.pt \
  --output-dir build
```

Equivalent stem-based form:

```bash
python main.py \
  --model-file models/mnist_cnn.py \
  --dataset-file datasets/mnist.py \
  --train false \
  --weights-path mnist_cnn_pruned \
  --output-dir build
```

If the loaded checkpoint stem is `mnist_cnn_pruned`, the exported ONNX file will also use that stem:

```text
build/mnist_cnn_pruned.onnx
```

### Static quantization with calibration data

```bash
python main.py \
  --model-file models/mnist_cnn.py \
  --dataset-file datasets/mnist.py \
  --train true \
  --quantize static \
  --output-dir build
```

Static quantization uses `get_calibration_loader()` from the dataset file when available. If the dataset file does not define it, the training loader is used for calibration.

### Static quantization of an existing ONNX model

```bash
python main.py \
  --onnx-path build/mnist_cnn.onnx \
  --dataset-file datasets/mnist.py \
  --quantize static \
  --output-dir build
```

This writes:

```text
build/mnist_cnn_quant.onnx
```

### Dynamic quantization

```bash
python main.py \
  --model-file models/mnist_mlp.py \
  --dataset-file datasets/mnist.py \
  --train true \
  --quantize dynamic \
  --output-dir build
```

Dynamic quantization does not use calibration data.

### Dynamic quantization of an existing ONNX model

```bash
python main.py \
  --onnx-path build/mnist_mlp.onnx \
  --quantize dynamic \
  --output-dir build
```

### Generate C from a PyTorch workflow

```bash
python main.py \
  --model-file models/mnist_cnn.py \
  --dataset-file datasets/mnist.py \
  --train true \
  --epochs 1 \
  --quantize static \
  --export-c \
  --output-dir build
```

This trains, exports ONNX, quantizes it, converts the selected ONNX model to C, and writes one test input sample to:

```text
build/mnist_input.bin
```

### Generate C from an already exported ONNX model

```bash
python main.py \
  --onnx-path build/mnist_cnn_quant.onnx \
  --export-c \
  --c-output-path build/mnist_cnn_quant.c \
  --output-dir build
```

If a dataset is also supplied, the script can additionally create one `.bin` input sample:

```bash
python main.py \
  --onnx-path build/mnist_cnn_quant.onnx \
  --dataset-file datasets/mnist.py \
  --export-c \
  --output-dir build
```

---

## Custom model file format

A custom model file must expose a `get_model()` function. For ONNX export, it should also expose `get_dummy_input()`.

Example:

```python
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)


def get_model():
    return MyModel()


def get_dummy_input():
    return torch.randn(1, 1, 28, 28)
```

The saved `.pt` file is a PyTorch state dictionary:

```python
torch.save(model.state_dict(), "my_model.pt")
```

If `get_dummy_input()` is not provided, `main.py` tries to infer a dummy input from the first batch of the test dataloader.

---

## Custom dataset file format

A custom dataset file must expose `get_dataloaders()` and return:

```python
train_loader, test_loader
```

For static quantization, it can also expose `get_calibration_loader()`.

Example:

```python
from torch.utils.data import DataLoader, TensorDataset
import torch


def get_dataloaders(train_batch_size=64, test_batch_size=1000):
    x_train = torch.randn(512, 1, 28, 28)
    y_train = torch.randint(0, 10, (512,))
    x_test = torch.randn(128, 1, 28, 28)
    y_test = torch.randint(0, 10, (128,))

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=train_batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=test_batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def get_calibration_loader(batch_size=1):
    train_loader, _ = get_dataloaders(train_batch_size=batch_size)
    return train_loader
```

`main.py` only forwards keyword arguments that a function accepts, so custom `get_model()`, `get_dummy_input()`, `get_dataloaders()`, and `get_calibration_loader()` functions can define only the parameters they need.

---

## Built-in modules

### Models

- `models/mnist_cnn.py`: simple convolutional classifier for 28×28 grayscale images.
- `models/mnist_mlp.py`: multilayer perceptron for MNIST-style inputs.
- `models/lstm.py`: simple LSTM regression model.
- `models/lstm_linear.py`: tools to replace `nn.LSTM` layers with linear-layer equivalents.

### Datasets

- `datasets/mnist.py`: MNIST train/test loaders using TorchVision.
- `datasets/sine.py`: synthetic sine-wave sequence data for regression experiments.

### Training and evaluation

- `trainers/train.py`: generic PyTorch training loop with an optional epoch-end callback.
- `trainers/evaluation.py`: classification accuracy and regression MSE evaluation helpers.

### Pruning

- `pruning/pruner.py`: helper functions around `torch.nn.utils.prune`.
- Supports local and global unstructured pruning.
- Supports local structured pruning for selected `Linear` and `Conv2d` layers.
- Can prune during training or once after training/loading.
- By default, `--prune-finalize true` removes pruning reparameterizations before saving, so the checkpoint can be loaded as a normal state dictionary.

### Quantization

The `quantization/quantizer.py` module contains:

- `quantize_model(...)`
- `GenericCalibrationDataReader`

Dynamic quantization is used when no calibration loader is provided. Static quantization is used when a calibration loader is provided.

### C conversion

The `converters/onnx_to_c.py` module wraps the `onnx2c` CLI:

```python
from converters import convert_to_c

convert_to_c(
    onnx_path="model_quant.onnx",
    output_dir="model_quant.c",
    onnx2c_executable="onnx2c",
    quant=True,
)
```

Despite the argument name `output_dir`, the current implementation writes to a C output file path. From the CLI, use `--c-output-path` to choose the generated C file path.

---

## Output files

Depending on the workflow, the framework can create:

```text
build/
├── model_name.pt                    # saved PyTorch state dict
├── model_name_pruned.pt             # saved pruned state dict, when pruning is applied
├── model_name.onnx                  # exported ONNX model
├── model_name_pruned.onnx           # exported ONNX model after pruning
├── model_name_quant.onnx            # quantized ONNX model
├── model_name_pruned_quant.onnx     # quantized ONNX model after pruning
├── onnx_stem.c                      # generated C source for selected ONNX model
└── dataset_name_input.bin           # one test input sample, when C export has a dataset loader
```

The exact filenames depend on the model filename, loaded checkpoint stem, pruning mode, quantization mode, selected ONNX file, and selected output directory.

---
## Test

`test.py` compares an FP32 ONNX model with its quantized ONNX version by running the same test samples through both models.
### Basic usage

```bash
python test.py \
  --fp32-model build/model.onnx \
  --quantized-model build/model_quant.onnx \
  --dataset-file datasets/my_dataset.py
```

By default, the script:

- Automatically detects classification or regression
- Tests 10 samples
- Prints 10 per-sample comparisons
- Uses an absolute tolerance of `1e-5`
- Uses a relative tolerance of `1e-3`

### Classification example

```bash
python test.py \
  --fp32-model build/mnist_cnn.onnx \
  --quantized-model build/mnist_cnn_quant.onnx \
  --dataset-file datasets/mnist.py \
  --task classification \
  --num-samples 20
```

For classification models, the summary includes:

- FP32 predicted class
- Quantized predicted class
- Prediction agreement rate
- FP32 accuracy, when labels are available
- Quantized accuracy, when labels are available
- Maximum softmax probability for each model

### Regression example

```bash
python test.py \
  --fp32-model build/sine_lstm.onnx \
  --quantized-model build/sine_lstm_quant.onnx \
  --dataset-file datasets/sine.py \
  --task regression \
  --num-samples 50 \
  --seq-len 12
```

For regression models, the summary can include:

- FP32 target MAE and MSE
- Quantized target MAE and MSE
- Numerical differences between model outputs

Target metrics are reported only when the dataset supplies targets with shapes compatible with the model output.

### Dataset file requirements

The file passed through `--dataset-file` must define a function named:

```python
def get_dataloaders(...):
    return train_loader, test_loader
```

The tester uses the second returned object as the test loader.

It attempts to pass these arguments when supported by the function:

```python
train_batch_size=1
test_batch_size=1
batch_size=1
seq_len=<value from --seq-len>
```

Unsupported arguments are ignored automatically.

A test loader may return batches in any of these forms:

```python
inputs, targets
```

```python
{"input": inputs, "target": targets}
```

```python
inputs
```

Recognized dictionary input keys are:

- `input`
- `inputs`
- `x`
- `data`

Recognized target keys are:

- `target`
- `targets`
- `y`
- `label`
- `labels`

The script extracts individual samples and keeps a batch dimension of one for ONNX inference.

### Passing custom dataset arguments

Use `--dataset-arg KEY=VALUE` to pass extra values to `get_dataloaders()`.

The option may be repeated:

```bash
python test.py \
  --fp32-model build/model.onnx \
  --quantized-model build/model_quant.onnx \
  --dataset-file datasets/my_dataset.py \
  --dataset-arg download=false \
  --dataset-arg data_dir='"./data"' \
  --dataset-arg normalize=true
```

Values are parsed as JSON when possible. This means:

- `true` and `false` become Boolean values
- Numbers become numeric values
- Quoted text becomes a string
- Unquoted text is passed as a string when JSON parsing fails

### Controlling displayed samples

Test 100 samples but print only the first 5:

```bash
python test.py \
  --fp32-model build/model.onnx \
  --quantized-model build/model_quant.onnx \
  --dataset-file datasets/my_dataset.py \
  --num-samples 100 \
  --show-samples 5
```

Limit the number of output values displayed per sample:

```bash
--max-output-values 5
```

This only affects terminal output. All output values are still used in the comparison and included in JSON results.

### Tolerances

Each sample is checked with:

```python
numpy.allclose(fp32_output, quantized_output, rtol=rtol, atol=atol)
```

Configure the tolerances with:

```bash
python test.py \
  --fp32-model build/model.onnx \
  --quantized-model build/model_quant.onnx \
  --dataset-file datasets/my_dataset.py \
  --atol 1e-4 \
  --rtol 1e-2
```

Quantized outputs are not normally bit-for-bit identical to FP32 outputs. Choose tolerances based on the model, output range, and acceptable quality loss.

### Saving JSON results

```bash
python test.py \
  --fp32-model build/model.onnx \
  --quantized-model build/model_quant.onnx \
  --dataset-file datasets/my_dataset.py \
  --num-samples 100 \
  --output-json build/comparison.json
```

The JSON file contains:

```json
{
  "summary": {
    "task": "classification",
    "samples": 100,
    "output_mae": 0.0012,
    "prediction_agreement": 0.99
  },
  "samples": [
    {
      "index": 0,
      "fp32_output": [],
      "quantized_output": [],
      "within_tolerance": true
    }
  ]
}
```

The exact fields depend on whether the task is classification or regression and whether targets are available.

### CI and automated validation

Fail with exit code `1` when any output exceeds the configured tolerance:

```bash
python test.py \
  --fp32-model build/model.onnx \
  --quantized-model build/model_quant.onnx \
  --dataset-file datasets/my_dataset.py \
  --fail-on-output-mismatch
```

For classification, fail when any predicted class changes:

```bash
python test.py \
  --fp32-model build/model.onnx \
  --quantized-model build/model_quant.onnx \
  --dataset-file datasets/my_dataset.py \
  --task classification \
  --fail-on-prediction-change
```

Both checks can be enabled together:

```bash
python test.py \
  --fp32-model build/model.onnx \
  --quantized-model build/model_quant.onnx \
  --dataset-file datasets/my_dataset.py \
  --task classification \
  --num-samples 100 \
  --atol 1e-4 \
  --rtol 1e-2 \
  --fail-on-output-mismatch \
  --fail-on-prediction-change \
  --output-json build/comparison.json
```

### Command-line options

| Option | Description | Default |
|---|---|---:|
| `--fp32-model` | Path to the FP32 ONNX model | Required |
| `--quantized-model` | Path to the quantized ONNX model | Required |
| `--quant-model` | Alias for `--quantized-model` | — |
| `--dataset-file` | Python dataset module containing `get_dataloaders()` | Required |
| `--task` | `auto`, `classification`, or `regression` | `auto` |
| `--num-samples` | Number of test samples to compare | `10` |
| `--show-samples` | Number of per-sample results to print | `10` |
| `--max-output-values` | Maximum values printed from each output | `10` |
| `--seq-len` | Sequence length passed to compatible datasets | `12` |
| `--dataset-arg` | Extra `KEY=VALUE` argument for the dataset loader | None |
| `--atol` | Absolute comparison tolerance | `1e-5` |
| `--rtol` | Relative comparison tolerance | `1e-3` |
| `--fail-on-output-mismatch` | Exit with status `1` when an output exceeds tolerance | Disabled |
| `--fail-on-prediction-change` | Exit with status `1` when a class prediction changes | Disabled |
| `--output-json` | Path for detailed JSON results | None |

Display the built-in help page with:

```bash
python test.py --help
```

### Reported numerical metrics

#### Output MAE

The average absolute difference between all FP32 and quantized output values.

Lower is better.

#### Output RMSE

The root mean square difference between outputs. It gives larger errors more weight than MAE.

Lower is better.

#### Maximum absolute difference

The largest individual absolute output difference observed across all tested samples.

Lower is better.

#### Relative L2 error

The L2 norm of the output difference divided by the L2 norm of the FP32 output.

Lower is better.

#### Cosine similarity

Measures whether the FP32 and quantized output vectors point in the same direction.

A value near `1.0` indicates very similar output direction.

#### Samples within tolerance

The percentage of samples whose complete output passed the configured `numpy.allclose()` check.

#### Prediction agreement

For classification, the percentage of samples for which FP32 and quantized models selected the same class.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'onnxruntime'`

Install the dependencies:

```bash
pip install -r requirements.txt
```

### `tkinter` cannot be imported

On some Linux systems, ONNX simplification or related tooling may require Tk support. Install the system Tk package:

```bash
sudo apt install python3-tk
```

### `onnx2c executable was not found`

Install or build `onnx2c`, then either:

- add it to your system `PATH`, or
- pass the executable path directly:

```bash
python main.py \
  --onnx-path build/model.onnx \
  --export-c \
  --onnx2c-executable /path/to/onnx2c
```

### CLI errors when no model or dataset file is provided

The PyTorch workflow requires both `--model-file` and `--dataset-file`. Use `--onnx-path` when you only want to quantize or convert an existing ONNX model.

### `--onnx-path was provided, but no action was requested`

When using `--onnx-path`, request at least one output action:

```bash
--export-c
```

or:

```bash
--quantize dynamic
```

or:

```bash
--quantize static --dataset-file datasets/mnist.py
```

### `--quantize static with --onnx-path requires --dataset-file`

Static quantization needs calibration data. Provide a dataset file:

```bash
python main.py \
  --onnx-path build/model.onnx \
  --dataset-file datasets/mnist.py \
  --quantize static
```

### `--onnx-path cannot be combined with --prune`

Pruning is a PyTorch-model operation in this framework. Prune before ONNX export, then pass the exported pruned ONNX file with `--onnx-path`.

### `--prune during` fails when `--train false`

During-training pruning requires training to run. Use:

```bash
--train true --prune during
```

or apply pruning once after loading/training with:

```bash
--train false --prune after
```

### Loading a checkpoint saved with `--prune-finalize false` fails

A checkpoint saved with `--prune-finalize false` may contain pruning masks and reparameterized parameter names. Save a finalized checkpoint instead:

```bash
--prune-finalize true
```

### Global structured pruning fails

Global pruning is only supported for unstructured methods in this project. Use `--prune-scope local` with structured methods, or use an unstructured method with `--prune-scope global`.

### ONNX export fails because input shape is wrong

Add or adjust `get_dummy_input()` in the model file:

```python
def get_dummy_input():
    return torch.randn(1, 1, 28, 28)
```

Change the shape to match your model input.

---
