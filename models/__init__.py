"""Model definitions available in the framework.

Each module in this package should expose at least two functions:

* ``get_model()`` – returns a new instance of the model.
* ``get_dummy_input()`` – returns a sample input tensor suitable for ONNX export
  (optional; a reasonable default will be used if omitted).

To add a new architecture, create a new module here and implement these
functions.
"""

__all__ = [
    'cnn',
    'mlp',
    'lstm',
]