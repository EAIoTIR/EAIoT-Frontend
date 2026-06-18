"""Top level package for MyAIFramework.

This package provides a simple interface for training, evaluating and converting
neural network models.  See the README for an overview of the available modules.
"""

from importlib import import_module

def get_model(name: str, **kwargs):
    """Return a model instance by name.

    Parameters
    ----------
    name: str
        The name of the model architecture.  Must correspond to a module in
        ``MyAIFramework.models`` that defines a ``get_model`` function.
    **kwargs:
        Additional keyword arguments forwarded to the model constructor.

    Returns
    -------
    torch.nn.Module
        An uninitialised model ready for training.
    """
    try:
        module = import_module(f'.models.{name}', package=__name__)
    except ModuleNotFoundError as e:
        raise ValueError(f"Unknown model '{name}'. Available models are defined in the models/ directory.") from e
    if not hasattr(module, 'get_model'):
        raise AttributeError(f"Model module '{name}' does not define a 'get_model' function.")
    return module.get_model(**kwargs)

def get_dummy_input(name: str, **kwargs):
    """Return a dummy input tensor for ONNX export.

    Not all models require a dummy input; in such cases this function will
    attempt to call ``get_dummy_input()`` on the model module.  If it does not
    exist, it will construct a reasonable tensor based on the known input
    dimensionality of common models.
    """
    module = import_module(f'.models.{name}', package=__name__)
    if hasattr(module, 'get_dummy_input'):
        return module.get_dummy_input(**kwargs)
    # Fallback: try to infer shapes
    if name.lower() in ('cnn', 'mlp'):
        import torch
        return torch.randn(1, 1, 28, 28)
    elif name.lower() == 'lstm':
        import torch
        # Default to a sequence length of 12 for the dummy input
        return torch.randn(12, 1)
    else:
        raise ValueError(f"Don't know how to generate a dummy input for model '{name}'.")