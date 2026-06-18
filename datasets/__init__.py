"""Dataset loaders for MyAIFramework.

Each module in this package should expose at least one function:

* ``get_dataloaders()`` – returns a tuple of ``(train_loader, test_loader)``
  suitable for training and evaluation.

Some datasets may also expose ``get_calibration_loader()`` to provide a
specialised loader for quantisation calibration.
"""

from importlib import import_module

def get_dataloaders(name: str, **kwargs):
    """Return training and testing dataloaders by dataset name.

    Parameters
    ----------
    name: str
        Name of the dataset module within ``datasets``.
    **kwargs:
        Additional parameters forwarded to the dataset loader.
    """
    try:
        module = import_module(f'.{name}', package=__name__)
    except ModuleNotFoundError as e:
        raise ValueError(f"Unknown dataset '{name}'. Available datasets are defined in the datasets/ directory.") from e
    if not hasattr(module, 'get_dataloaders'):
        raise AttributeError(f"Dataset module '{name}' does not define a 'get_dataloaders' function.")
    return module.get_dataloaders(**kwargs)

def get_calibration_loader(name: str, **kwargs):
    """Return a dataloader suitable for quantisation calibration.

    If the dataset module does not provide a custom calibration loader, this
    function will fall back to returning the training loader.
    """
    module = import_module(f'.{name}', package=__name__)
    if hasattr(module, 'get_calibration_loader'):
        return module.get_calibration_loader(**kwargs)
    # Fallback: use part of the training set for calibration
    train_loader, _ = get_dataloaders(name, **kwargs)
    return train_loader