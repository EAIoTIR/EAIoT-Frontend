"""Training and evaluation utilities for MyAIFramework."""

from .train import train_model
from .evaluation import evaluate_classification, evaluate_regression

__all__ = ['train_model', 'evaluate_classification', 'evaluate_regression']