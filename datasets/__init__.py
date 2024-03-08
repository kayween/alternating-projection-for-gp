"""Regression datasets."""
from . import uci
from ._synthetic import SyntheticDataset

# Public classes and functions. Order is reflected in documentation.
__all__ = [
    "SyntheticDataset",
]

# Set correct module paths. Corrects links and module paths in documentation.
SyntheticDataset.__module__ = "datasets"
