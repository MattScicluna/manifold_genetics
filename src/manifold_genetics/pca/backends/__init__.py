"""PCA backends: pure-Python by default, external flashpca binary optionally."""

from .base import PCABackend, PCAModel
from .sklearn_backend import SklearnPCABackend

__all__ = ["PCABackend", "PCAModel", "SklearnPCABackend"]
