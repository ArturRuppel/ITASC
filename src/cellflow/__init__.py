"""Top-level package metadata for CellFlow.

The supported top-level API is intentionally small while CellFlow is under
active research development. Import concrete workflow functions from their
subpackage modules.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cellflow")
except PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = ["__version__"]
