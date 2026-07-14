"""Top-level package metadata for ITASC.

The supported top-level API is intentionally small while ITASC is under
active research development. Import concrete workflow functions from their
subpackage modules.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("itasc")
except PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = ["__version__"]
