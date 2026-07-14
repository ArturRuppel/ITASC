"""Ultrack-based ILP tracker for ITASC candidate databases."""
from __future__ import annotations

__all__ = ["TrackingConfig"]


def __getattr__(name: str):
    if name == "TrackingConfig":
        from itasc.tracking_ultrack.config import TrackingConfig

        return TrackingConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
