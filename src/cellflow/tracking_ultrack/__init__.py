"""Ultrack-based ILP tracker for CellFlow v2 hypotheses."""
from __future__ import annotations

__all__ = ["TrackingConfig", "ingest_hypotheses_to_db"]


def __getattr__(name: str):
    if name == "TrackingConfig":
        from cellflow.tracking_ultrack.config import TrackingConfig

        return TrackingConfig
    if name == "ingest_hypotheses_to_db":
        from cellflow.tracking_ultrack.ingest import ingest_hypotheses_to_db

        return ingest_hypotheses_to_db
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
