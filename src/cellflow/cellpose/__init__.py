"""Cellpose stage + standalone segment/track tool.

Two roles share this package:

* **Integrated app** — the local Cellpose-SAM runner (``cellpose_runner``) and
  the divergence-derived foreground/contour maps (``build_divergence_maps``) that
  the orchestrator's in-app stage produces for nucleus tracking
  (``cellflow-tracking``) and cell segmentation.
* **Standalone ``cellflow-cellpose`` tool** — ``native_masks`` captures the
  Cellpose native masks the runner otherwise discards, and ``track_laptrack``
  links them across time, giving a self-contained "segment then track" product.

The Cellpose model is the optional ``[cellpose]`` extra and ``laptrack`` is the
``[laptrack]`` extra; both are imported lazily, so importing this package does
not require either.
"""
from __future__ import annotations

from cellflow.cellpose import cellpose_runner, native_masks, track_laptrack
from cellflow.cellpose.divergence_maps import (
    DivergenceMapsReport,
    build_divergence_maps,
)

__all__ = [
    "DivergenceMapsReport",
    "build_divergence_maps",
    "cellpose_runner",
    "native_masks",
    "track_laptrack",
]
