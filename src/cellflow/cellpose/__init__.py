"""Cellpose stage — local Cellpose-SAM runner + divergence-based map building.

This is the shared upstream stage of the CellFlow workflow: it turns raw
``0_input`` stacks into the ``1_cellpose`` probability/flow maps (via Cellpose)
and the divergence-derived foreground/contour maps that both nucleus tracking
(``cellflow-tracking``) and cell segmentation (``cellflow-segmentation``)
consume as ``.tif`` inputs.

Shipped as the independently-installable ``cellflow-cellpose`` distribution; the
Cellpose model itself is an optional ``[cellpose]`` extra (imported lazily by
``cellpose_runner``), so importing this package — and running the divergence-map
builder on precomputed prob/dp maps — does not require it.
"""
from __future__ import annotations

from cellflow.cellpose import cellpose_runner
from cellflow.cellpose.divergence_maps import (
    DivergenceMapsReport,
    build_divergence_maps,
)

__all__ = [
    "DivergenceMapsReport",
    "build_divergence_maps",
    "cellpose_runner",
]
