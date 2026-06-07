"""Backward-compatible shim.

The lineage swimlane model is shared substrate (used by the correction lineage
panel, which is part of the independently-installable tracking/correction piece),
so it now lives in :mod:`cellflow.core.lineage`. This re-export keeps the historic
``cellflow.segmentation.lineage`` import path working for the full orchestrator and
any out-of-repo consumers.
"""
from __future__ import annotations

from cellflow.core.lineage import (
    LineageModel,
    TrackLane,
    TrackSegment,
    build_lineage,
)

__all__ = ["LineageModel", "TrackLane", "TrackSegment", "build_lineage"]
