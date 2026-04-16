"""Runtime stage discovery via Python entry points.

Usage::

    from cellflow.napari._plugin import STAGES, refresh

    # All installed cellflow stages, keyed by entry-point name:
    stage = STAGES["cellpose_nucleus"]

    # Re-scan after a new package is installed at runtime:
    refresh()
"""
from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Dict

# Canonical display order for stage entry-point names.
# Stages not listed here are appended alphabetically after these.
STAGE_ORDER: list[str] = [
    "raw_import",
    "cellpose_nucleus",
    "cellpose_cell",
    "contours",
    "tracking",
    "correction",
    "cell_segmentation",
    "graph_extraction",
    "topology_analysis",
]

# Human-readable labels for the UI (tab titles, dialog checkboxes).
STAGE_DISPLAY_NAMES: Dict[str, str] = {
    "raw_import":        "Input Export",
    "cellpose_nucleus":  "Cellpose Nuclei (3D)",
    "cellpose_cell":     "Cellpose Cells (2D)",
    "contours":          "Contours",
    "tracking":          "Ultrack",
    "correction":        "Correction",
    "cell_segmentation": "Cell Segmentation",
    "graph_extraction":  "Graph Extraction",
    "topology_analysis": "Topology Analysis",
}

# Which manifest key(s) drive the badge for each top-level pipeline tab.
# "Tracking" (LapTrack retracking) and "ForSys" have no manifest keys —
# they are part of the correction loop / downstream analysis.
TAB_STAGE_KEYS: Dict[str, list[str]] = {
    "Prepare Input Data": ["raw_import"],
    "Cellpose":     ["cellpose_nucleus", "cellpose_cell"],
    "Ultrack":      ["contours", "tracking"],
    "Correction":   ["correction"],
    "Flow Watershed": ["cell_segmentation"],
    "Edge Analysis": ["graph_extraction", "topology_analysis"],
}

# Module-level dict: populated at import time; refreshed via refresh().
STAGES: Dict[str, Any] = {}


def _sorted_stages(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return *raw* sorted by ``STAGE_ORDER``, unknowns appended alphabetically."""
    order = {name: i for i, name in enumerate(STAGE_ORDER)}
    keys = sorted(raw, key=lambda k: (order.get(k, len(STAGE_ORDER)), k))
    return {k: raw[k] for k in keys}


def refresh() -> Dict[str, Any]:
    """Reload all ``cellflow.stages`` entry points and update :data:`STAGES`."""
    global STAGES
    raw = {}
    for ep in entry_points(group="cellflow.stages"):
        try:
            raw[ep.name] = ep.load()
        except Exception:
            pass  # missing optional dep — stage simply absent from dict
    STAGES.clear()
    STAGES.update(_sorted_stages(raw))
    return STAGES


# Populate on first import.
refresh()
