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
    "flow_watershed",
    "contours",
    "tracking",
    "project2d",
    "cell_labels",
    "graph_extraction",
    "topology_analysis",
]

# Human-readable labels for the UI (tab titles, dialog checkboxes).
STAGE_DISPLAY_NAMES: Dict[str, str] = {
    "raw_import":        "Raw Import (s00)",
    "cellpose_nucleus":  "Cellpose Nucleus (s01a)",
    "cellpose_cell":     "Cellpose Cell (s01b)",
    "flow_watershed":    "Flow Watershed (s02)",
    "contours":          "Contours (s02c)",
    "tracking":          "Tracking (s03)",
    "project2d":         "Project 2D (s04)",
    "cell_labels":       "Cell Labels (s05)",
    "graph_extraction":  "Graph Extraction",
    "topology_analysis": "Topology Analysis",
}

# Which manifest key(s) drive the badge for each top-level pipeline tab.
TAB_STAGE_KEYS: Dict[str, list[str]] = {
    "Data Prep":      ["raw_import"],
    "Cellpose":       ["cellpose_nucleus", "cellpose_cell"],
    "Flow Watershed": ["flow_watershed", "contours"],
    "Ultrack":        ["tracking"],
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
