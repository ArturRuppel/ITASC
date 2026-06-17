"""Build the premade SuperPlot analyses for one CellFlow tidy table.

For every numeric descriptor and every comparison axis present in the table, emit
one Iris analysis spec (grammar 2.0): a swarm + notched box of the objects
coloured by date, with the per-date replicate means overlaid as distinct-shaped
markers, and the inferential test reading the **date** level (Iris's
pseudoreplication guard — the coarsest layer level is the unit the test compares).

The two axes (``condition`` and ``class_label``) each get their own analyses and
are never crossed. A 1-level axis degrades to a single-column descriptive
SuperPlot (no test) inside Iris; we still emit it because the same template
becomes a real comparison on multi-condition data. See
``docs/superpowers/specs/2026-06-17-iris-export-design.md``.
"""
from __future__ import annotations

import pandas as pd

from .schema import numeric_descriptors

SPEC_VERSION = "2.0"

#: Iris blocks a per-row geom (the swarm) above this many marks (its ``POINT_CAP``).
#: Above it we drop the swarm layer and keep the box + per-date overlay — a swarm
#: of thousands of points would be unreadable regardless.
POINT_CAP = 3000

#: Comparison axes in emission order. Each yields its own set of analyses; they
#: are never combined into one figure.
COMPARISON_AXES = ("condition", "class_label")

#: The replicate column: colour + shape encoding, and the test's independent unit.
REPLICATE = "date"


def build_analyses(df: pd.DataFrame, schema: dict, object_key: str) -> list[dict]:
    """All premade analyses for *df*.

    *object_key* is the table's finest object level (e.g. ``cell_id``); the spine
    is ``date → position_id → object_key → frame`` restricted to present columns.
    The swarm layer is included only when the object count is within Iris's point
    cap.
    """
    spine = [c for c in (REPLICATE, "position_id", object_key, "frame") if c in df.columns]
    descriptors = numeric_descriptors(schema)
    include_swarm = _object_count(df, object_key) <= POINT_CAP

    analyses: list[dict] = []
    for axis in COMPARISON_AXES:
        if axis not in df.columns:
            continue
        # A test needs ≥2 groups; a single-level axis (e.g. one condition) renders
        # describe-only — Iris raises 422 for an unpinned test on one group. The
        # same template becomes a real comparison once the axis has ≥2 levels.
        describe_only = int(df[axis].nunique(dropna=True)) < 2
        for descriptor in descriptors:
            analyses.append(
                _superplot(axis, descriptor, spine, object_key,
                           include_swarm, describe_only)
            )
    return analyses


def _object_count(df: pd.DataFrame, object_key: str) -> int:
    """Number of distinct objects the swarm would draw — one per
    (date, position, object_key), with ``frame`` averaged away at the object level."""
    keys = [c for c in (REPLICATE, "position_id", object_key) if c in df.columns]
    return int(df[keys].drop_duplicates().shape[0]) if keys else len(df)


def _superplot(
    axis: str, descriptor: str, spine: list[str], object_key: str,
    include_swarm: bool, describe_only: bool,
) -> dict:
    layers: list[dict] = []
    if include_swarm:
        layers.append({"geom": "dot", "level": object_key, "params": {"layout": "swarm"}})
    layers.append({"geom": "box", "level": object_key, "params": {}})
    layers.append({"geom": "dot", "level": REPLICATE, "params": {"jitter": 0.0}})
    # Unpinned ⇒ Iris recommends the test on load (coarsest layer level = date, so
    # the test compares per-date aggregates). describe_only ⇒ no test (single-group
    # axis), shown as a descriptive SuperPlot.
    stats = {"chosen_by": "describe_only"} if describe_only else {}
    return {
        "spec_version": SPEC_VERSION,
        "id": f"superplot__{axis}__{descriptor}",
        # Iris encodings are objects ({"column": name}), not bare strings; an
        # unmapped channel is None.
        "encodings": {
            "x": {"column": axis}, "y": {"column": descriptor},
            "color": {"column": REPLICATE}, "shape": {"column": REPLICATE},
            "size": None,
        },
        "facet": {"row": None, "col": None, "share_x": True, "share_y": True},
        "hierarchy": {"spine": spine, "fn": {}},
        "style": {"overrides": {"notch": True}},
        "layers": layers,
        "stats": stats,
    }
