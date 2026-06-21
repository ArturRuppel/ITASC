"""Build the premade SuperPlot analyses for one CellFlow tidy table.

For every (kept) numeric descriptor and every comparison axis present in the
table, emit one Iris analysis spec (grammar 2.0): a **violin** of the per-object
distribution with the **per-date replicate means overlaid as a swarm** of
distinct-shaped markers, and the inferential test reading the **date** level
(Iris's pseudoreplication guard — the coarsest layer level is the unit the test
compares).

Design choices (see ``docs/superpowers/specs/2026-06-17-iris-export-design.md``):

* No per-object swarm and no notched box. With tens of thousands of cells a swarm
  is unreadable and a median-CI notch is wildly overconfident. A violin shows the
  distribution shape honestly; the only points drawn are the few date means.
* Descriptors that are reference-frame coordinates, velocity components, or raw
  angles are pruned — comparing them across conditions is not meaningful.
* Analyses are ordered and titled by quantity *family* (``cell_shape`` →
  "Cell shape", …) so they cluster as groups in Iris's flat analysis list (Iris
  has no section UI; ordering + a family-prefixed ``title`` is the grouping).
* The two axes (``condition`` and ``class_label``) each get their own analyses and
  are never crossed. A <2-level axis is emitted describe-only (Iris 422s on an
  unpinned test over one group).
"""
from __future__ import annotations

import pandas as pd

from .schema import numeric_descriptors

SPEC_VERSION = "2.0"

#: Comparison axes in emission order. Each yields its own analyses; never crossed.
COMPARISON_AXES = ("condition", "class_label")

#: The replicate column: colour + shape encoding, and the test's independent unit.
REPLICATE = "date"

#: Descriptor leaf names that are reference-frame coordinates / vector components /
#: raw angles — a cross-condition comparison of them is not meaningful, so they get
#: no premade analysis (the columns still live in the table for manual use).
PRUNE_LEAVES = frozenset({
    "x_um", "y_um", "centroid_x_um", "centroid_y_um",
    "vx_um_per_s", "vy_um_per_s", "orientation",
})
#: Full descriptor names dropped as redundant (duplicated by another family's
#: column — ``shape_relational`` re-exports the cell/nucleus areas).
PRUNE_COLUMNS = frozenset({
    "shape_relational.cell_area_um2", "shape_relational.nucleus_area_um2",
})

#: Quantity-family display names, in emission/grouping order. An unknown family is
#: title-cased and sorted after the known ones.
FAMILY_DISPLAY: dict[str, str] = {
    "cell_shape": "Cell shape",
    "cell_dynamics": "Cell motion",
    "nucleus_shape": "Nucleus shape",
    "nucleus_dynamics": "Nucleus motion",
    "shape_relational": "Nucleus–cell shape",
    "neighbor_count": "Neighbors",
}

#: Axis display names used in titles.
AXIS_DISPLAY = {"condition": "condition", "class_label": "class"}

#: Unit suffixes stripped from a descriptor leaf when building its display label
#: (the unit is then re-appended in parentheses from the schema).
_UNIT_SUFFIX_STRIP = ("_um2", "_um", "_per_s")


def build_analyses(df: pd.DataFrame, schema: dict, object_key: str) -> list[dict]:
    """All premade analyses for *df*, ordered/titled by quantity family.

    *object_key* is the table's finest object level (e.g. ``cell_id``); the spine
    is ``date → position_id → object_key → frame`` restricted to present columns.
    """
    spine = [c for c in (REPLICATE, "position_id", object_key, "frame") if c in df.columns]
    units = {c["name"]: c.get("unit") for c in schema["columns"]}
    descriptors = _family_sorted(d for d in numeric_descriptors(schema) if _keep(d))

    analyses: list[dict] = []
    for descriptor in descriptors:
        for axis in COMPARISON_AXES:
            if axis not in df.columns:
                continue
            # A test needs ≥2 groups; a single-level axis renders describe-only
            # (Iris raises 422 for an unpinned test over one group).
            describe_only = int(df[axis].nunique(dropna=True)) < 2
            analyses.append(
                _superplot(axis, descriptor, spine, object_key,
                           describe_only, units.get(descriptor))
            )
    return analyses


def _superplot(axis: str, descriptor: str, spine: list[str], object_key: str,
               describe_only: bool, unit: str | None) -> dict:
    layers = [
        # The per-object distribution (handles large n without overconfident CIs).
        {"geom": "violin", "level": object_key, "params": {}},
        # The per-date replicate means — the only points drawn, as a beeswarm of
        # distinct-shaped markers (the SuperPlot's honest n).
        {"geom": "dot", "level": REPLICATE, "params": {"layout": "swarm"}},
    ]
    stats = {"chosen_by": "describe_only"} if describe_only else {}
    return {
        "spec_version": SPEC_VERSION,
        "id": f"superplot__{axis}__{descriptor}",
        "title": _title(axis, descriptor, unit),
        # Iris encodings are objects ({"column": name}), not bare strings; an
        # unmapped channel is None. The violin takes the group colour; the date
        # dots are coloured + shaped per date (date is an identifier, so it does
        # not dodge the violin).
        "encodings": {
            "x": {"column": axis}, "y": {"column": descriptor},
            "color": {"column": REPLICATE}, "shape": {"column": REPLICATE},
            "size": None,
        },
        "facet": {"row": None, "col": None, "share_x": True, "share_y": True},
        "hierarchy": {"spine": spine, "fn": {}},
        "layers": layers,
        "stats": stats,
    }


def _keep(name: str) -> bool:
    return name.split(".")[-1] not in PRUNE_LEAVES and name not in PRUNE_COLUMNS


def _family(name: str) -> str:
    return name.split(".")[0] if "." in name else ""


def _family_display(family: str) -> str:
    if family in FAMILY_DISPLAY:
        return FAMILY_DISPLAY[family]
    return family.replace("_", " ").capitalize() if family else "Other"


def _family_sorted(descriptors) -> list[str]:
    """Descriptors ordered by family (known families first, in declared order),
    preserving the original order within a family — so the flat analysis list
    clusters into family groups."""
    rank = {family: i for i, family in enumerate(FAMILY_DISPLAY)}
    indexed = list(enumerate(descriptors))
    return [d for _, d in sorted(
        indexed, key=lambda iv: (rank.get(_family(iv[1]), len(rank)), iv[0]))]


def _title(axis: str, descriptor: str, unit: str | None) -> str:
    return (f"{_family_display(_family(descriptor))} · "
            f"{_descriptor_label(descriptor, unit)} — by {AXIS_DISPLAY.get(axis, axis)}")


def _descriptor_label(name: str, unit: str | None) -> str:
    leaf = name.split(".")[-1]
    # Strip every trailing unit token (``speed_um_per_s`` → ``speed``), not just one.
    stripped = True
    while stripped:
        stripped = False
        for suffix in _UNIT_SUFFIX_STRIP:
            if leaf.endswith(suffix):
                leaf = leaf[: -len(suffix)]
                stripped = True
                break
    human = leaf.replace("_", " ").strip().capitalize() or name.split(".")[-1]
    return f"{human} ({unit})" if unit else human
