"""Export CellFlow aggregate tables as Iris ``.iris`` analysis bundles.

Backend-only (no Qt / napari). Each tidy table becomes one ``.iris`` document
carrying the table plus a comprehensive set of premade SuperPlot analyses (a
swarm + notched box of the objects coloured by date, with the per-date replicate
means overlaid by shape, and the inferential test pinned to the date level).

This package is coupled to Iris **only through the ``.iris`` file format** (v1.0);
it never imports the Iris engine, so the standalone ``cellflow-aggregate`` wheel
stays free of that dependency. See
``docs/superpowers/specs/2026-06-17-iris-export-design.md``.
"""

from .analyses import build_analyses
from .document import write_iris
from .export import export_dir, export_table
from .schema import infer_schema

__all__ = [
    "build_analyses",
    "export_dir",
    "export_table",
    "infer_schema",
    "write_iris",
]
