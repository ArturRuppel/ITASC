"""Export CellFlow aggregate tables as Iris ``.iris`` analysis bundles.

Backend-only (no Qt / napari). Each tidy table becomes one ``.iris`` document
carrying the table plus a comprehensive set of premade SuperPlot analyses (a
swarm + notched box of the objects coloured by date, with the per-date replicate
means overlaid by shape, and the inferential test pinned to the date level).

Export (writing ``.iris``) is coupled to Iris **only through the file format**
(v1.0) and never imports the Iris engine, so the standalone ``cellflow-aggregate``
wheel stays free of that dependency. The optional :mod:`.figures` helper *renders*
those bundles to static figures and does need the engine — but it imports it
lazily, at call time, so merely importing this package never pulls it in. See
``docs/superpowers/specs/2026-06-17-iris-export-design.md``.
"""

from .analyses import build_analyses
from .document import write_iris
from .export import export_dir, export_table
from .figures import render_export_dir, render_iris
from .schema import infer_schema

__all__ = [
    "build_analyses",
    "export_dir",
    "export_table",
    "infer_schema",
    "render_export_dir",
    "render_iris",
    "write_iris",
]
