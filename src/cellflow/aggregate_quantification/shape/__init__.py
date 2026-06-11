"""Shape compute core: per-object morphology + relational nucleus-vs-cell shape.

The headless backend for the shape quantifiers. :func:`build_object_shape` runs
:func:`skimage.measure.regionprops` over any tracked-label stack (cell *or*
nucleus) and persists a tidy CSV; :func:`build_relational` pairs each nucleus
with its cell and emits relational quantities. Each table reads back via
:func:`read_shape_table` / :func:`read_relational_table` as a column-major dict.
No Qt / napari import, so scripts and the standalone wheel can use it.
"""

from .core import (
    DESCRIPTOR_COLUMNS,
    build_object_shape,
    provenance_path,
    read_shape_table,
)
from .relational import RELATIONAL_COLUMNS, build_relational, read_relational_table

__all__ = [
    "DESCRIPTOR_COLUMNS",
    "RELATIONAL_COLUMNS",
    "build_object_shape",
    "build_relational",
    "provenance_path",
    "read_relational_table",
    "read_shape_table",
]
