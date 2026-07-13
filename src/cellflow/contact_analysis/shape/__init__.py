"""Shape compute core: per-object morphology + relational nucleus-vs-cell shape.

The headless backend for the shape quantifiers. :func:`compute_object_shape` runs
:func:`skimage.measure.regionprops` over any tracked-label stack (cell *or*
nucleus) and returns a tidy column-major table; :func:`compute_relational_table`
pairs each nucleus with its cell and emits relational quantities. Both compute in
memory — the aggregate stage pools the tables, nothing is persisted per position.
No Qt / napari import, so scripts and the standalone wheel can use it.
"""

from .core import DESCRIPTOR_COLUMNS, compute_object_shape
from .relational import RELATIONAL_COLUMNS, compute_relational_table

__all__ = [
    "DESCRIPTOR_COLUMNS",
    "RELATIONAL_COLUMNS",
    "compute_object_shape",
    "compute_relational_table",
]
