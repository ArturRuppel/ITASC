"""Cell-shape compute core: per-cell, per-frame morphology descriptors.

The headless backend for the ``cell_shape`` quantifier. :func:`build_cell_shape`
runs :func:`skimage.measure.regionprops` over a tracked cell-label stack and
persists a tidy ``shape/table`` to ``cell_shape.h5``; :func:`read_cell_shape`
parses it back as a column-major dict. No Qt / napari import, so scripts and the
standalone wheel can use it.
"""

from .build import DESCRIPTOR_COLUMNS, build_cell_shape, read_cell_shape

__all__ = ["DESCRIPTOR_COLUMNS", "build_cell_shape", "read_cell_shape"]
