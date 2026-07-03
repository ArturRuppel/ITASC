"""Cell-density quantifier: counts off the cell labels, requires a FOV area.

Cell density counts unique non-zero labels per frame straight from the tracked
cell-label TIFF, so it needs no ``contact_analysis.h5``. The field-of-view area is
a required build param (no silent image-area fallback). One ``all`` total row per
frame (label-agnostic). Backend-only — no Qt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from cellflow.contact_analysis.quantifier import (
    PositionInputs,
    available_quantifiers,
)


def _quantifier():
    return {c.quantity_id: c for c in available_quantifiers()}["cell_density"]()


def _labels_stack() -> np.ndarray:
    # One frame, four cell labels (1..4) on a background of 0.
    frame = np.zeros((10, 10), dtype=np.uint16)
    frame[0:5, 0:5] = 1
    frame[0:5, 5:10] = 2
    frame[5:10, 0:5] = 3
    frame[5:10, 5:10] = 4
    return frame[np.newaxis, ...]


def _inputs(tmp_path: Path) -> PositionInputs:
    pos = tmp_path / "p1"
    pos.mkdir()
    cell_path = pos / "cells.tif"
    tifffile.imwrite(cell_path, _labels_stack())
    return PositionInputs(position_dir=pos, cell_labels_path=cell_path)


def _build(inputs: PositionInputs, params: dict | None = None):
    q = _quantifier()
    out = q.default_output(inputs)
    q.build(inputs, out, params=params)
    return q.object_table(out)


def test_requires_cell_labels_not_contacts():
    q = _quantifier()
    assert q.requires == ("cell_labels_path",)


def test_counts_cells_from_labels_over_fov(tmp_path):
    table = _build(_inputs(tmp_path), params={"fov_area_mm2": 2.0})
    rows = dict(zip(table["label"].tolist(), table["density"].tolist()))
    counts = dict(zip(table["label"].tolist(), table["n_cells"].tolist()))
    # 4 cell labels over a 2 mm² field of view → 2 cells/mm².
    assert counts["all"] == 4
    assert rows["all"] == 2.0


def test_all_row_only_label_agnostic(tmp_path):
    table = _build(_inputs(tmp_path), params={"fov_area_mm2": 1.0})
    assert table["label"].tolist() == ["all"]


def test_missing_fov_raises_clear_error(tmp_path):
    q = _quantifier()
    inputs = _inputs(tmp_path)
    with pytest.raises(ValueError, match="field-of-view"):
        q.build(inputs, q.default_output(inputs), params={})
