"""Cell-density quantifier: counts off the cell labels, requires a FOV area.

Cell density moved off the contacts path — it counts unique non-zero labels per
frame straight from the tracked cell-label TIFF, so it needs no
``contact_analysis.h5``. The field-of-view area is a required build param (no
silent image-area fallback). Backend-only — no Qt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from cellflow.aggregate_quantification.contacts.nls_classification import (
    nls_classification_csv_path,
    write_nls_classification_csv,
)
from cellflow.aggregate_quantification.quantifier import (
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


def _inputs(tmp_path: Path, *, classify: bool = False) -> PositionInputs:
    pos = tmp_path / "p1"
    pos.mkdir()
    cell_path = pos / "cells.tif"
    tifffile.imwrite(cell_path, _labels_stack())
    if classify:
        write_nls_classification_csv(
            nls_classification_csv_path(pos),
            {1: "positive", 2: "positive", 3: "negative", 4: "negative"},
            positive_label="positive",
            negative_label="negative",
        )
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


def test_per_class_breakdown_when_classified(tmp_path):
    table = _build(_inputs(tmp_path, classify=True), params={"fov_area_mm2": 4.0})
    counts = dict(zip(table["label"].tolist(), table["n_cells"].tolist()))
    assert counts["all"] == 4
    assert counts["positive"] == 2
    assert counts["negative"] == 2


def test_all_row_only_without_classification(tmp_path):
    table = _build(_inputs(tmp_path), params={"fov_area_mm2": 1.0})
    assert table["label"].tolist() == ["all"]


def test_missing_fov_raises_clear_error(tmp_path):
    q = _quantifier()
    inputs = _inputs(tmp_path)
    with pytest.raises(ValueError, match="field-of-view"):
        q.build(inputs, q.default_output(inputs), params={})
