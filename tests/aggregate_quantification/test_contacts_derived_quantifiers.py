"""The contacts-derived quantifiers: build → persist → object_table round-trips.

These label-agnostic quantities (neighbor count / signed contact length) read a
position's ``contact_analysis.h5`` and persist a tidy CSV its plot later pools. The
compute itself is covered by ``test_neighborhood.py`` /
``test_signed_contact_length.py``; here we assert the quantifier wrapper persists
and reads back the right columns/dtypes and honours its build params. Backend-only
— no Qt.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from cellflow.aggregate_quantification.quantifier import (
    PositionInputs,
    available_quantifiers,
)

_DERIVED = (
    "neighbor_count",
    "signed_contact_length",
)


def _quantifier(quantity_id: str):
    return {c.quantity_id: c for c in available_quantifiers()}[quantity_id]()


def _write_contacts_h5(path: Path) -> None:
    with h5py.File(path, "w") as h5:
        prov = h5.create_group("provenance")
        prov.attrs["cell_tracked_labels_path"] = "cells.tif"
        prov.attrs["nucleus_tracked_labels_path"] = "nuclei.tif"
        cells = h5.create_group("cells/table")
        cells.create_dataset("frame", data=np.array([0, 0, 0, 0], dtype=np.int64))
        cells.create_dataset("cell_id", data=np.array([1, 2, 3, 4], dtype=np.int64))
        edges = h5.create_group("edges/table")
        edges.create_dataset("frame", data=np.array([0, 0], dtype=np.int64))
        edges.create_dataset("cell_a", data=np.array([1, 3], dtype=np.int64))
        edges.create_dataset("cell_b", data=np.array([2, 4], dtype=np.int64))
        edges.create_dataset("kind", data=np.array(["cell_cell"] * 2, dtype=object))
        edges.create_dataset("length", data=np.ones(2, dtype=float))
        h5.create_group("t1_events/table")
        h5.create_dataset("edges/coordinates/y", data=np.empty(0))
        h5.create_dataset("edges/coordinates/x", data=np.empty(0))


def _inputs(tmp_path: Path) -> PositionInputs:
    pos = tmp_path / "p1"
    pos.mkdir()
    h5_path = pos / "contact_analysis.h5"
    _write_contacts_h5(h5_path)
    return PositionInputs(
        position_dir=pos, contact_analysis_path=h5_path, pixel_size_um=1.0
    )


def _build(quantity_id: str, inputs: PositionInputs, params: dict | None = None):
    q = _quantifier(quantity_id)
    out = q.default_output(inputs)
    q.build(inputs, out, params=params)
    return q.object_table(out)


@pytest.mark.parametrize("quantity_id", _DERIVED)
def test_derived_quantifier_persists_and_reads_back(quantity_id, tmp_path):
    table = _build(quantity_id, _inputs(tmp_path))
    # frame / *_id keys round-trip as integers; the read survives string columns.
    assert "frame" in table
    assert table["frame"].dtype.kind == "i"


def test_neighbor_count_columns_and_degree(tmp_path):
    table = _build("neighbor_count", _inputs(tmp_path))
    assert {"frame", "cell_id", "n_neighbors"} <= set(table)
    # Each of the 4 cells has exactly one neighbor in the fixture.
    assert table["n_neighbors"].tolist() == [1, 1, 1, 1]


def test_signed_contact_length_empty_without_t1(tmp_path):
    # The fixture carries no T1 events → an empty (but valid) length table.
    table = _build("signed_contact_length", _inputs(tmp_path))
    assert "signed_length" in table
    assert len(table["signed_length"]) == 0


def test_missing_contacts_artifact_raises(tmp_path):
    q = _quantifier("neighbor_count")
    inputs = PositionInputs(
        position_dir=tmp_path, contact_analysis_path=tmp_path / "missing.h5"
    )
    with pytest.raises(FileNotFoundError):
        q.build(inputs, q.default_output(inputs))
