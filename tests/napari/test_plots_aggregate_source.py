"""The plot data-source flip: PoolPlots read the persisted aggregated table.

These exercise ``pool_from_aggregate`` (the seam ``PoolPlot.pool`` uses) without a
viewer — it is headless. Artifacts are written directly at each quantifier's
output path so the heavy compute never runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cellflow.aggregate_quantification.records import output_for_record
from cellflow.aggregate_quantification.shape_tables import aggregate, catalogue_root
from cellflow.aggregate_quantification.quantifiers.cell_shape import CellShapeQuantifier
from cellflow.aggregate_quantification.quantifiers.neighbor_count import (
    NeighborCountQuantifier,
)
from cellflow.napari.aggregate_quantification.plots._pooling import pool_from_aggregate


def _record(tmp, pid, condition="ctrl", date="d1"):
    pdir = tmp / condition / pid
    pdir.mkdir(parents=True, exist_ok=True)
    return {
        "id": pid,
        "condition": condition,
        "date": date,
        "position_path": pdir,
        "cell_tracked_labels_path": pdir / "cells.tif",
        "contact_analysis_path": pdir / "aggregate_quantification" / "contact_analysis.h5",
    }


def _write(quantifier, record, table):
    out = output_for_record(quantifier, record)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(out, index=False)


def _cell_shape(cell_ids, frames):
    rows = [
        {"frame": fr, "cell_id": c, "area_um2": 10.0 + c + fr}
        for c in cell_ids
        for fr in frames
    ]
    return {k: np.asarray([r[k] for r in rows]) for k in ("frame", "cell_id", "area_um2")}


def _neighbors(cell_ids, frames):
    rows = [
        {"frame": fr, "cell_id": c, "n_neighbors": c + 1}
        for c in cell_ids
        for fr in frames
    ]
    return {k: np.asarray([r[k] for r in rows]) for k in ("frame", "cell_id", "n_neighbors")}


def test_pool_returns_bare_columns_for_its_quantity(tmp_path):
    cs, nc = CellShapeQuantifier(), NeighborCountQuantifier()
    rec = _record(tmp_path, "a")
    _write(cs, rec, _cell_shape([1, 2], [0]))
    _write(nc, rec, _neighbors([1, 2], [0]))
    aggregate([rec], catalogue_root([rec]))

    df = pool_from_aggregate("cell_shape", [rec])
    # The quantity's value column is bare (prefix stripped); the co-targeting
    # quantity's columns are absent from this product's frame.
    assert "area_um2" in df.columns
    assert not any(c.startswith("neighbor_count") for c in df.columns)
    assert "n_neighbors" not in df.columns
    # Keys + metadata + class_label rode through.
    assert {"frame", "cell_id", "condition", "date", "position_id", "class_label"} <= set(
        df.columns
    )


def test_pool_reads_the_persisted_table_not_the_artifact(tmp_path):
    cs = CellShapeQuantifier()
    rec = _record(tmp_path, "a")
    _write(cs, rec, _cell_shape([1, 2], [0, 1]))
    aggregate([rec], catalogue_root([rec]))

    # Delete the per-position artifact: a live pool would now find nothing, but the
    # persisted aggregated CSV still carries the rows.
    output_for_record(cs, rec).unlink()
    df = pool_from_aggregate("cell_shape", [rec])
    assert len(df) == 4
    assert df["area_um2"].notna().all()


def test_pool_drops_rows_contributed_only_by_cotargeting_quantity(tmp_path):
    cs, nc = CellShapeQuantifier(), NeighborCountQuantifier()
    rec = _record(tmp_path, "a")
    # neighbor_count covers cells 1,2,3; cell_shape only 1,2 → row for cell 3 is
    # contributed solely by neighbor_count and must not appear in cell_shape's pool.
    _write(cs, rec, _cell_shape([1, 2], [0]))
    _write(nc, rec, _neighbors([1, 2, 3], [0]))
    aggregate([rec], catalogue_root([rec]))

    df = pool_from_aggregate("cell_shape", [rec])
    assert set(df["cell_id"]) == {1, 2}
    assert df["area_um2"].notna().all()


def test_pool_falls_back_to_in_memory_when_not_aggregated(tmp_path):
    cs = CellShapeQuantifier()
    rec = _record(tmp_path, "a")
    _write(cs, rec, _cell_shape([1], [0]))
    # No aggregate() call → no CSV; pool still works (in-memory build).
    df = pool_from_aggregate("cell_shape", [rec])
    assert len(df) == 1
    assert "area_um2" in df.columns
