"""Aggregated shape tables: per-quantity pooling, NLS join, materialized views."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cellflow.aggregate_quantification.records import output_for_record
from cellflow.aggregate_quantification.shape_tables import (
    aggregate,
    build_table,
    catalogue_root,
    read_table,
    shape_table_registry,
    table_for_quantity,
    table_path,
)
from cellflow.aggregate_quantification.quantifiers.cell_shape import CellShapeQuantifier
from cellflow.aggregate_quantification.quantifiers.neighbor_count import (
    NeighborCountQuantifier,
)


def _record(
    tmp: Path, pid: str, condition: str = "ctrl", date: str = "d1",
    experiment_id: str | None = None,
) -> dict:
    pdir = tmp / condition / pid
    pdir.mkdir(parents=True, exist_ok=True)
    return {
        "id": pid,
        "condition": condition,
        "date": date,
        "experiment_id": experiment_id if experiment_id is not None else date,
        "position_path": pdir,
        "cell_tracked_labels_path": pdir / "cells.tif",
        "contact_analysis_path": pdir / "aggregate_quantification" / "contact_analysis.h5",
    }


def _write_object_table(quantifier, record: dict, table: dict) -> None:
    """Write *table* as the quantifier's on-disk artifact so ``object_table`` /
    ``is_built`` see a built product without running the heavy compute."""
    out = output_for_record(quantifier, record)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(out, index=False)


def _cell_shape_table(cell_ids, frames, area0=10.0):
    rows = []
    for cid in cell_ids:
        for fr in frames:
            rows.append({"frame": fr, "cell_id": cid, "area_um2": area0 + cid + fr})
    return {k: np.asarray([r[k] for r in rows]) for k in ("frame", "cell_id", "area_um2")}


def _neighbor_count_table(cell_ids, frames):
    rows = []
    for cid in cell_ids:
        for fr in frames:
            rows.append({"frame": fr, "cell_id": cid, "n_neighbors": cid + 1})
    return {k: np.asarray([r[k] for r in rows]) for k in ("frame", "cell_id", "n_neighbors")}


# --------------------------------------------------------------------- pooling


def test_experiment_id_broadcast_onto_pooled_rows(tmp_path):
    """The catalogue's experiment_id (paired-replicate key) is stamped onto rows,
    distinct from date."""
    cs = CellShapeQuantifier()
    rec = _record(tmp_path, "a", date="2026-05-09", experiment_id="EXP-01")
    _write_object_table(cs, rec, _cell_shape_table([1, 2], [0]))

    df = build_table("cell_shape", [rec])

    assert "experiment_id" in df.columns
    assert (df["experiment_id"] == "EXP-01").all()
    assert (df["date"] == "2026-05-09").all()


def test_build_table_assigns_deterministic_row_id(tmp_path):
    """Each row gets a stable ``id`` derived from its identity — the join key the
    curation artifact references, so it must survive a regeneration unchanged."""
    cs = CellShapeQuantifier()
    rec = _record(tmp_path, "a", experiment_id="EXP-01")
    _write_object_table(cs, rec, _cell_shape_table([1, 2], [0]))

    df1 = build_table("cell_shape", [rec])
    df2 = build_table("cell_shape", [rec])

    assert "id" in df1.columns
    assert list(df1["id"]) == list(df2["id"])  # deterministic, not a row counter
    assert df1["id"].is_unique


def test_row_id_distinguishes_same_cell_across_positions(tmp_path):
    """cell_id/frame alone do not identify a row: position + condition matter."""
    cs = CellShapeQuantifier()
    rec_a = _record(tmp_path, "a", condition="ctrl", experiment_id="EXP-01")
    rec_b = _record(tmp_path, "b", condition="drug", experiment_id="EXP-01")
    _write_object_table(cs, rec_a, _cell_shape_table([1], [0]))
    _write_object_table(cs, rec_b, _cell_shape_table([1], [0]))

    df = build_table("cell_shape", [rec_a, rec_b])

    assert df["id"].nunique() == 2


def test_two_positions_pool_into_one_table(tmp_path):
    cs = CellShapeQuantifier()
    rec_a = _record(tmp_path, "a")
    rec_b = _record(tmp_path, "b")
    _write_object_table(cs, rec_a, _cell_shape_table([1, 2], [0, 1]))
    _write_object_table(cs, rec_b, _cell_shape_table([1], [0]))

    df = build_table("cell_shape", [rec_a, rec_b])

    assert set(df["position_id"]) == {"a", "b"}
    assert {"condition", "date", "position_id", "frame", "cell_id"} <= set(df.columns)
    # Value columns are namespaced by quantity_id so cross-quantity names never clash.
    assert "cell_shape.area_um2" in df.columns
    assert len(df) == 4 + 1  # a: 2 cells × 2 frames, b: 1 cell × 1 frame


def test_each_quantity_is_its_own_table(tmp_path):
    """Two quantities at the same grain land in separate per-quantifier tables,
    not one shared (god) table."""
    cs, nc = CellShapeQuantifier(), NeighborCountQuantifier()
    rec = _record(tmp_path, "a")
    _write_object_table(cs, rec, _cell_shape_table([1, 2], [0]))
    _write_object_table(nc, rec, _neighbor_count_table([1, 2], [0]))

    shape = build_table("cell_shape", [rec])
    neighbors = build_table("neighbor_count", [rec])

    assert "cell_shape.area_um2" in shape.columns
    assert "neighbor_count.n_neighbors" not in shape.columns
    assert "neighbor_count.n_neighbors" in neighbors.columns
    assert "cell_shape.area_um2" not in neighbors.columns


def test_quantity_built_for_only_some_positions(tmp_path):
    """A quantity's table carries only the positions that built it; an unbuilt
    position simply contributes no rows (no cross-quantity NaN padding)."""
    nc = NeighborCountQuantifier()
    rec_a = _record(tmp_path, "a")
    rec_b = _record(tmp_path, "b")
    _write_object_table(nc, rec_a, _neighbor_count_table([1], [0]))  # b never built

    df = build_table("neighbor_count", [rec_a, rec_b])

    assert set(df["position_id"]) == {"a"}
    assert "neighbor_count.n_neighbors" in df.columns


def test_ids_do_not_collide_across_positions(tmp_path):
    cs = CellShapeQuantifier()
    rec_a = _record(tmp_path, "a")
    rec_b = _record(tmp_path, "b")
    _write_object_table(cs, rec_a, _cell_shape_table([1], [0], area0=100))
    _write_object_table(cs, rec_b, _cell_shape_table([1], [0], area0=200))

    df = build_table("cell_shape", [rec_a, rec_b])
    # Same cell_id=1 in both positions stays two distinct rows (metadata differs).
    rows = df[df["cell_id"] == 1]
    assert len(rows) == 2
    assert set(rows["position_id"]) == {"a", "b"}


# ----------------------------------------------------------- label-agnostic


def test_pooled_table_has_no_class_label(tmp_path):
    """The pooled tables are label-agnostic: no ``class_label`` is joined, even on a
    cell-keyed table. A downstream consumer joins a classification itself."""
    cs = CellShapeQuantifier()
    rec = _record(tmp_path, "a")
    _write_object_table(cs, rec, _cell_shape_table([1, 2], [0]))

    df = build_table("cell_shape", [rec])

    assert "class_label" not in df.columns


# --------------------------------------------------------- materialized views


def test_aggregate_writes_csv_and_rewrites_whole(tmp_path):
    cs = CellShapeQuantifier()
    rec_a = _record(tmp_path, "a")
    rec_b = _record(tmp_path, "b")
    _write_object_table(cs, rec_a, _cell_shape_table([1], [0]))
    _write_object_table(cs, rec_b, _cell_shape_table([1], [0]))
    out_dir = tmp_path / "catalogue"

    written = aggregate([rec_a, rec_b], out_dir)
    path = written["cell_shape"]
    assert path == table_path(out_dir, "cell_shape")
    assert set(read_table(path)["position_id"]) == {"a", "b"}

    # Re-aggregating a narrower scope rewrites the file whole (b's rows are gone).
    aggregate([rec_a], out_dir)
    assert set(read_table(path)["position_id"]) == {"a"}


def test_aggregate_skips_empty_tables(tmp_path):
    rec = _record(tmp_path, "a")  # nothing built
    written = aggregate([rec], tmp_path / "catalogue")
    assert written == {}


def test_read_table_restores_integer_keys(tmp_path):
    cs = CellShapeQuantifier()
    rec = _record(tmp_path, "a")
    _write_object_table(cs, rec, _cell_shape_table([1, 2], [0, 1]))
    written = aggregate([rec], tmp_path / "catalogue")
    df = read_table(written["cell_shape"])
    assert df["frame"].dtype == np.int64
    assert df["cell_id"].dtype == np.int64


# ------------------------------------------------------------------- registry


def test_registry_is_one_table_per_quantity():
    reg = shape_table_registry()
    # Each aggregating quantity is its own table, keyed by its own grain.
    assert reg["cell_shape"].keys == ("frame", "cell_id")
    assert reg["cell_shape"].quantity_ids == ("cell_shape",)
    assert reg["neighbor_count"].keys == ("frame", "cell_id")
    # No god table pooling several quantities under one name.
    assert "cells_by_frame" not in reg
    assert table_for_quantity("cell_shape") == "cell_shape"
    assert table_for_quantity("contacts") is None  # not aggregated


def test_catalogue_root_is_common_ancestor(tmp_path):
    rec_a = _record(tmp_path, "a", condition="ctrl")
    rec_b = _record(tmp_path, "b", condition="drug")
    root = catalogue_root([rec_a, rec_b])
    assert Path(rec_a["position_path"]).is_relative_to(root)
    assert Path(rec_b["position_path"]).is_relative_to(root)
