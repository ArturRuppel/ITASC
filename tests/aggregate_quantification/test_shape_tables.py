"""Aggregated shape tables: pooling, outer-join, NLS join, materialized views."""
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


def _write_nls(record: dict, labels: dict[int, str]) -> None:
    out = Path(record["position_path"]) / "aggregate_quantification" / "nls_classification.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": list(labels), "label": list(labels.values())}).to_csv(out, index=False)


# --------------------------------------------------------------------- pooling


def test_experiment_id_broadcast_onto_pooled_rows(tmp_path):
    """The catalogue's experiment_id (paired-replicate key) is stamped onto rows,
    distinct from date."""
    cs = CellShapeQuantifier()
    rec = _record(tmp_path, "a", date="2026-05-09", experiment_id="EXP-01")
    _write_object_table(cs, rec, _cell_shape_table([1, 2], [0]))

    df = build_table("cells_by_frame", [rec])

    assert "experiment_id" in df.columns
    assert (df["experiment_id"] == "EXP-01").all()
    assert (df["date"] == "2026-05-09").all()


def test_two_positions_pool_into_one_table(tmp_path):
    cs = CellShapeQuantifier()
    rec_a = _record(tmp_path, "a")
    rec_b = _record(tmp_path, "b")
    _write_object_table(cs, rec_a, _cell_shape_table([1, 2], [0, 1]))
    _write_object_table(cs, rec_b, _cell_shape_table([1], [0]))

    df = build_table("cells_by_frame", [rec_a, rec_b])

    assert set(df["position_id"]) == {"a", "b"}
    assert {"condition", "date", "position_id", "frame", "cell_id"} <= set(df.columns)
    # Value columns are namespaced by quantity_id so cross-quantity names never clash.
    assert "cell_shape.area_um2" in df.columns
    assert len(df) == 4 + 1  # a: 2 cells × 2 frames, b: 1 cell × 1 frame


def test_multiple_quantities_outer_join_on_keys(tmp_path):
    cs, nc = CellShapeQuantifier(), NeighborCountQuantifier()
    rec = _record(tmp_path, "a")
    _write_object_table(cs, rec, _cell_shape_table([1, 2], [0]))
    _write_object_table(nc, rec, _neighbor_count_table([1, 2], [0]))

    df = build_table("cells_by_frame", [rec])

    assert "cell_shape.area_um2" in df.columns
    assert "neighbor_count.n_neighbors" in df.columns
    assert len(df) == 2  # joined on (frame, cell_id), not stacked
    assert df["neighbor_count.n_neighbors"].notna().all()


def test_position_missing_a_quantity_yields_nan_not_error(tmp_path):
    cs, nc = CellShapeQuantifier(), NeighborCountQuantifier()
    rec_a = _record(tmp_path, "a")
    rec_b = _record(tmp_path, "b")
    _write_object_table(cs, rec_a, _cell_shape_table([1], [0]))
    _write_object_table(nc, rec_a, _neighbor_count_table([1], [0]))
    _write_object_table(cs, rec_b, _cell_shape_table([1], [0]))  # b has no neighbor_count

    df = build_table("cells_by_frame", [rec_a, rec_b])

    b_rows = df[df["position_id"] == "b"]
    assert b_rows["neighbor_count.n_neighbors"].isna().all()
    assert b_rows["cell_shape.area_um2"].notna().all()


def test_ids_do_not_collide_across_positions(tmp_path):
    cs = CellShapeQuantifier()
    rec_a = _record(tmp_path, "a")
    rec_b = _record(tmp_path, "b")
    _write_object_table(cs, rec_a, _cell_shape_table([1], [0], area0=100))
    _write_object_table(cs, rec_b, _cell_shape_table([1], [0], area0=200))

    df = build_table("cells_by_frame", [rec_a, rec_b])
    # Same cell_id=1 in both positions stays two distinct rows (metadata differs).
    rows = df[df["cell_id"] == 1]
    assert len(rows) == 2
    assert set(rows["position_id"]) == {"a", "b"}


# ------------------------------------------------------------------- NLS join


def test_nls_class_label_joined_by_cell_id(tmp_path):
    cs = CellShapeQuantifier()
    rec_a = _record(tmp_path, "a")
    rec_b = _record(tmp_path, "b")
    _write_object_table(cs, rec_a, _cell_shape_table([1, 2], [0]))
    _write_object_table(cs, rec_b, _cell_shape_table([1], [0]))
    _write_nls(rec_a, {1: "epithelial", 2: "mesenchymal"})  # b left unclassified

    df = build_table("cells_by_frame", [rec_a, rec_b])

    a = df[df["position_id"] == "a"].set_index("cell_id")["class_label"]
    assert a.loc[1] == "epithelial" and a.loc[2] == "mesenchymal"
    assert (df[df["position_id"] == "b"]["class_label"] == "unclassified").all()


# --------------------------------------------------------- materialized views


def test_aggregate_writes_csv_and_rewrites_whole(tmp_path):
    cs = CellShapeQuantifier()
    rec_a = _record(tmp_path, "a")
    rec_b = _record(tmp_path, "b")
    _write_object_table(cs, rec_a, _cell_shape_table([1], [0]))
    _write_object_table(cs, rec_b, _cell_shape_table([1], [0]))
    out_dir = tmp_path / "catalogue"

    written = aggregate([rec_a, rec_b], out_dir)
    path = written["cells_by_frame"]
    assert path == table_path(out_dir, "cells_by_frame")
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
    df = read_table(written["cells_by_frame"])
    assert df["frame"].dtype == np.int64
    assert df["cell_id"].dtype == np.int64


# ------------------------------------------------------------------- registry


def test_registry_and_table_for_quantity():
    reg = shape_table_registry()
    assert reg["cells_by_frame"].keys == ("frame", "cell_id")
    assert "cell_shape" in reg["cells_by_frame"].quantity_ids
    assert table_for_quantity("cell_shape") == "cells_by_frame"
    assert table_for_quantity("contacts") is None  # not aggregated


def test_catalogue_root_is_common_ancestor(tmp_path):
    rec_a = _record(tmp_path, "a", condition="ctrl")
    rec_b = _record(tmp_path, "b", condition="drug")
    root = catalogue_root([rec_a, rec_b])
    assert Path(rec_a["position_path"]).is_relative_to(root)
    assert Path(rec_b["position_path"]).is_relative_to(root)
