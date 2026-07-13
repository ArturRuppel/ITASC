"""Aggregated shape tables: per-quantity pooling, NLS join, materialized views."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cellflow.contact_analysis.shape_tables import (
    aggregate,
    build_table,
    catalogue_root,
    read_table,
    shape_table_registry,
    table_for_quantity,
    table_path,
)
from cellflow.contact_analysis.quantifiers.cell_shape import CellShapeQuantifier
from cellflow.contact_analysis.quantifiers.neighbor_count import (
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
        "contact_analysis_path": pdir / "4_contact_analysis" / "contact_analysis.h5",
        "pixel_size_um": 0.5,
    }


def _stub_compute(monkeypatch, quantifier_cls, tables_by_pid):
    """Make the pooled quantifier return a fixed table per position folder name."""
    def compute(self, inputs, *, params=None):
        return tables_by_pid.get(Path(inputs.position_dir).name)
    monkeypatch.setattr(quantifier_cls, "compute_object_table", compute)


def _mark_contacts_built(record: dict) -> None:
    """Touch *record*'s ``contact_analysis_path`` so a contacts-derived quantifier's
    ``requires`` gate (checked against the file actually existing) is satisfied —
    needed for :class:`NeighborCountQuantifier` even when its compute is stubbed."""
    path = Path(record["contact_analysis_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


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


def test_experiment_id_broadcast_onto_pooled_rows(tmp_path, monkeypatch):
    """The catalogue's experiment_id (paired-replicate key) is stamped onto rows."""
    rec = _record(tmp_path, "a", date="2026-05-09", experiment_id="EXP-01")
    _stub_compute(monkeypatch, CellShapeQuantifier, {"a": _cell_shape_table([1, 2], [0])})

    df = build_table("cell_shape", [rec])

    assert "experiment_id" in df.columns
    assert (df["experiment_id"] == "EXP-01").all()


def test_free_form_column_broadcast_onto_pooled_rows(tmp_path, monkeypatch):
    """A folder-derived / manual free-form column rides onto every pooled row as
    a constant descriptor, without disturbing the recognized axes."""
    rec = _record(tmp_path, "a", condition="WT", date="d1", experiment_id="E1")
    rec["columns"] = {
        "condition": "WT",
        "experiment_id": "E1",
        "date": "d1",
        "position_id": "a",
        "replicate": "r2",
    }
    _stub_compute(monkeypatch, CellShapeQuantifier, {"a": _cell_shape_table([1, 2], [0])})

    df = build_table("cell_shape", [rec])

    assert "replicate" in df.columns
    assert (df["replicate"] == "r2").all()
    # The recognized axis is still stamped exactly once (no duplicate column).
    assert list(df.columns).count("condition") == 1


def test_build_table_assigns_deterministic_row_id(tmp_path, monkeypatch):
    """Each row gets a stable ``id`` derived from its identity — the join key the
    curation artifact references, so it must survive a regeneration unchanged."""
    rec = _record(tmp_path, "a", experiment_id="EXP-01")
    _stub_compute(monkeypatch, CellShapeQuantifier, {"a": _cell_shape_table([1, 2], [0])})

    df1 = build_table("cell_shape", [rec])
    df2 = build_table("cell_shape", [rec])

    assert "id" in df1.columns
    assert list(df1["id"]) == list(df2["id"])  # deterministic, not a row counter
    assert df1["id"].is_unique


def test_row_id_distinguishes_same_cell_across_positions(tmp_path, monkeypatch):
    """cell_id/frame alone do not identify a row: position + condition matter."""
    rec_a = _record(tmp_path, "a", condition="ctrl", experiment_id="EXP-01")
    rec_b = _record(tmp_path, "b", condition="drug", experiment_id="EXP-01")
    _stub_compute(monkeypatch, CellShapeQuantifier, {
        "a": _cell_shape_table([1], [0]),
        "b": _cell_shape_table([1], [0]),
    })

    df = build_table("cell_shape", [rec_a, rec_b])

    assert df["id"].nunique() == 2


def test_two_positions_pool_into_one_table(tmp_path, monkeypatch):
    rec_a = _record(tmp_path, "a")
    rec_b = _record(tmp_path, "b")
    _stub_compute(monkeypatch, CellShapeQuantifier, {
        "a": _cell_shape_table([1, 2], [0, 1]),
        "b": _cell_shape_table([1], [0]),
    })

    df = build_table("cell_shape", [rec_a, rec_b])

    assert set(df["position_id"]) == {"a", "b"}
    assert {"condition", "position_id", "frame", "cell_id"} <= set(df.columns)
    # Value columns are namespaced by quantity_id so cross-quantity names never clash.
    assert "cell_shape.area_um2" in df.columns
    assert len(df) == 4 + 1  # a: 2 cells × 2 frames, b: 1 cell × 1 frame


def test_each_quantity_is_its_own_table(tmp_path, monkeypatch):
    """Two quantities at the same grain land in separate per-quantifier tables,
    not one shared (god) table."""
    rec = _record(tmp_path, "a")
    _mark_contacts_built(rec)
    _stub_compute(monkeypatch, CellShapeQuantifier, {"a": _cell_shape_table([1, 2], [0])})
    _stub_compute(monkeypatch, NeighborCountQuantifier, {"a": _neighbor_count_table([1, 2], [0])})

    shape = build_table("cell_shape", [rec])
    neighbors = build_table("neighbor_count", [rec])

    assert "cell_shape.area_um2" in shape.columns
    assert "neighbor_count.n_neighbors" not in shape.columns
    assert "neighbor_count.n_neighbors" in neighbors.columns
    assert "cell_shape.area_um2" not in neighbors.columns


def test_quantity_built_for_only_some_positions(tmp_path, monkeypatch):
    """A quantity's table carries only the positions that built it; an unbuilt
    position simply contributes no rows (no cross-quantity NaN padding)."""
    rec_a = _record(tmp_path, "a")
    rec_b = _record(tmp_path, "b")
    _mark_contacts_built(rec_a)
    # b never built: its contact_analysis_path stays absent, so its `requires` is
    # never satisfied and it is gated out before compute is even attempted.
    _stub_compute(monkeypatch, NeighborCountQuantifier, {"a": _neighbor_count_table([1], [0])})

    df = build_table("neighbor_count", [rec_a, rec_b])

    assert set(df["position_id"]) == {"a"}
    assert "neighbor_count.n_neighbors" in df.columns


def test_ids_do_not_collide_across_positions(tmp_path, monkeypatch):
    rec_a = _record(tmp_path, "a")
    rec_b = _record(tmp_path, "b")
    _stub_compute(monkeypatch, CellShapeQuantifier, {
        "a": _cell_shape_table([1], [0], area0=100),
        "b": _cell_shape_table([1], [0], area0=200),
    })

    df = build_table("cell_shape", [rec_a, rec_b])
    # Same cell_id=1 in both positions stays two distinct rows (metadata differs).
    rows = df[df["cell_id"] == 1]
    assert len(rows) == 2
    assert set(rows["position_id"]) == {"a", "b"}


def test_pooled_table_has_no_date_column(tmp_path, monkeypatch):
    """``date`` is a catalog-only descriptor now; it is no longer stamped onto the
    pooled tables (removed with the catalog's date column)."""
    rec = _record(tmp_path, "a", date="2026-05-09")
    _stub_compute(monkeypatch, CellShapeQuantifier, {"a": _cell_shape_table([1, 2], [0])})

    df = build_table("cell_shape", [rec])

    assert "date" not in df.columns
    assert {"condition", "experiment_id", "position_id"} <= set(df.columns)


# ----------------------------------------------------------- label-agnostic


def test_pooled_table_has_no_class_label(tmp_path, monkeypatch):
    """The pooled tables are label-agnostic: no ``class_label`` is joined, even on a
    cell-keyed table. A downstream consumer joins a classification itself."""
    rec = _record(tmp_path, "a")
    _stub_compute(monkeypatch, CellShapeQuantifier, {"a": _cell_shape_table([1, 2], [0])})

    df = build_table("cell_shape", [rec])

    assert "class_label" not in df.columns


# --------------------------------------------------------- materialized views


def test_aggregate_writes_csv_and_rewrites_whole(tmp_path, monkeypatch):
    rec_a = _record(tmp_path, "a")
    rec_b = _record(tmp_path, "b")
    _stub_compute(monkeypatch, CellShapeQuantifier, {
        "a": _cell_shape_table([1], [0]),
        "b": _cell_shape_table([1], [0]),
    })
    out_dir = tmp_path / "catalogue"

    written = aggregate([rec_a, rec_b], out_dir)
    path = written["cell_shape"]
    assert path == table_path(out_dir, "cell_shape")
    assert set(read_table(path)["position_id"]) == {"a", "b"}

    # Re-aggregating a narrower scope rewrites the file whole (b's rows are gone).
    aggregate([rec_a], out_dir)
    assert set(read_table(path)["position_id"]) == {"a"}


def test_aggregate_skips_empty_tables(tmp_path):
    rec = _record(tmp_path, "a")
    # Nothing built *and* no real inputs on disk: drop the cell-labels path so
    # every pooled quantifier's `requires` gate fails cleanly instead of
    # attempting a real compute against a file that was never created.
    del rec["cell_tracked_labels_path"]
    written = aggregate([rec], tmp_path / "catalogue")
    assert written == {}


def test_aggregate_writes_run_level_provenance(tmp_path, monkeypatch):
    import json

    from cellflow.contact_analysis.shape_tables import PROVENANCE_NAME

    rec_a = _record(tmp_path, "a", experiment_id="exp1")
    rec_b = _record(tmp_path, "b", experiment_id="exp1")
    _stub_compute(monkeypatch, CellShapeQuantifier, {
        "a": _cell_shape_table([1, 2], [0]),
        "b": _cell_shape_table([1], [0]),
    })
    out_dir = tmp_path / "catalogue"

    aggregate([rec_a, rec_b], out_dir, params={"pixel_size_um": 0.5})

    prov = json.loads((out_dir / PROVENANCE_NAME).read_text())
    assert prov["params"] == {"pixel_size_um": 0.5}
    assert "created_at" in prov and "cellflow_version" in prov
    # Every contributing position is recorded by identity + source paths.
    assert {p["position_id"] for p in prov["positions"]} == {"a", "b"}
    assert {p["experiment_id"] for p in prov["positions"]} == {"exp1"}
    # The quantifier that pooled rows records its columns + row count (a=2, b=1 → 3).
    assert prov["quantifiers"]["cell_shape"]["rows"] == 3
    assert "cell_shape.area_um2" in prov["quantifiers"]["cell_shape"]["columns"]
    # Provenance is seam-uniform: every in-scope pooled quantifier is recorded,
    # including those that pooled to zero rows (here everything but cell_shape,
    # whose inputs these stub records don't supply) — so an empty quantifier is
    # auditable, not silently absent. It records rows: 0 and wrote no CSV.
    assert set(prov["quantifiers"]) == set(shape_table_registry())
    assert prov["quantifiers"]["neighbor_count"] == {"rows": 0, "columns": []}
    assert not table_path(out_dir, "neighbor_count").exists()


def test_aggregate_writes_no_provenance_when_nothing_pooled(tmp_path):
    from cellflow.contact_analysis.shape_tables import PROVENANCE_NAME

    rec = _record(tmp_path, "a")
    del rec["cell_tracked_labels_path"]  # no inputs → no tables → no provenance
    out_dir = tmp_path / "catalogue"
    assert aggregate([rec], out_dir) == {}
    assert not (out_dir / PROVENANCE_NAME).exists()


def test_read_table_restores_integer_keys(tmp_path, monkeypatch):
    rec = _record(tmp_path, "a")
    _stub_compute(monkeypatch, CellShapeQuantifier, {"a": _cell_shape_table([1, 2], [0, 1])})
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


# ----------------------------------------------------- quantities-filtered write


def test_aggregate_quantities_filter_writes_only_selected(tmp_path, monkeypatch):
    """``aggregate(quantities=[...])`` writes only the named tables, even when other
    quantities would pool non-empty rows for the same records."""
    rec = _record(tmp_path, "a", experiment_id="EXP-01")
    _mark_contacts_built(rec)
    _stub_compute(monkeypatch, CellShapeQuantifier, {"a": _cell_shape_table([1, 2], [0])})
    _stub_compute(monkeypatch, NeighborCountQuantifier, {"a": _neighbor_count_table([1, 2], [0])})

    # neighbor_count pools non-empty when selected...
    both = aggregate([rec], tmp_path / "both", quantities=["cell_shape", "neighbor_count"])
    assert set(both) == {"cell_shape", "neighbor_count"}

    # ...but is omitted when the filter names only cell_shape.
    written = aggregate([rec], tmp_path / "sel", quantities=["cell_shape"])
    assert set(written) == {"cell_shape"}
    assert not (tmp_path / "sel" / "neighbor_count.csv").exists()


def test_aggregate_empty_quantities_writes_nothing(tmp_path, monkeypatch):
    """An empty ``quantities`` sequence is an explicit "no tables", distinct from
    ``None`` (= all)."""
    rec = _record(tmp_path, "a", experiment_id="EXP-01")
    _stub_compute(monkeypatch, CellShapeQuantifier, {"a": _cell_shape_table([1, 2], [0])})
    assert aggregate([rec], tmp_path / "none", quantities=[]) == {}
