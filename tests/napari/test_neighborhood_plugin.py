"""Pool + headless-launch tests for the Neighborhood & Density plugin."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pandas as pd
from qtpy.QtWidgets import QApplication

from cellflow.aggregate_quantification.contacts.nls_classification import (
    nls_classification_csv_path,
    write_nls_classification_csv,
)
from cellflow.napari.aggregate_quantification.plugins import AnalysisContext
from cellflow.napari.aggregate_quantification.plugins.neighborhood import (
    NeighborhoodPlugin,
    _is_built,
    _pool_neighborhood,
)


def _app():
    return QApplication.instance() or QApplication([])


class _FakeWindow:
    def __init__(self) -> None:
        self.docks: list[tuple] = []

    def add_dock_widget(self, widget, *, area=None, name=None):
        self.docks.append((widget, area, name))
        return object()


class _FakeViewer:
    def __init__(self) -> None:
        self.window = _FakeWindow()
        self.layers: list = []


def _write_contacts_h5(path, cells_rows, edges_rows) -> None:
    """A minimal contact_analysis.h5 with cells + edges tables (no T1 events)."""
    with h5py.File(path, "w") as h5:
        prov = h5.create_group("provenance")
        prov.attrs["cell_tracked_labels_path"] = "cells.tif"
        prov.attrs["nucleus_tracked_labels_path"] = "nuclei.tif"

        cells = h5.create_group("cells/table")
        c_frame, c_id = zip(*cells_rows) if cells_rows else ((), ())
        cells.create_dataset("frame", data=np.asarray(c_frame, dtype=np.int64))
        cells.create_dataset("cell_id", data=np.asarray(c_id, dtype=np.int64))

        edges = h5.create_group("edges/table")
        if edges_rows:
            e_frame, e_a, e_b = zip(*edges_rows)
        else:
            e_frame, e_a, e_b = (), (), ()
        edges.create_dataset("frame", data=np.asarray(e_frame, dtype=np.int64))
        edges.create_dataset("cell_a", data=np.asarray(e_a, dtype=np.int64))
        edges.create_dataset("cell_b", data=np.asarray(e_b, dtype=np.int64))
        edges.create_dataset(
            "kind", data=np.asarray(["cell_cell"] * len(e_frame), dtype=object)
        )
        edges.create_dataset("length", data=np.ones(len(e_frame), dtype=float))

        h5.create_group("t1_events/table")
        h5.create_dataset("edges/coordinates/y", data=np.empty(0))
        h5.create_dataset("edges/coordinates/x", data=np.empty(0))


def _record(tmp_path, name, condition, *, classify=True) -> dict:
    pos = tmp_path / name
    pos.mkdir()
    h5_path = pos / "contact_analysis.h5"
    _write_contacts_h5(
        h5_path,
        cells_rows=[(0, 1), (0, 2), (0, 3), (0, 4)],
        edges_rows=[(0, 1, 2), (0, 3, 4)],  # AA + BB (fully sorted)
    )
    if classify:
        write_nls_classification_csv(
            nls_classification_csv_path(pos),
            {1: "positive", 2: "positive", 3: "negative", 4: "negative"},
            positive_label="positive",
            negative_label="negative",
        )
    return {
        "position_path": pos,
        "contact_analysis_path": h5_path,
        "condition": condition,
        "date": "d1",
        "id": name,
    }


# ----------------------------------------------------------------- pooling
def test_neighbor_count_pools_with_class_label(tmp_path):
    records = [_record(tmp_path, "p1", "A"), _record(tmp_path, "p2", "B")]
    pooled, notes = _pool_neighborhood(records, "neighbor_count", None, 100)
    assert {"n_neighbors", "class_label", "condition", "position_id"} <= set(pooled.columns)
    assert set(pooled["condition"]) == {"A", "B"}
    assert set(pooled["class_label"]) == {"positive", "negative"}
    assert notes == []


def test_enrichment_pools_with_focal_and_neighbor_labels(tmp_path):
    records = [_record(tmp_path, "p1", "A")]
    pooled, notes = _pool_neighborhood(records, "enrichment", None, 100)
    assert {"enrichment", "focal_label", "neighbor_label"} <= set(pooled.columns)
    # Fully-sorted layout → homotypic enrichment above 1.
    homo = pooled[pooled["focal_label"] == pooled["neighbor_label"]]["enrichment"]
    assert (homo.dropna() > 1).all()


def test_zscore_pools_with_contact_type(tmp_path):
    records = [_record(tmp_path, "p1", "A")]
    pooled, notes = _pool_neighborhood(records, "zscore", None, 500)
    assert {"z_score", "contact_type"} <= set(pooled.columns)
    assert set(pooled["contact_type"]) <= {"positive·positive", "negative·positive", "negative·negative"}


def test_density_pools_with_manual_fov(tmp_path):
    records = [_record(tmp_path, "p1", "A")]
    pooled, notes = _pool_neighborhood(records, "density", 2.0, 100)
    assert {"density", "label", "n_cells"} <= set(pooled.columns)
    all_row = pooled[pooled["label"] == "all"]
    assert (all_row["n_cells"] == 4).all()
    assert (all_row["density"] == 2.0).all()


def test_unclassified_position_contributes_untyped_skips_typed(tmp_path):
    records = [
        _record(tmp_path, "p1", "A", classify=True),
        _record(tmp_path, "p2", "B", classify=False),  # no NLS CSV
    ]
    # Neighbor count: both positions contribute; the unclassified one fills
    # class_label = "unclassified".
    counts, notes = _pool_neighborhood(records, "neighbor_count", None, 100)
    assert set(counts["position_id"]) == {"p1", "p2"}
    assert "unclassified" in set(counts["class_label"])

    # Enrichment: the unclassified position is skipped and noted.
    enrich, notes = _pool_neighborhood(records, "enrichment", None, 100)
    assert set(enrich["position_id"]) == {"p1"}
    assert any("p2" in n for n in notes)


def test_density_without_pixel_size_notes_unavailable(tmp_path):
    records = [_record(tmp_path, "p1", "A")]
    # No manual FOV and the provenance TIFF path doesn't exist → density NaN.
    pooled, notes = _pool_neighborhood(records, "density", None, 100)
    assert pooled["density"].isna().all()
    assert any("unavailable" in n for n in notes)


# ----------------------------------------------------------------- plugin shell
def test_is_built_false_for_missing_artifact(tmp_path):
    assert _is_built({"contact_analysis_path": tmp_path / "nope.h5"}) is False
    assert _is_built({}) is False


def test_plot_enabled_only_when_built_and_viewer(tmp_path):
    app = _app()
    records = [_record(tmp_path, "p1", "A")]
    plugin = NeighborhoodPlugin()
    plugin.set_context(AnalysisContext(records=records))
    assert plugin._plot_btn.isEnabled() is False  # no viewer
    plugin.set_context(AnalysisContext(records=records, viewer=_FakeViewer()))
    assert plugin._plot_btn.isEnabled() is True
    plugin.deleteLater()
    app.processEvents()


def test_view_toggles_relevant_inputs(tmp_path):
    app = _app()
    plugin = NeighborhoodPlugin()
    plugin._view_combo.setCurrentIndex(plugin._view_combo.findData("density"))
    assert plugin._fov_row.isVisibleTo(plugin) is True
    assert plugin._shuffles_row.isVisibleTo(plugin) is False
    plugin._view_combo.setCurrentIndex(plugin._view_combo.findData("zscore"))
    assert plugin._shuffles_row.isVisibleTo(plugin) is True
    assert plugin._fov_row.isVisibleTo(plugin) is False
    plugin.deleteLater()
    app.processEvents()


def test_on_pool_done_opens_panel_with_view_defaults(tmp_path):
    app = _app()
    records = [_record(tmp_path, "p1", "A")]
    viewer = _FakeViewer()
    plugin = NeighborhoodPlugin()
    plugin.set_context(AnalysisContext(records=records, viewer=viewer))

    pooled, notes = _pool_neighborhood(records, "zscore", None, 200)
    plugin._on_pool_done(("zscore", pooled, notes))

    assert len(viewer.window.docks) == 1
    panel = plugin._panel
    assert panel._plot_combo.currentData() == "bar"
    assert "z_score" in panel._value_columns
    plugin.deleteLater()
    app.processEvents()


def test_empty_pool_opens_no_dock(tmp_path):
    app = _app()
    viewer = _FakeViewer()
    plugin = NeighborhoodPlugin()
    plugin.set_context(AnalysisContext(records=[], viewer=viewer))
    plugin._on_pool_done(("neighbor_count", pd.DataFrame(), []))
    assert viewer.window.docks == []
    assert "No data in scope" in plugin._plot_status.text()
    plugin.deleteLater()
    app.processEvents()
