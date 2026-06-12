"""Contacts-family Plot consumers: registration + each view's prepare/panel."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
from qtpy.QtWidgets import QApplication

from cellflow.aggregate_quantification.contacts.nls_classification import (
    nls_classification_csv_path,
    write_nls_classification_csv,
)
from cellflow.napari.aggregate_quantification.plots import PlotContext, available_plots
from cellflow.napari.aggregate_quantification.plots.contacts import (
    ContactEnergeticsPlot,
    ContactTypeZScorePlot,
    DensityPlot,
    NeighborCountPlot,
    NeighborEnrichmentPlot,
)


def _app():
    return QApplication.instance() or QApplication([])


def _write_contacts_h5(path, cells_rows, edges_rows) -> None:
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
        edges.create_dataset("kind", data=np.asarray(["cell_cell"] * len(e_frame), dtype=object))
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
        edges_rows=[(0, 1, 2), (0, 3, 4)],
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


def test_contacts_plots_register_under_contacts_family():
    plots = {p.plot_id: p for p in available_plots()}
    expected = {
        "contact_energetics",
        "neighbor_count",
        "neighbor_enrichment",
        "contact_type_zscore",
        "cell_density",
    }
    assert expected <= set(plots)
    for plot_id in expected:
        assert plots[plot_id].family == "Contacts"
        assert plots[plot_id].consumes == ("contacts",)


def test_neighbor_count_prepare_joins_class_label(tmp_path):
    records = [_record(tmp_path, "p1", "A"), _record(tmp_path, "p2", "B")]
    df = NeighborCountPlot().prepare(records)
    assert {"n_neighbors", "class_label", "condition", "position_id"} <= set(df.columns)
    assert set(df["class_label"]) == {"positive", "negative"}


def test_enrichment_prepare_has_focal_and_neighbor_labels(tmp_path):
    df = NeighborEnrichmentPlot().prepare([_record(tmp_path, "p1", "A")])
    assert {"enrichment", "focal_label", "neighbor_label"} <= set(df.columns)


def test_zscore_prepare_has_contact_type(tmp_path):
    df = ContactTypeZScorePlot().prepare([_record(tmp_path, "p1", "A")])
    assert {"z_score", "contact_type"} <= set(df.columns)


def test_density_prepare_auto_resolves(tmp_path):
    df = DensityPlot().prepare([_record(tmp_path, "p1", "A")])
    assert {"density", "label"} <= set(df.columns)


def test_density_honours_shared_fov_param(tmp_path):
    from cellflow.napari.aggregate_quantification.plots import PlotParams

    df = DensityPlot().prepare([_record(tmp_path, "p1", "A")], PlotParams(fov_area_mm2=2.0))
    all_row = df[df["label"] == "all"]
    # 4 cells over a 2 mm² field of view → 2 cells/mm².
    assert float(all_row["density"].iloc[0]) == 2.0


def test_typed_views_skip_unclassified_positions(tmp_path):
    unclassified = _record(tmp_path, "p1", "A", classify=False)
    assert NeighborEnrichmentPlot().prepare([unclassified]).empty


def test_energetics_prepare_handles_no_t1(tmp_path):
    # No T1 events in the fixture → empty landscape (no crash).
    df = ContactEnergeticsPlot().prepare([_record(tmp_path, "p1", "A")])
    assert df.empty


def test_contacts_plots_build_panels(tmp_path):
    from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel

    app = _app()
    records = [_record(tmp_path, "p1", "A")]
    ctx = PlotContext(records=records, viewer=None)
    for cls in (NeighborCountPlot, NeighborEnrichmentPlot, ContactTypeZScorePlot, DensityPlot):
        panel = cls().create_panel(ctx)
        try:
            assert isinstance(panel, PlotPanel)
        finally:
            panel.deleteLater()
            app.processEvents()
