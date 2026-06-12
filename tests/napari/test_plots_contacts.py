"""Contacts-family Plot consumers: registration + pooling pre-built products.

The contacts plots no longer compute — each is a plain ``PoolPlot`` over a
Build-stage product (see ``test_contacts_derived_quantifiers.py``). So every test
here first *builds* the product(s) for the in-scope positions, then asserts the
plot pools the persisted tables. Building uses the same studio helpers
(``position_inputs_from_record`` / ``output_for_record``) the real Build path does,
so the products land where ``pool_quantity`` looks for them.
"""
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
from cellflow.aggregate_quantification.quantifier import available_quantifiers
from cellflow.napari.aggregate_quantification.plots import PlotContext, available_plots
from cellflow.napari.aggregate_quantification.plots.contacts import (
    ContactEnergeticsPlot,
    ContactTypeZScorePlot,
    DensityPlot,
    NeighborCountPlot,
    NeighborEnrichmentPlot,
)
from cellflow.napari.studio_plugins import (
    built_quantity_ids,
    output_for_record,
    position_inputs_from_record,
)

_DERIVED = (
    "neighbor_count",
    "neighbor_enrichment",
    "contact_type_zscore",
    "cell_density",
    "contact_energetics",
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
        e_frame, e_a, e_b = zip(*edges_rows) if edges_rows else ((), (), ())
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


def _build_products(records, *, params=None, only=_DERIVED) -> None:
    """Run the derived quantifiers for *records*, persisting their products."""
    quantifiers = {c.quantity_id: c for c in available_quantifiers()}
    for record in records:
        inputs = position_inputs_from_record(record)
        for quantity_id in only:
            quantifier = quantifiers[quantity_id]()
            quantifier.build(inputs, output_for_record(quantifier, record), params=params)


def test_contacts_plots_register_under_contacts_family():
    plots = {p.plot_id: p for p in available_plots()}
    for plot_id in _DERIVED:
        assert plots[plot_id].family == "Contacts"
        # Each plot now consumes its own Build-stage product, not raw "contacts".
        assert plots[plot_id].consumes == (plot_id,)


def test_built_quantity_ids_sees_built_products(tmp_path):
    records = [_record(tmp_path, "p1", "A")]
    assert not (set(_DERIVED) & built_quantity_ids(records))  # nothing built yet
    _build_products(records, params={"fov_area_mm2": 1.0})
    assert set(_DERIVED) <= built_quantity_ids(records)


def test_neighbor_count_pools_and_joins_class_label(tmp_path):
    records = [_record(tmp_path, "p1", "A"), _record(tmp_path, "p2", "B")]
    _build_products(records)
    df = NeighborCountPlot().prepare(records)
    assert {"n_neighbors", "class_label", "condition", "position_id"} <= set(df.columns)
    assert set(df["class_label"]) == {"positive", "negative"}


def test_enrichment_pool_has_focal_and_neighbor_labels(tmp_path):
    records = [_record(tmp_path, "p1", "A")]
    _build_products(records)
    df = NeighborEnrichmentPlot().prepare(records)
    assert {"enrichment", "focal_label", "neighbor_label"} <= set(df.columns)


def test_zscore_pool_has_contact_type(tmp_path):
    records = [_record(tmp_path, "p1", "A")]
    _build_products(records, params={"shuffles": 25})
    df = ContactTypeZScorePlot().prepare(records)
    assert {"z_score", "contact_type"} <= set(df.columns)


def test_density_pool_honours_built_fov(tmp_path):
    records = [_record(tmp_path, "p1", "A")]
    _build_products(records, params={"fov_area_mm2": 2.0})
    df = DensityPlot().prepare(records)
    all_row = df[df["label"] == "all"]
    # 4 cells over a 2 mm² field of view → 2 cells/mm².
    assert float(all_row["density"].iloc[0]) == 2.0


def test_typed_views_empty_for_unclassified_positions(tmp_path):
    records = [_record(tmp_path, "p1", "A", classify=False)]
    _build_products(records)
    assert NeighborEnrichmentPlot().prepare(records).empty


def test_energetics_pool_handles_no_t1(tmp_path):
    records = [_record(tmp_path, "p1", "A")]
    _build_products(records)
    assert ContactEnergeticsPlot().prepare(records).empty


def test_contacts_plots_build_panels(tmp_path):
    from cellflow.napari.aggregate_quantification.plot_panel import PlotPanel

    app = _app()
    records = [_record(tmp_path, "p1", "A")]
    _build_products(records, params={"fov_area_mm2": 1.0})
    ctx = PlotContext(records=records, viewer=None)
    for cls in (NeighborCountPlot, NeighborEnrichmentPlot, ContactTypeZScorePlot, DensityPlot):
        panel = cls().create_panel(ctx)
        try:
            assert isinstance(panel, PlotPanel)
        finally:
            panel.deleteLater()
            app.processEvents()
