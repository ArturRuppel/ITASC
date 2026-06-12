"""The plot-consumer registry + product-availability gating (headless)."""
from __future__ import annotations

import pytest

from cellflow.aggregate_quantification.quantifier import OUTPUT_SUBDIR
from cellflow.napari.aggregate_quantification import plots as plots_mod
from cellflow.napari.aggregate_quantification.plots import (
    Plot,
    PlotContext,
    available_plots,
)
from cellflow.napari.studio_plugins import built_quantity_ids


def _unregister(*plot_ids: str) -> None:
    for plot_id in plot_ids:
        plots_mod._REGISTRY.pop(plot_id, None)


def test_subclassing_registers_plot():
    class _FakePlot(Plot):
        plot_id = "fake_plot_for_test"
        display_name = "Fake plot (test)"
        family = "Shape"
        consumes = ("cell_shape",)

    try:
        assert _FakePlot in available_plots()
    finally:
        _unregister("fake_plot_for_test")


def test_intermediate_base_without_id_is_not_registered():
    before = set(available_plots())

    class _Abstract(Plot):  # no plot_id -> not a real plot
        pass

    assert set(available_plots()) == before


def test_available_plots_sorted_by_family_then_display_name():
    class _BPlot(Plot):
        plot_id = "p_b"
        display_name = "Alpha"
        family = "Zeta"

    class _APlot(Plot):
        plot_id = "p_a"
        display_name = "Beta"
        family = "Alpha"

    class _A2Plot(Plot):
        plot_id = "p_a2"
        display_name = "Aardvark"
        family = "Alpha"

    try:
        ordered = [p for p in available_plots() if p.plot_id in {"p_a", "p_a2", "p_b"}]
        assert ordered == [_A2Plot, _APlot, _BPlot]
    finally:
        _unregister("p_a", "p_a2", "p_b")


def test_is_available_and_missing_track_consumed_products():
    class _Plot(Plot):
        plot_id = "consumes_two"
        display_name = "Needs two"
        family = "Combined"
        consumes = ("cell_shape", "contacts")

    try:
        plot = _Plot()
        assert plot.is_available({"cell_shape", "contacts", "extra"}) is True
        assert plot.is_available({"cell_shape"}) is False
        assert plot.missing({"cell_shape"}) == ("contacts",)
        assert plot.missing({"cell_shape", "contacts"}) == ()
        # A plot that consumes nothing is always available.

        class _Free(Plot):
            plot_id = "consumes_none"
            display_name = "Free"
            family = "Misc"

        assert _Free().is_available(set()) is True
    finally:
        _unregister("consumes_two", "consumes_none")


def test_plot_context_load_requires_loader():
    ctx = PlotContext(records=[{"id": "p1"}])
    with pytest.raises(RuntimeError):
        ctx.load("nowhere")

    ctx2 = PlotContext(loader=lambda path: f"loaded:{path}")
    assert ctx2.load("file.h5") == "loaded:file.h5"


def test_built_quantity_ids_reports_built_products(tmp_path):
    # A position whose cell_shape product exists but whose contacts product does not.
    out_dir = tmp_path / OUTPUT_SUBDIR
    out_dir.mkdir()
    (out_dir / "cell_shape.csv").write_text("cell_id,frame\n1,0\n")
    record = {"id": "p1", "position_path": str(tmp_path)}

    built = built_quantity_ids([record])
    assert "cell_shape" in built
    assert "contacts" not in built
    assert "nucleus_shape" not in built


def test_built_quantity_ids_unions_across_records(tmp_path):
    a = tmp_path / "a" / OUTPUT_SUBDIR
    b = tmp_path / "b" / OUTPUT_SUBDIR
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "cell_shape.csv").write_text("cell_id,frame\n1,0\n")
    (b / "nucleus_shape.csv").write_text("cell_id,frame\n1,0\n")
    records = [
        {"id": "a", "position_path": str(tmp_path / "a")},
        {"id": "b", "position_path": str(tmp_path / "b")},
    ]
    built = built_quantity_ids(records)
    assert {"cell_shape", "nucleus_shape"} <= built
