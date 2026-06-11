from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import tifffile
from qtpy.QtWidgets import QApplication
from skimage.draw import disk

from cellflow.aggregate_quantification.quantifier import PositionInputs
from cellflow.aggregate_quantification.quantifiers.cell_dynamics import (
    CellDynamicsQuantifier,
)
from cellflow.napari.aggregate_quantification.plugins import AnalysisContext
from cellflow.napari.aggregate_quantification.plugins.track_dynamics import (
    TrackDynamicsPlugin,
    _curve_records,
    _pool_records,
    _tissue_records,
)


def _app():
    return QApplication.instance() or QApplication([])


class _FakeWindow:
    def __init__(self) -> None:
        self.docks: list[tuple] = []

    def add_dock_widget(self, widget, *, area=None, name=None):
        self.docks.append((widget, area, name))
        return object()


class _FakeLabelsLayer:
    def __init__(self, data, name) -> None:
        self.data, self.name = data, name
        self.selected_label = None
        self.show_selected_label = False


class _FakeDims:
    def set_current_step(self, axis, value) -> None:
        pass


class _FakeCamera:
    center = None


class _FakeViewer:
    def __init__(self) -> None:
        self.window = _FakeWindow()
        self.layers: list = []
        self.dims = _FakeDims()
        self.camera = _FakeCamera()

    def add_labels(self, data, name=None):
        layer = _FakeLabelsLayer(data, name)
        self.layers.append(layer)
        return layer


def _moving_disk_stack(centers, shape=(80, 80), radius=6, label=1):
    frames = []
    for row, col in centers:
        frame = np.zeros(shape, dtype=np.uint16)
        rr, cc = disk((row, col), radius, shape=shape)
        frame[rr, cc] = label
        frames.append(frame)
    return np.stack(frames)


def _built_cell_position(tmp_path, name, condition):
    pos = tmp_path / name
    pos.mkdir()
    cell_path = pos / "cells.tif"
    centers = [(40, 10 + 2 * i) for i in range(16)]
    tifffile.imwrite(cell_path, _moving_disk_stack(centers))
    q = CellDynamicsQuantifier()
    inputs = PositionInputs(
        position_dir=pos, cell_labels_path=cell_path, pixel_size_um=0.5, time_interval_s=2.0
    )
    q.build(inputs, q.default_output(inputs))
    return {
        "position_path": pos,
        "cell_tracked_labels_path": cell_path,
        "condition": condition,
        "date": "d1",
        "id": name,
    }


# ----------------------------------------------------------------------- scope
def test_default_scope_prefers_cell_then_nucleus(tmp_path):
    app = _app()
    plugin = TrackDynamicsPlugin()
    plugin.set_context(
        AnalysisContext(records=[{"id": "p1", "nucleus_tracked_labels_path": tmp_path / "n.tif"}])
    )
    assert plugin._scope == "nucleus"
    plugin.set_context(
        AnalysisContext(records=[{"id": "p2", "cell_tracked_labels_path": tmp_path / "c.tif"}])
    )
    assert plugin._scope == "cell"
    plugin.deleteLater()
    app.processEvents()


# --------------------------------------------------------------------- compute
def test_build_blocked_until_pixel_size_and_interval(tmp_path):
    app = _app()
    captured: list = []
    plugin = TrackDynamicsPlugin()
    plugin.set_build_callback(lambda q, recs, ow: captured.append((q, recs, ow)))
    record = {"id": "p1", "cell_tracked_labels_path": tmp_path / "c.tif", "position_path": tmp_path}
    plugin.set_context(AnalysisContext(records=[record]))
    # Neither pixel size nor interval resolvable -> blocked.
    assert plugin._build_btn.isEnabled() is False
    plugin._pixel_size_edit.setText("0.5")
    assert plugin._build_btn.isEnabled() is False  # still missing interval
    plugin._interval_edit.setText("2.0")
    assert plugin._build_btn.isEnabled() is True

    plugin._on_build()
    q, recs, _ = captured[-1]
    assert q.quantity_id == "cell_dynamics"
    assert recs[0]["pixel_size_um"] == 0.5 and recs[0]["time_interval_s"] == 2.0
    plugin.deleteLater()
    app.processEvents()


def test_build_uses_nucleus_quantifier_when_scope_is_nucleus(tmp_path):
    app = _app()
    captured: list = []
    plugin = TrackDynamicsPlugin()
    plugin.set_build_callback(lambda q, recs, ow: captured.append((q, recs, ow)))
    plugin.set_context(
        AnalysisContext(
            records=[{"id": "p1", "nucleus_tracked_labels_path": tmp_path / "n.tif",
                      "position_path": tmp_path}]
        )
    )
    plugin._pixel_size_edit.setText("0.5")
    plugin._interval_edit.setText("1.0")
    assert plugin._scope == "nucleus"
    plugin._on_build()
    assert captured[-1][0].quantity_id == "nucleus_dynamics"
    plugin.deleteLater()
    app.processEvents()


# --------------------------------------------------------------------- pooling
def test_pool_per_frame_and_per_track(tmp_path):
    records = [
        _built_cell_position(tmp_path, "p1", "A"),
        _built_cell_position(tmp_path, "p2", "B"),
    ]
    frame_df = _pool_records(CellDynamicsQuantifier(), records, "frame")
    assert {"speed_um_per_s", "net_disp_um", "frame", "cell_id", "position_id", "class_label"} <= set(
        frame_df.columns
    )
    assert set(frame_df["condition"]) == {"A", "B"}
    assert set(frame_df["class_label"]) == {"unclassified"}

    track_df = _pool_records(CellDynamicsQuantifier(), records, "track")
    assert {
        "directionality_ratio", "persistence_time_s", "cell_id", "position_id",
        "msd_D_um2_per_s", "msd_alpha",
    } <= set(track_df.columns)
    # One track per position (a single moving cell), straight -> ratio ≈ 1.
    np.testing.assert_allclose(track_df["directionality_ratio"], 1.0, atol=1e-6)
    # Straight ballistic single track -> per-track MSD exponent ≈ 2.
    np.testing.assert_allclose(track_df["msd_alpha"], 2.0, atol=0.05)


def test_tissue_records_one_row_per_built_position(tmp_path):
    records = [
        _built_cell_position(tmp_path, "p1", "A"),
        _built_cell_position(tmp_path, "p2", "B"),
    ]
    tissue = _tissue_records(CellDynamicsQuantifier(), records)
    assert len(tissue) == 2
    assert {
        "msd_D_um2_per_s", "msd_alpha", "persistence_time_s", "corr_length_um", "order_param",
        "condition", "date", "position_id",
    } <= set(tissue.columns)
    assert set(tissue["condition"]) == {"A", "B"}


def test_tissue_view_opens_a_dock(tmp_path):
    app = _app()
    records = [_built_cell_position(tmp_path, "p1", "A")]
    viewer = _FakeViewer()
    plugin = TrackDynamicsPlugin()
    plugin.set_context(AnalysisContext(records=records, viewer=viewer))
    plugin._on_pool_done(("tissue", _tissue_records(CellDynamicsQuantifier(), records)))
    assert len(viewer.window.docks) == 1
    plugin.deleteLater()
    app.processEvents()


def test_curve_records_carries_fits(tmp_path):
    records = [_built_cell_position(tmp_path, "p1", "A")]
    curves = _curve_records(CellDynamicsQuantifier(), records)
    assert len(curves) == 1
    c = curves[0]
    assert c.group == "A"
    assert c.msd_lag_s.size > 0
    assert abs(c.msd_alpha - 2.0) < 0.1  # straight track is ballistic


# --------------------------------------------------------------------- plotting
def test_plot_enabled_only_when_built_and_viewer(tmp_path):
    app = _app()
    records = [_built_cell_position(tmp_path, "p1", "A")]
    plugin = TrackDynamicsPlugin()
    plugin.set_context(AnalysisContext(records=records))
    assert plugin._plot_btn.isEnabled() is False
    plugin.set_context(AnalysisContext(records=records, viewer=_FakeViewer()))
    assert plugin._plot_btn.isEnabled() is True
    plugin.deleteLater()
    app.processEvents()


def test_plots_share_one_dock_as_tabs(tmp_path):
    from qtpy.QtWidgets import QTabWidget

    app = _app()
    records = [_built_cell_position(tmp_path, "p1", "A")]
    viewer = _FakeViewer()
    plugin = TrackDynamicsPlugin()
    plugin.set_context(AnalysisContext(records=records, viewer=viewer))

    plugin._on_pool_done(("frame", _pool_records(CellDynamicsQuantifier(), records, "frame")))
    plugin._on_pool_done(("track", _pool_records(CellDynamicsQuantifier(), records, "track")))
    plugin._on_pool_done(("curves", _curve_records(CellDynamicsQuantifier(), records)))

    # One dock holding a tab widget — every view is a tab, not its own dock
    # (which made napari split the area and shrink every plot).
    assert len(viewer.window.docks) == 1
    widget, area, name = viewer.window.docks[0]
    assert area == "right"
    assert name == "Dynamics plots"
    assert isinstance(widget, QTabWidget)
    assert [widget.tabText(i) for i in range(widget.count())] == ["Plot 1", "Plot 2", "Plot 3"]
    plugin.deleteLater()
    app.processEvents()


def test_dynamics_distribution_panel_gets_resolver(tmp_path):
    app = _app()
    records = [_built_cell_position(tmp_path, "p1", "A")]
    viewer = _FakeViewer()
    plugin = TrackDynamicsPlugin()
    plugin.set_context(AnalysisContext(records=records, viewer=viewer))

    # Per-track view: identity has frame_start (no frame) -> target.frame == frame_start.
    plugin._on_pool_done(("track", _pool_records(CellDynamicsQuantifier(), records, "track")))
    panel = plugin._panel
    assert panel._target_resolver is not None
    target = panel._target_resolver({"position_id": "p1", "frame_start": 5, "cell_id": 1})
    assert target is not None
    assert target.frame == 5

    # Load must work even after GC: the loader's lifetime is tied to the panel
    # (a bare bound-method signal connection would have been collected).
    import gc

    gc.collect()
    panel._select_row(0)
    assert panel._load_btn.isEnabled()
    panel._load_btn.click()
    assert len(viewer.layers) == 1
    plugin.deleteLater()
    app.processEvents()
