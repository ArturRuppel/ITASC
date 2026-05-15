from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import napari
from qtpy.QtWidgets import QApplication, QSizePolicy


def _make_viewer():
    app = QApplication.instance() or QApplication([])
    viewer = napari.Viewer(show=False)
    return app, viewer


def _install_import_stubs() -> None:
    src_root = Path(__file__).resolve().parents[2] / "src" / "cellflow"
    package_root = src_root / "napari"

    napari_pkg = types.ModuleType("cellflow.napari")
    napari_pkg.__path__ = [str(package_root)]
    sys.modules["cellflow.napari"] = napari_pkg

    tracking_pkg = types.ModuleType("cellflow.tracking_ultrack")
    tracking_pkg.__path__ = [str(src_root / "tracking_ultrack")]
    sys.modules["cellflow.tracking_ultrack"] = tracking_pkg

    class _StubTrackingConfig:
        def __init__(self, **kwargs):
            self.min_area = 100
            self.max_distance = 15.0
            self.max_neighbors = 5
            self.linking_mode = "default"
            self.area_weight = 1.0
            self.iou_weight = 1.0
            self.distance_weight = 0.25
            self.min_area_ratio = 0.3
            self.power = 4.0
            self.quality_weight = 1.0
            self.quality_exponent = 8.0
            self.circularity_weight = 0.25
            self.appear_weight = -0.001
            self.disappear_weight = -0.001
            self.division_weight = -0.001
            self.bias = 0.0
            self.solution_gap = 0.001
            self.time_limit = 36000
            self.window_size = 0
            self.max_segments_per_time = 1_000_000
            self.__dict__.update(kwargs)

    stub_exports = {
        "cellflow.tracking_ultrack.config": {"TrackingConfig": _StubTrackingConfig},
        "cellflow.tracking_ultrack.corrections": {
            "Correction": __import__(
                "cellflow.tracking_ultrack.corrections",
                fromlist=["Correction"],
            ).Correction,
        },
        "cellflow.tracking_ultrack.db_build": {
            "apply_annotations_and_score": lambda *args, **kwargs: None,
        },
        "cellflow.tracking_ultrack.export": {"export_tracked_labels": lambda *args, **kwargs: None},
        "cellflow.tracking_ultrack.ingest": {
            "ingest_hypotheses_to_db": lambda *args, **kwargs: None,
            "_select_solver": lambda: "CBC",
            "_build_ultrack_config": lambda *args, **kwargs: None,
        },
        "cellflow.tracking_ultrack.linking": {"run_linking": lambda *args, **kwargs: iter(())},
        "cellflow.tracking_ultrack.multi_threshold": {
            "build_ultrack_database_from_sources": lambda *args, **kwargs: None,
            "preview_ultrack_source_stack_frame": lambda *args, **kwargs: (None, None, 0, []),
            "write_ultrack_source_stacks": lambda *args, **kwargs: [],
        },
        "cellflow.tracking_ultrack.extend": {
            "extend_track": lambda *args, **kwargs: None,
            "extend_track_from_db": lambda *args, **kwargs: None,
        },
        "cellflow.tracking_ultrack.reseed": {
            "resolve_with_validation": lambda *args, **kwargs: None,
            "resolve_with_canonical_segment": lambda *args, **kwargs: (None, {}),
        },
        "cellflow.tracking_ultrack.seed_prior": {
            "write_seed_prior_node_probs": lambda *args, **kwargs: None,
        },
        "cellflow.tracking_ultrack.solve": {
            "database_has_annotations": lambda *args, **kwargs: False,
            "run_solve": lambda *args, **kwargs: iter(()),
        },
        "cellflow.segmentation": {
            "apply_gamma": lambda logits, gamma: logits,
            "build_nucleus_averaged_maps": lambda *args, **kwargs: None,
            "build_consensus_boundary": lambda *args, **kwargs: (None, None),
        },
    }

    for module_name, attrs in stub_exports.items():
        module = types.ModuleType(module_name)
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
        sys.modules[module_name] = module


def _load_widget_class():
    _install_import_stubs()
    module = importlib.import_module("cellflow.napari.nucleus_workflow_widget")
    return module.NucleusWorkflowWidget


def test_database_browser_activate_button_expands_and_deactivation_removes_layers():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    viewer.add_labels(np.ones((1, 4, 4), dtype=np.uint8), name="Ultrack DB Preview")
    viewer.add_labels(np.ones((1, 4, 4), dtype=np.uint8), name="Ultrack DB Selection")
    viewer.add_labels(np.ones((1, 4, 4), dtype=np.uint8), name="Ultrack DB Annotations")

    assert widget.ultrack_db_browser_section.is_expanded is False

    widget.ultrack_db_active_btn.setChecked(True)

    assert widget.ultrack_db_browser_section.is_expanded is True
    assert widget._ultrack_db_browser_active is True

    widget.ultrack_db_active_btn.setChecked(False)

    assert widget.ultrack_db_browser_section.is_expanded is False
    assert widget._ultrack_db_browser_active is False
    assert "Ultrack DB Preview" not in viewer.layers
    assert "Ultrack DB Selection" not in viewer.layers
    assert "Ultrack DB Annotations" not in viewer.layers

    widget.deleteLater()
    viewer.close()

# ── Task 7: Ultrack DB browser ────────────────────────────────────────────────

def test_ultrack_db_browser_shows_missing_db_status(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget._pos_dir = pos_dir
    widget._ultrack_db_browser_active = True

    widget._refresh_ultrack_db_browser()

    text = widget.ultrack_db_section_status_lbl.text().lower()
    assert "data.db" in text or "missing" in text or "not found" in text

    widget.deleteLater()
    viewer.close()


def _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, heights):
    module = sys.modules[widget_class.__module__]
    states = tuple(module._HierarchyCutState((), height) for height in heights)
    monkeypatch.setattr(widget, "_query_hierarchy_cut_states", lambda *a: states)
    monkeypatch.setattr(
        widget,
        "_render_hierarchy_cut_state",
        lambda path, frame, state: widget._render_hierarchy_cut(path, frame, state.height),
    )
    return states


def test_ultrack_db_browser_summary_label_wraps_instead_of_widening():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.ultrack_db_info_lbl.wordWrap() is True
    assert (
        widget.ultrack_db_info_lbl.sizePolicy().horizontalPolicy()
        != QSizePolicy.Policy.Expanding
    )

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_exposes_hierarchy_only_controls():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert not hasattr(widget, "ultrack_db_mode_combo")
    assert widget.ultrack_db_hierarchy_slider.minimum() == 0
    assert widget.ultrack_db_hierarchy_slider.maximum() == 100
    assert widget.ultrack_db_hierarchy_slider.value() == 50
    assert widget._ultrack_db_slider_row.isHidden() is False

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_exposes_connected_focus_controls():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.ultrack_db_connected_focus_check.text() == "Connected focus"
    assert widget.ultrack_db_edge_alpha_check.text() == "Edge weight transparency"
    assert widget.ultrack_db_prob_alpha_check.text() == "Node prob transparency"
    assert not widget.ultrack_db_connected_focus_check.isEnabled()
    assert not widget.ultrack_db_edge_alpha_check.isEnabled()

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_normalizes_preview_metadata():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    labels = np.array([[1, 0]], dtype=np.uint32)
    normalized = widget._normalize_ultrack_db_preview(
        (labels, "status", {1: 0.5}, {1: 101}, {101: 1})
    )

    assert normalized[0] is labels
    assert normalized[1] == "status"
    assert normalized[2] == {1: 0.5}
    assert normalized[3] == {1: 101}
    assert normalized[4] == {101: 1}

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_builds_display_label_node_id_metadata():
    widget_class = _load_widget_class()

    prob_dict, label_to_node_id, node_id_to_label = (
        widget_class._ultrack_db_node_preview_metadata(
            [
                types.SimpleNamespace(id=101, node_prob=0.25),
                types.SimpleNamespace(id=202, node_prob=0.75),
            ]
        )
    )

    assert prob_dict == {1: 0.25, 2: 0.75}
    assert label_to_node_id == {1: 101, 2: 202}
    assert node_id_to_label == {101: 1, 202: 2}


def test_ultrack_db_browser_click_selects_node_id_from_display_label(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    monkeypatch.setattr(widget, "_current_t", lambda: 4)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda *a: "summary")
    _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, (0.5,))
    monkeypatch.setattr(
        widget,
        "_render_hierarchy_cut",
        lambda *a: (
            np.array([[0, 2], [0, 0]], dtype=np.uint32),
            "rendered",
            {2: 0.8},
            {2: 222},
            {222: 2},
        ),
    )

    widget._refresh_ultrack_db_browser()
    widget._select_ultrack_db_preview_label(2, frame=4)

    assert widget._ultrack_db_selected_node_id == 222
    assert widget._ultrack_db_selected_frame == 4
    assert "Selected node 222" in widget.ultrack_db_section_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_selection_highlight_uses_cyan_contour():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    labels = np.zeros((6, 6), dtype=np.uint32)
    labels[2:5, 1:4] = 7

    widget._update_ultrack_db_highlight(labels, 7)

    layer = viewer.layers["Ultrack DB Selection"]
    assert layer.visible
    assert len(layer.data) == 1
    assert layer.name == "Ultrack DB Selection"

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_connected_focus_filters_by_viewer_frame(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    widget.ultrack_db_connected_focus_check.setChecked(True)
    widget._ultrack_db_selected_node_id = 222
    widget._ultrack_db_selected_frame = 4
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda *a: "summary")
    _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, (0.5,))
    monkeypatch.setattr(
        widget,
        "_query_ultrack_db_connected_nodes",
        lambda *a: ({111: 0.25}, {333: 0.9}),
    )

    labels_by_frame = {
        3: (
            np.array([[1, 0], [2, 0]], dtype=np.uint32),
            {1: 111, 2: 999},
            {111: 1, 999: 2},
        ),
        4: (
            np.array([[1, 0], [2, 0]], dtype=np.uint32),
            {1: 222, 2: 999},
            {222: 1, 999: 2},
        ),
        5: (
            np.array([[1, 0], [2, 0]], dtype=np.uint32),
            {1: 333, 2: 999},
            {333: 1, 999: 2},
        ),
    }
    current = {"t": 4}
    monkeypatch.setattr(widget, "_current_t", lambda: current["t"])

    def _render(*args):
        labels, label_to_node_id, node_id_to_label = labels_by_frame[current["t"]]
        return labels, "rendered", {1: 1.0, 2: 1.0}, label_to_node_id, node_id_to_label

    monkeypatch.setattr(widget, "_render_hierarchy_cut", _render)

    widget._refresh_ultrack_db_browser()
    assert set(np.unique(viewer.layers["Ultrack DB Preview"].data)) == {0, 1}

    current["t"] = 3
    widget._refresh_ultrack_db_browser()
    assert set(np.unique(viewer.layers["Ultrack DB Preview"].data)) == {0, 1}

    current["t"] = 5
    widget._refresh_ultrack_db_browser()
    assert set(np.unique(viewer.layers["Ultrack DB Preview"].data)) == {0, 1}

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_edge_and_node_prob_transparency_multiply(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    widget.ultrack_db_connected_focus_check.setChecked(True)
    widget.ultrack_db_edge_alpha_check.setChecked(True)
    widget.ultrack_db_prob_alpha_check.setChecked(True)
    widget._ultrack_db_selected_node_id = 222
    widget._ultrack_db_selected_frame = 4
    monkeypatch.setattr(widget, "_current_t", lambda: 5)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda *a: "summary")
    _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, (0.5,))
    monkeypatch.setattr(
        widget,
        "_query_ultrack_db_connected_nodes",
        lambda *a: ({}, {333: 0.5, 444: 1.0}),
    )
    monkeypatch.setattr(
        widget,
        "_render_hierarchy_cut",
        lambda *a: (
            np.array([[1, 0], [2, 0]], dtype=np.uint32),
            "rendered",
            {1: 0.2, 2: 0.8},
            {1: 333, 2: 444},
            {333: 1, 444: 2},
        ),
    )

    widget._refresh_ultrack_db_browser()

    layer = viewer.layers["Ultrack DB Preview"]
    assert layer.data.shape == (2, 2, 4)
    assert layer.data[0, 0, 3] == pytest.approx(0.075)
    assert layer.data[1, 0, 3] == pytest.approx(1.0)
    assert layer.data[0, 1, 3] == 0

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_refresh_reanchors_selection_contour(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    monkeypatch.setattr(widget, "_current_t", lambda: 4)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda *a: "summary")
    _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, (0.5,))
    renders = [
        (
            np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint32),
            "rendered",
            {1: 0.8},
            {1: 222},
            {222: 1},
        ),
        (
            np.array([[0, 0, 0], [0, 2, 2], [0, 2, 2]], dtype=np.uint32),
            "rendered",
            {2: 0.8},
            {2: 222},
            {222: 2},
        ),
    ]
    monkeypatch.setattr(widget, "_render_hierarchy_cut", lambda *a: renders.pop(0))

    widget._refresh_ultrack_db_browser()
    widget._select_ultrack_db_preview_label(1, frame=4)
    first_contour = np.asarray(viewer.layers["Ultrack DB Selection"].data[0]).copy()
    widget._ultrack_db_preview_cache.clear()
    widget._refresh_ultrack_db_browser()
    second_contour = np.asarray(viewer.layers["Ultrack DB Selection"].data[0])

    assert widget._ultrack_db_node_id_to_label == {222: 2}
    assert second_contour[:, 0].max() > first_contour[:, 0].max()
    assert second_contour[:, 1].max() > first_contour[:, 1].max()

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_connected_focus_reports_hidden_selected_node(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    widget.ultrack_db_connected_focus_check.setChecked(True)
    widget._ultrack_db_selected_node_id = 222
    widget._ultrack_db_selected_frame = 4
    monkeypatch.setattr(widget, "_current_t", lambda: 4)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda *a: "summary")
    _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, (0.5,))
    monkeypatch.setattr(widget, "_query_ultrack_db_connected_nodes", lambda *a: ({}, {}))
    monkeypatch.setattr(
        widget,
        "_render_hierarchy_cut",
        lambda *a: (
            np.array([[1, 0]], dtype=np.uint32),
            "rendered",
            {1: 0.8},
            {1: 999},
            {999: 1},
        ),
    )

    widget._refresh_ultrack_db_browser()

    assert "hidden" in widget.ultrack_db_section_status_lbl.text().lower()
    assert "222" in widget.ultrack_db_section_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_hierarchy_cut_caches_by_frame_and_slider(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget.ultrack_db_hierarchy_slider.setValue(1)
    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True  # skip middle-frame jump; no real DB to query

    calls = []

    def _fake_summary(path, frame):
        return "3 nodes | 2 links | frame 0: 1 nodes"

    def _fake_render(path, frame, slider_int):
        calls.append((path, frame, slider_int))
        return np.zeros((5, 5), dtype=np.uint32), "rendered hierarchy cut"

    monkeypatch.setattr(widget, "_ultrack_db_summary_text", _fake_summary)
    _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, (0.25, 0.75))
    monkeypatch.setattr(widget, "_render_hierarchy_cut", _fake_render)

    widget._refresh_ultrack_db_browser()
    widget._refresh_ultrack_db_browser()

    assert len(calls) == 1
    assert calls[0] == (db_path, 0, 0.75)
    assert widget.ultrack_db_info_lbl.text() == "3 nodes | 2 links | frame 0: 1 nodes"
    assert widget.ultrack_db_section_status_lbl.text() == "rendered hierarchy cut"
    assert "Ultrack DB Preview" in viewer.layers

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_probability_transparency_renders_rgba_preview(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget.ultrack_db_hierarchy_slider.setValue(0)
    widget.ultrack_db_prob_alpha_check.setChecked(True)
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda path, frame: "summary")
    _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, (0.5,))

    labels = np.array(
        [
            [1, 0],
            [0, 2],
        ],
        dtype=np.uint32,
    )
    monkeypatch.setattr(
        widget,
        "_render_hierarchy_cut",
        lambda *args: (labels, "rendered hierarchy cut", {1: 0.2, 2: 0.8}),
    )

    widget._refresh_ultrack_db_browser()

    layer = viewer.layers["Ultrack DB Preview"]
    assert layer.data.shape == (2, 2, 4)
    assert layer.data[0, 0, 3] < layer.data[1, 1, 3]
    assert layer.data[0, 1, 3] == 0

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_shows_summary_while_rendering_hierarchy(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True

    calls = []
    labels = np.zeros((5, 5), dtype=np.uint32)
    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda path, frame: "summary stats")
    _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, (0.5,))
    monkeypatch.setattr(
        widget,
        "_render_hierarchy_cut",
        lambda path, frame, height: calls.append((path, frame, height))
        or (labels, "rendered hierarchy cut"),
    )

    widget._refresh_ultrack_db_browser()

    assert calls == [(db_path, 0, 0.5)]
    assert widget.ultrack_db_info_lbl.text() == "summary stats"
    assert widget.ultrack_db_section_status_lbl.text() == "rendered hierarchy cut"
    assert "Ultrack DB Preview" in viewer.layers

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_does_not_add_contour_or_foreground_layers(tmp_path, monkeypatch):
    import tifffile

    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    pos_dir = tmp_path / "pos00"
    db_path = pos_dir / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    tifffile.imwrite(pos_dir / "2_nucleus" / "contour_maps.tif", np.zeros((1, 4, 4), dtype=np.float32))
    tifffile.imwrite(pos_dir / "2_nucleus" / "foreground_scores.tif", np.zeros((1, 4, 4), dtype=np.float32))
    widget._pos_dir = pos_dir
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True

    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda path, frame: "summary")
    _stub_ultrack_db_cut_states(widget_class, monkeypatch, widget, (0.5,))
    monkeypatch.setattr(
        widget,
        "_render_hierarchy_cut",
        lambda *args: (np.zeros((4, 4), dtype=np.uint32), "rendered hierarchy cut"),
    )

    widget._refresh_ultrack_db_browser()

    assert "Ultrack DB Preview" in viewer.layers
    assert "Ultrack DB Annotations" not in viewer.layers
    assert "Contour Maps: Nucleus" not in viewer.layers
    assert "Foreground Masks: Nucleus" not in viewer.layers

    widget.deleteLater()
    viewer.close()

# ── DB Browser hierarchy slider: discrete height-index tests ────────────


def _make_ultrack_db_with_heights(db_path: Path, heights: list[float]) -> None:
    import pickle
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import Base
    from ultrack.core.database import NodeDB
    from ultrack.core.segmentation.node import Node
    from ultrack.utils.constants import NO_PARENT

    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = sqla.create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for i, height in enumerate(heights, start=1):
            mask = np.ones((1, 2, 2), dtype=bool)
            bbox = np.array([0, i, i, 1, i + 2, i + 2], dtype=np.int64)
            node_obj = Node.from_mask(time=0, mask=mask, bbox=bbox, node_id=i)
            session.add(
                NodeDB(
                    id=i,
                    t=0,
                    t_node_id=i,
                    t_hier_id=1,
                    z=0,
                    y=i + 1,
                    x=i + 1,
                    area=4,
                    height=float(height),
                    hier_parent_id=NO_PARENT,
                    pickle=pickle.dumps(node_obj),
                )
            )
        session.commit()
        assert session.query(NodeDB.height).distinct().count() == len(set(heights))
    engine.dispose()


def _add_ultrack_node(
    session,
    *,
    node_id: int,
    parent_id: int,
    height: float,
    bbox: tuple[int, int, int, int],
) -> None:
    import pickle
    from ultrack.core.database import NodeDB
    from ultrack.core.segmentation.node import Node

    y0, x0, y1, x1 = bbox
    mask = np.ones((1, y1 - y0, x1 - x0), dtype=bool)
    node_obj = Node.from_mask(
        time=0,
        mask=mask,
        bbox=np.array([0, y0, x0, 1, y1, x1], dtype=np.int64),
        node_id=node_id,
    )
    session.add(
        NodeDB(
            id=node_id,
            t=0,
            t_node_id=node_id,
            t_hier_id=1,
            z=0,
            y=(y0 + y1) / 2,
            x=(x0 + x1) / 2,
            area=int(mask.sum()),
            height=float(height),
            hier_parent_id=parent_id,
            pickle=pickle.dumps(node_obj),
        )
    )


def _make_ultrack_db_with_equal_height_plateau(db_path: Path) -> None:
    pytest.importorskip("ultrack")
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import Base
    from ultrack.utils.constants import NO_PARENT

    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = sqla.create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        _add_ultrack_node(
            session,
            node_id=100,
            parent_id=NO_PARENT,
            height=10.0,
            bbox=(0, 0, 2, 4),
        )
        _add_ultrack_node(
            session,
            node_id=101,
            parent_id=100,
            height=10.0,
            bbox=(0, 0, 2, 2),
        )
        _add_ultrack_node(
            session,
            node_id=102,
            parent_id=100,
            height=10.0,
            bbox=(0, 2, 2, 4),
        )
        _add_ultrack_node(
            session,
            node_id=201,
            parent_id=101,
            height=1.0,
            bbox=(0, 0, 1, 1),
        )
        session.commit()
    engine.dispose()


def test_ultrack_db_hierarchy_slider_uses_frame_cut_states(tmp_path, monkeypatch):
    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    _make_ultrack_db_with_equal_height_plateau(db_path)
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    widget.ultrack_db_hierarchy_slider.setValue(1)
    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda _path, _frame: "ok")

    widget._refresh_ultrack_db_browser()

    assert widget.ultrack_db_hierarchy_slider.minimum() == 0
    assert widget.ultrack_db_hierarchy_slider.maximum() == 2
    assert widget.ultrack_db_hierarchy_slider.value() == 1
    assert "1" in widget.ultrack_db_height_lbl.text()
    assert "10.00" in widget.ultrack_db_height_lbl.text()
    assert set(widget._ultrack_db_label_to_node_id.values()) == {101, 102}

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_hierarchy_cut_keeps_equal_height_intermediate_nodes(tmp_path):
    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    _make_ultrack_db_with_equal_height_plateau(db_path)
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    _labels, status, _probs, label_to_node_id, _node_id_to_label, _annotations = (
        widget._render_hierarchy_cut(db_path, frame=0, h_actual=10.0)
    )

    assert set(label_to_node_id.values()) == {101, 102}
    assert 100 not in label_to_node_id.values()
    assert 201 not in label_to_node_id.values()
    assert "2 segment(s)" in status

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_hierarchy_slider_states_eventually_show_equal_height_parent(tmp_path):
    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    _make_ultrack_db_with_equal_height_plateau(db_path)
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    states = widget._query_hierarchy_cut_states(
        db_path, db_path.stat().st_mtime_ns, frame=0
    )
    rendered_node_ids = []
    for index, state in enumerate(states):
        labels, status, _probs, label_to_node_id, _node_id_to_label, _annotations = (
            widget._render_hierarchy_cut_state(db_path, frame=0, state=state)
        )
        assert labels.size > 0
        assert f"i={index}" not in status
        rendered_node_ids.append(set(label_to_node_id.values()))

    assert rendered_node_ids == [
        {102, 201},
        {101, 102},
        {100},
    ]
    assert {100, 101, 102, 201} == set().union(*rendered_node_ids)

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_hierarchy_states_treat_null_parent_as_root(tmp_path):
    pytest.importorskip("ultrack")
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import Base

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = sqla.create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        _add_ultrack_node(
            session,
            node_id=100,
            parent_id=None,
            height=10.0,
            bbox=(0, 0, 2, 4),
        )
        _add_ultrack_node(
            session,
            node_id=101,
            parent_id=100,
            height=1.0,
            bbox=(0, 0, 2, 2),
        )
        session.commit()
    engine.dispose()

    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    states = widget._query_hierarchy_cut_states(
        db_path, db_path.stat().st_mtime_ns, frame=0
    )

    assert [state.node_ids for state in states] == [(101,), (100,)]

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_hierarchy_slider_clamps_when_cut_states_shrink(tmp_path, monkeypatch):
    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    module = sys.modules[widget_class.__module__]
    widget = widget_class(viewer)
    calls = []

    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    widget.ultrack_db_hierarchy_slider.setValue(3)
    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda _path, _frame: "ok")
    state_sets = [
        (
            module._HierarchyCutState((1,), 0.1),
            module._HierarchyCutState((2,), 0.3),
            module._HierarchyCutState((3,), 0.5),
            module._HierarchyCutState((4,), 0.7),
        ),
        (
            module._HierarchyCutState((5,), 0.2),
            module._HierarchyCutState((6,), 0.6),
        ),
    ]

    def _states(*_args):
        return state_sets[0]

    def _render(_db_path, _frame, state):
        calls.append(state.height)
        return np.zeros((5, 5), dtype=np.uint32), "ok"

    monkeypatch.setattr(widget, "_query_hierarchy_cut_states", _states)
    monkeypatch.setattr(widget, "_render_hierarchy_cut_state", _render)
    widget._refresh_ultrack_db_browser()

    state_sets.pop(0)
    widget._ultrack_db_preview_cache.clear()
    widget._refresh_ultrack_db_browser()

    assert widget.ultrack_db_hierarchy_slider.maximum() == 1
    assert widget.ultrack_db_hierarchy_slider.value() == 1
    assert "1" in widget.ultrack_db_height_lbl.text()
    assert "0.60" in widget.ultrack_db_height_lbl.text()
    assert calls == [0.7, 0.6]

    widget.deleteLater()
    viewer.close()
