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
from qtpy.QtWidgets import QApplication, QSizePolicy, QToolButton


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
            "corrections_from_validated_tracks": (
                lambda validated_tracks, tracked_labels: []
            ),
        },
        "cellflow.tracking_ultrack.db_build": {
            "apply_annotations_and_score": lambda *args, **kwargs: None,
            "annotate_database_from_corrections": lambda *args, **kwargs: None,
            "build_atom_union_database": lambda *args, **kwargs: None,
        },
        "cellflow.tracking_ultrack.export": {"export_tracked_labels": lambda *args, **kwargs: None},
        "cellflow.tracking_ultrack.ingest": {
            "_select_solver": lambda: "CBC",
            "_build_ultrack_config": lambda *args, **kwargs: None,
        },
        "cellflow.tracking_ultrack.linking": {"run_linking": lambda *args, **kwargs: iter(())},
        "cellflow.tracking_ultrack.multi_threshold": {
            "build_ultrack_database_from_thresholds": lambda *args, **kwargs: None,
            "build_ultrack_database_from_threshold_pairs": lambda *args, **kwargs: None,
            "build_ultrack_database_from_sources": lambda *args, **kwargs: None,
            "build_ultrack_source_stacks": lambda *args, **kwargs: (
                np.zeros((1, 1, 1, 1), dtype=np.float32),
                np.zeros((1, 1, 1, 1), dtype=np.uint8),
                [],
            ),
            "build_ultrack_source_stacks_from_pairs": lambda *args, **kwargs: (
                np.zeros((1, 1, 1, 1), dtype=np.float32),
                np.zeros((1, 1, 1, 1), dtype=np.uint8),
                [],
            ),
            "preview_ultrack_source_stack_frame": lambda *args, **kwargs: (None, None, 0, []),
            "write_ultrack_source_stacks": lambda *args, **kwargs: [],
        },
        "cellflow.tracking_ultrack.extend": {
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
            "build_consensus_boundary_flow_following": lambda *args, **kwargs: (None, None),
            "CancelledError": type("CancelledError", (Exception,), {}),
        },
    }

    for module_name, attrs in stub_exports.items():
        module = types.ModuleType(module_name)
        if module_name == "cellflow.segmentation":
            segmentation_dir = (
                Path(__file__).resolve().parents[2] / "src" / "cellflow" / "segmentation"
            )
            module.__path__ = [str(segmentation_dir)]
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
    widget._ultrack_db_path = lambda: Path(__file__)
    widget._refresh_ultrack_db_browser = lambda: None

    viewer.add_labels(np.ones((1, 4, 4), dtype=np.uint8), name="[Database] Ultrack DB Preview")
    viewer.add_labels(np.ones((1, 4, 4), dtype=np.uint8), name="[Database] Ultrack DB Selection")
    viewer.add_labels(np.ones((1, 4, 4), dtype=np.uint8), name="[Database] Ultrack DB Annotations")

    assert widget.ultrack_db_browser_section.is_expanded is False

    widget.ultrack_db_active_btn.setChecked(True)

    assert widget.ultrack_db_browser_section.is_expanded is True
    assert widget._ultrack_db_browser_active is True

    widget.ultrack_db_active_btn.setChecked(False)

    assert widget.ultrack_db_browser_section.is_expanded is False
    assert widget._ultrack_db_browser_active is False
    assert "[Database] Ultrack DB Preview" not in viewer.layers
    assert "[Database] Ultrack DB Selection" not in viewer.layers
    assert "[Database] Ultrack DB Annotations" not in viewer.layers

    widget.deleteLater()
    viewer.close()


def test_database_browser_preview_mode_hides_restores_and_cleans_preview_layers():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    widget._ultrack_db_path = lambda: Path(__file__)
    widget._refresh_ultrack_db_browser = lambda: None

    raw = viewer.add_image(np.zeros((1, 4, 4), dtype=np.float32), name="Raw data")
    tracked = viewer.add_labels(np.ones((1, 4, 4), dtype=np.uint8), name="Tracked labels")
    tracked.visible = False

    widget.ultrack_db_active_btn.setChecked(True)

    assert raw.visible is False
    assert tracked.visible is False

    labels = np.ones((1, 4, 4), dtype=np.uint8)
    widget._update_ultrack_db_preview_layer(labels, {}, {})
    widget._update_ultrack_db_annotation_layer(labels, {1: 10}, {10: "REAL"})
    widget._get_ultrack_db_highlight_layer()

    assert "[Database] Ultrack DB Preview" in viewer.layers
    assert "[Database] Ultrack DB Annotations" in viewer.layers
    assert "[Database] Ultrack DB Selection" in viewer.layers

    widget.ultrack_db_active_btn.setChecked(False)

    assert raw.visible is True
    assert tracked.visible is False
    assert "[Database] Ultrack DB Preview" not in viewer.layers
    assert "[Database] Ultrack DB Annotations" not in viewer.layers
    assert "[Database] Ultrack DB Selection" not in viewer.layers

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_annotation_toggle_controls_overlay_layer():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    labels = np.array([[0, 1], [2, 3]], dtype=np.uint32)
    label_to_node_id = {1: 10, 2: 20, 3: 30}
    node_annotations = {10: "REAL", 20: "FAKE", 30: "UNKNOWN"}

    widget.ultrack_db_annotation_check.setChecked(False)
    widget._refresh_ultrack_db_annotation_visualization(
        labels, label_to_node_id, node_annotations,
    )

    assert "[Database] Ultrack DB Annotations" not in viewer.layers

    widget.ultrack_db_annotation_check.setChecked(True)
    widget._refresh_ultrack_db_annotation_visualization(
        labels, label_to_node_id, node_annotations,
    )

    layer = viewer.layers["[Database] Ultrack DB Annotations"]
    overlay = layer.data
    assert set(np.unique(overlay)) == {0, 1, 2, 3}
    np.testing.assert_allclose(
        layer.colormap.map(np.array([0, 1, 2, 3])),
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.75, 0.0, 0.75],
                [1.0, 0.0, 0.0, 0.75],
                [0.5, 0.5, 0.5, 0.45],
            ],
            dtype=np.float32,
        ),
    )

    widget.ultrack_db_annotation_check.setChecked(False)
    widget._refresh_ultrack_db_annotation_visualization(
        labels, label_to_node_id, node_annotations,
    )

    assert "[Database] Ultrack DB Annotations" not in viewer.layers

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_counts_visible_annotations_from_preview():
    widget_class = _load_widget_class()
    labels = np.array(
        [
            [0, 1, 1],
            [2, 0, 3],
            [4, 4, 0],
        ],
        dtype=np.uint32,
    )

    counts = widget_class._ultrack_db_visible_annotation_counts(
        labels,
        {1: 10, 2: 20, 3: 30, 4: 40},
        {10: "REAL", 20: "FAKE", 30: "UNKNOWN", 40: "REAL"},
    )

    assert counts == {"REAL": 2, "FAKE": 1, "UNKNOWN": 1}


def test_database_browser_header_uses_icon_activate_button_distinct_from_run():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert isinstance(widget.ultrack_db_active_btn, QToolButton)
    assert widget.ultrack_db_active_btn.text() == "⏻"
    assert widget.ultrack_db_active_btn.text() != widget.db_run_btn.text()
    assert widget.ultrack_db_browser_header_lbl.text() == "Database Browser"
    assert not widget.ultrack_db_browser_header_lbl.isHidden()
    assert not widget.ultrack_db_browser_section._toggle.isVisible()
    assert not hasattr(widget, "ultrack_db_refresh_btn")

    widget.deleteLater()
    viewer.close()


def test_database_browser_activation_without_database_restores_button_and_section(tmp_path):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    widget._pos_dir = tmp_path / "Position_1"
    widget._pos_dir.mkdir()

    widget.ultrack_db_active_btn.setChecked(True)

    assert widget.ultrack_db_active_btn.isChecked() is False
    assert widget.ultrack_db_browser_section.is_expanded is False
    assert widget._ultrack_db_browser_active is False
    assert "data.db" in widget.ultrack_db_section_status_lbl.text()

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


def _stub_ultrack_db_partition(monkeypatch, widget, *, sizes=(1,), classes=((1,),)):
    """Stub the union-size (vertical) and merge-group (horizontal) queries.

    Tests provide the rendered preview by monkeypatching ``_render_union_partition``,
    which the browser calls with ``(db_path, frame, color_node_ids)``.
    """
    monkeypatch.setattr(widget, "_query_union_sizes", lambda *a: tuple(sizes))
    monkeypatch.setattr(widget, "_query_union_color_classes", lambda *a: tuple(classes))


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
    assert widget.ultrack_db_hierarchy_slider.maximum() == 0
    assert widget.ultrack_db_hierarchy_slider.value() == 0
    assert widget._ultrack_db_slider_row.isHidden() is False

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_sliders_have_step_buttons():
    app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    for row, slider in (
        (widget._ultrack_db_source_slider_row, widget.ultrack_db_source_slider),
        (widget._ultrack_db_slider_row, widget.ultrack_db_hierarchy_slider),
    ):
        buttons = {
            button.objectName(): button
            for button in row.findChildren(QToolButton)
        }
        decrement = buttons["slider_decrement_button"]
        increment = buttons["slider_increment_button"]

        assert not decrement.isEnabled()
        assert not increment.isEnabled()

        slider.setRange(0, 2)
        slider.setValue(1)
        slider.setEnabled(True)

        increment.click()
        assert slider.value() == 2

        decrement.click()
        assert slider.value() == 1

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_exposes_connected_focus_controls():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    assert widget.ultrack_db_connected_focus_check.text() == "Connected focus"
    assert widget.ultrack_db_prob_alpha_check.text() == "Node prob transparency"
    assert not widget.ultrack_db_connected_focus_check.isEnabled()
    assert not hasattr(widget, "ultrack_db_edge_alpha_check")
    assert not hasattr(widget, "ultrack_db_show_validated_check")
    assert not hasattr(widget, "ultrack_db_show_fake_check")

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
    _stub_ultrack_db_partition(monkeypatch, widget)
    monkeypatch.setattr(
        widget,
        "_render_union_partition",
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


def test_ultrack_db_browser_selected_node_status_includes_annotation_probability_and_links(
    tmp_path, monkeypatch,
):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_label_to_node_id = {2: 222}
    widget._ultrack_db_label_probabilities = {2: 0.25}
    widget._ultrack_db_node_annotations = {222: "REAL"}
    widget._ultrack_db_preview_labels = np.array([[0, 2], [0, 0]], dtype=np.uint32)
    monkeypatch.setattr(widget, "_current_t", lambda: 4)
    monkeypatch.setattr(
        widget,
        "_query_ultrack_db_link_annotation_counts",
        lambda _db_path, _node_id: {"REAL": 2, "FAKE": 1, "UNKNOWN": 3},
    )

    widget._select_ultrack_db_preview_label(2, frame=4)

    status = widget.ultrack_db_section_status_lbl.text()
    assert "Selected node 222" in status
    assert "[REAL]" in status
    assert "p=0.250" in status
    assert "links REAL 2" in status
    assert "FAKE 1" in status
    assert "UNKNOWN 3" in status

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_selection_highlight_uses_cyan_contour():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    labels = np.zeros((6, 6), dtype=np.uint32)
    labels[2:5, 1:4] = 7

    widget._update_ultrack_db_highlight(labels, 7)

    layer = viewer.layers["[Database] Ultrack DB Selection"]
    assert layer.visible
    assert len(layer.data) == 1
    assert layer.name == "[Database] Ultrack DB Selection"

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
    _stub_ultrack_db_partition(monkeypatch, widget)
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

    monkeypatch.setattr(widget, "_render_union_partition", _render)

    widget._refresh_ultrack_db_browser()
    assert set(np.unique(viewer.layers["[Database] Ultrack DB Preview"].data)) == {0, 1}

    current["t"] = 3
    widget._refresh_ultrack_db_browser()
    assert set(np.unique(viewer.layers["[Database] Ultrack DB Preview"].data)) == {0, 1}

    current["t"] = 5
    widget._refresh_ultrack_db_browser()
    assert set(np.unique(viewer.layers["[Database] Ultrack DB Preview"].data)) == {0, 1}

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_connected_focus_preserves_node_prob_transparency(
    tmp_path, monkeypatch
):
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
    widget.ultrack_db_prob_alpha_check.setChecked(True)
    widget._ultrack_db_selected_node_id = 222
    widget._ultrack_db_selected_frame = 4
    monkeypatch.setattr(widget, "_current_t", lambda: 5)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda *a: "summary")
    _stub_ultrack_db_partition(monkeypatch, widget)
    monkeypatch.setattr(
        widget,
        "_query_ultrack_db_connected_nodes",
        lambda *a: ({}, {333: 0.5, 444: 1.0}),
    )
    monkeypatch.setattr(
        widget,
        "_render_union_partition",
        lambda *a: (
            np.array([[1, 0], [2, 0]], dtype=np.uint32),
            "rendered",
            {1: 0.2, 2: 0.8},
            {1: 333, 2: 444},
            {333: 1, 444: 2},
        ),
    )

    widget._refresh_ultrack_db_browser()

    layer = viewer.layers["[Database] Ultrack DB Preview"]
    assert layer.data.shape == (2, 2, 4)
    assert layer.data[0, 0, 3] == pytest.approx(0.15)
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
    _stub_ultrack_db_partition(monkeypatch, widget)
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
    monkeypatch.setattr(widget, "_render_union_partition", lambda *a: renders.pop(0))

    widget._refresh_ultrack_db_browser()
    widget._select_ultrack_db_preview_label(1, frame=4)
    first_contour = np.asarray(viewer.layers["[Database] Ultrack DB Selection"].data[0]).copy()
    widget._ultrack_db_preview_cache.clear()
    widget._refresh_ultrack_db_browser()
    second_contour = np.asarray(viewer.layers["[Database] Ultrack DB Selection"].data[0])

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
    _stub_ultrack_db_partition(monkeypatch, widget)
    monkeypatch.setattr(widget, "_query_ultrack_db_connected_nodes", lambda *a: ({}, {}))
    monkeypatch.setattr(
        widget,
        "_render_union_partition",
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

    def _fake_render(path, frame, color_node_ids, union_size=None):
        calls.append((path, frame, color_node_ids))
        return np.zeros((5, 5), dtype=np.uint32), "rendered partition"

    monkeypatch.setattr(widget, "_ultrack_db_summary_text", _fake_summary)
    _stub_ultrack_db_partition(monkeypatch, widget, sizes=(1, 2), classes=((10,),))
    monkeypatch.setattr(widget, "_render_union_partition", _fake_render)

    widget._refresh_ultrack_db_browser()
    widget._refresh_ultrack_db_browser()

    assert len(calls) == 1
    assert calls[0] == (db_path, 0, (10,))
    assert widget.ultrack_db_info_lbl.text() == "3 nodes | 2 links | frame 0: 1 nodes"
    assert widget.ultrack_db_section_status_lbl.text() == "rendered partition"
    assert "[Database] Ultrack DB Preview" in viewer.layers

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_slider_change_preserves_render_cache(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_preview_cache["existing render"] = object()
    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    monkeypatch.setattr(
        widget,
        "_query_union_sizes",
        lambda *_args: (1, 2, 3),
    )

    widget._on_ultrack_db_slider_changed(1)

    assert "existing render" in widget._ultrack_db_preview_cache

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_reuses_summary_text_for_same_database_frame(
    tmp_path, monkeypatch,
):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    widget.ultrack_db_hierarchy_slider.setValue(0)
    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    _stub_ultrack_db_partition(monkeypatch, widget)
    monkeypatch.setattr(
        widget,
        "_render_union_partition",
        lambda *args: (np.zeros((5, 5), dtype=np.uint32), "rendered partition"),
    )
    calls = []

    def _summary(path, frame):
        calls.append((path, frame))
        return "summary stats"

    monkeypatch.setattr(widget, "_ultrack_db_summary_text", _summary)

    widget._refresh_ultrack_db_browser()
    widget._refresh_ultrack_db_browser()

    assert calls == [(db_path, 0)]
    assert widget.ultrack_db_info_lbl.text() == "summary stats"

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_no_movie_initializes_stack_from_middle_frame_only(
    tmp_path, monkeypatch,
):
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    monkeypatch.setattr(widget, "_query_db_frames", lambda *_args: (0, 1, 2))
    _stub_ultrack_db_partition(monkeypatch, widget)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda *_args: "summary")
    rendered_frames = []

    def _render(_db_path, frame, _color_node_ids, _union_size=None):
        rendered_frames.append(frame)
        return (
            np.full((2, 2), frame + 1, dtype=np.uint32),
            f"rendered frame {frame}",
            {},
            {},
            {},
            {},
        )

    monkeypatch.setattr(widget, "_render_union_partition", _render)

    widget._load_full_db_stack(db_path)

    layer = viewer.layers["[Database] Ultrack DB Preview"]
    assert rendered_frames == [1]
    assert layer.data.shape == (3, 2, 2)
    assert not np.any(layer.data[0])
    assert np.all(layer.data[1] == 2)
    assert not np.any(layer.data[2])
    assert widget.ultrack_db_section_status_lbl.text() == (
        "Loaded frame 1/2 from database; other frames render when visited."
    )

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_browser_stack_frame_update_expands_to_fit_larger_frame():
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    viewer.add_labels(
        np.zeros((3, 2, 2), dtype=np.uint32),
        name="[Database] Ultrack DB Preview",
    )

    updated = widget._update_ultrack_db_stack_frame(
        2,
        np.full((4, 5), 7, dtype=np.uint32),
    )

    layer = viewer.layers["[Database] Ultrack DB Preview"]
    assert updated is True
    assert layer.data.shape == (3, 4, 5)
    assert np.all(layer.data[2] == 7)
    assert not np.any(layer.data[0])

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
    _stub_ultrack_db_partition(monkeypatch, widget)

    labels = np.array(
        [
            [1, 0],
            [0, 2],
        ],
        dtype=np.uint32,
    )
    monkeypatch.setattr(
        widget,
        "_render_union_partition",
        lambda *args: (labels, "rendered partition", {1: 0.2, 2: 0.8}),
    )

    widget._refresh_ultrack_db_browser()

    layer = viewer.layers["[Database] Ultrack DB Preview"]
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
    _stub_ultrack_db_partition(monkeypatch, widget)
    monkeypatch.setattr(
        widget,
        "_render_union_partition",
        lambda path, frame, color_node_ids, union_size=None: calls.append(
            (path, frame, color_node_ids)
        )
        or (labels, "rendered partition"),
    )

    widget._refresh_ultrack_db_browser()

    assert calls == [(db_path, 0, (1,))]
    assert widget.ultrack_db_info_lbl.text() == "summary stats"
    assert widget.ultrack_db_section_status_lbl.text() == "rendered partition"
    assert "[Database] Ultrack DB Preview" in viewer.layers

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
    cellpose_dir = pos_dir / "1_cellpose"
    cellpose_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(cellpose_dir / "nucleus_contours.tif", np.zeros((1, 4, 4), dtype=np.float32))
    tifffile.imwrite(cellpose_dir / "nucleus_foreground.tif", np.zeros((1, 4, 4), dtype=np.float32))
    widget._pos_dir = pos_dir
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True

    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda path, frame: "summary")
    _stub_ultrack_db_partition(monkeypatch, widget)
    monkeypatch.setattr(
        widget,
        "_render_union_partition",
        lambda *args: (np.zeros((4, 4), dtype=np.uint32), "rendered partition"),
    )

    widget._refresh_ultrack_db_browser()

    assert "[Database] Ultrack DB Preview" in viewer.layers
    assert "[Database] Ultrack DB Annotations" not in viewer.layers
    assert "Contour Maps: Nucleus" not in viewer.layers
    assert "Foreground Masks: Nucleus" not in viewer.layers

    widget.deleteLater()
    viewer.close()

def test_ultrack_db_summary_counts_annotated_links(tmp_path):
    pytest.importorskip("ultrack")
    import pickle
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import Base, LinkDB, NodeDB, VarAnnotation
    from ultrack.core.segmentation.node import Node
    from ultrack.utils.constants import NO_PARENT
    from cellflow.tracking_ultrack.db_query import summary_text

    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = sqla.create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for node_id, annotation in (
            (1, VarAnnotation.REAL),
            (2, VarAnnotation.FAKE),
            (3, VarAnnotation.UNKNOWN),
        ):
            mask = np.ones((1, 2, 2), dtype=bool)
            node_obj = Node.from_mask(
                time=0,
                mask=mask,
                bbox=np.array([0, 0, 0, 1, 2, 2], dtype=np.int64),
                node_id=node_id,
            )
            session.add(
                NodeDB(
                    id=node_id,
                    t=0,
                    t_node_id=node_id,
                    t_hier_id=1,
                    z=0,
                    y=1,
                    x=1,
                    area=4,
                    height=1.0,
                    hier_parent_id=NO_PARENT,
                    pickle=pickle.dumps(node_obj),
                    node_annot=annotation,
                )
            )
        session.add_all(
            [
                LinkDB(source_id=1, target_id=2, weight=1.0, annotation=VarAnnotation.REAL),
                LinkDB(source_id=2, target_id=3, weight=1.0, annotation=VarAnnotation.FAKE),
                LinkDB(source_id=3, target_id=1, weight=1.0, annotation=VarAnnotation.UNKNOWN),
            ]
        )
        session.commit()
    engine.dispose()

    text = summary_text(db_path, frame=0)

    assert "UNKNOWN 1" in text
    assert "REAL links 1" in text
    assert "FAKE links 1" in text
    assert "UNKNOWN links 1" in text


def _make_node_edges(selected_id=222, selected_t=4):
    from cellflow.tracking_ultrack.db_query import NodeEdge, NodeEdges

    return NodeEdges(
        selected_id=selected_id,
        selected_t=selected_t,
        selected_prob=0.9,
        selected_annot="REAL",
        edges=(
            NodeEdge(111, 0.25, "UNKNOWN", "pred", selected_t - 1, 0.4, "REAL"),
            # neighbor 333 has TWO succ links -> must not collapse to one edge.
            NodeEdge(333, 0.8, "REAL", "succ", selected_t + 1, 0.5, "REAL"),
            NodeEdge(333, 0.2, "FAKE", "succ", selected_t + 1, 0.5, "REAL"),
        ),
    )


def test_node_graph_panel_populates_on_select_and_clears_on_deselect(tmp_path, monkeypatch):
    pytest.importorskip("pyqtgraph")
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "data.db"
    db_path.write_bytes(b"sqlite placeholder")
    monkeypatch.setattr(widget, "_ultrack_db_path", lambda: db_path)
    monkeypatch.setattr(widget, "_query_ultrack_db_node_edges", lambda *a: _make_node_edges())

    widget._ultrack_db_selected_node_id = 222
    widget._refresh_node_graph_panel()

    # selected + pred(111) + succ(333); two succ links kept as separate edges.
    assert widget._ultrack_db_node_graph_node_ids == [222, 111, 333]
    assert widget._ultrack_db_node_graph_edge_count == 3
    assert widget._ultrack_db_node_graph_neighbor_t == {222: 4, 111: 3, 333: 5}

    widget._ultrack_db_selected_node_id = None
    widget._refresh_node_graph_panel()
    assert widget._ultrack_db_node_graph_node_ids == []
    assert widget._ultrack_db_node_graph_edge_count == 0

    widget.deleteLater()
    viewer.close()


def test_node_graph_panel_click_routes_to_select_preview_label(monkeypatch):
    pytest.importorskip("pyqtgraph")
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    monkeypatch.setattr(widget, "_current_t", lambda: 4)
    widget._ultrack_db_node_id_to_label = {333: 5}
    widget._ultrack_db_node_graph_neighbor_t = {333: 4}  # same frame, no slider move
    widget._ultrack_db_node_graph_node_ids = [222, 333]

    routed = {}
    monkeypatch.setattr(
        widget,
        "_select_ultrack_db_preview_label",
        lambda display_label, *, frame=None: routed.update(label=display_label, frame=frame),
    )

    widget._node_graph_select_node(333)

    assert routed == {"label": 5, "frame": 4}

    widget.deleteLater()
    viewer.close()


def test_node_graph_orders_neighbors_by_weight_center_out():
    from cellflow.tracking_ultrack.db_query import NodeEdge

    widget_class = _load_widget_class()
    mapping = {
        10: [NodeEdge(10, 0.2, "UNKNOWN", "succ", 5, 1.0, "REAL")],
        11: [NodeEdge(11, 0.9, "UNKNOWN", "succ", 5, 1.0, "REAL")],
        12: [NodeEdge(12, 0.5, "UNKNOWN", "succ", 5, 1.0, "REAL")],
    }

    order = widget_class._node_graph_pick_neighbors(mapping)
    assert order == [11, 12, 10]  # heaviest link first

    pos = np.zeros((4, 2))
    index_of = {0: 0, 11: 1, 12: 2, 10: 3}
    widget_class._node_graph_place_column(pos, order, index_of, x=1.0)

    # Heaviest neighbor aligned with the selected node row (y=0); lighter ones fan out.
    assert pos[index_of[11]].tolist() == [1.0, 0.0]
    assert pos[index_of[12]][1] == 1.0
    assert pos[index_of[10]][1] == -1.0


def test_node_graph_panel_marks_node_navigated_from(tmp_path, monkeypatch):
    pytest.importorskip("pyqtgraph")
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    db_path = tmp_path / "data.db"
    db_path.write_bytes(b"sqlite placeholder")
    monkeypatch.setattr(widget, "_ultrack_db_path", lambda: db_path)
    monkeypatch.setattr(widget, "_current_t", lambda: 4)

    # Selected 222 with a successor 333 (same frame so no slider move on click).
    from cellflow.tracking_ultrack.db_query import NodeEdge, NodeEdges

    edges_222 = NodeEdges(222, 4, 0.9, "REAL", (
        NodeEdge(333, 0.8, "REAL", "succ", 4, 0.5, "REAL"),
    ))
    # After navigating to 333, its graph has 222 as a (predecessor) neighbor.
    edges_333 = NodeEdges(333, 4, 0.5, "REAL", (
        NodeEdge(222, 0.8, "REAL", "pred", 4, 0.9, "REAL"),
    ))
    graphs = {222: edges_222, 333: edges_333}
    monkeypatch.setattr(
        widget, "_query_ultrack_db_node_edges", lambda _db, nid: graphs[int(nid)]
    )
    # Selecting the preview label just sets the selected id and rebuilds.
    widget._ultrack_db_node_id_to_label = {222: 1, 333: 2}

    def _fake_select(display_label, *, frame=None):
        widget._ultrack_db_selected_node_id = {1: 222, 2: 333}[display_label]
        widget._refresh_node_graph_panel()

    monkeypatch.setattr(widget, "_select_ultrack_db_preview_label", _fake_select)

    widget._ultrack_db_selected_node_id = 222
    widget._refresh_node_graph_panel()
    assert widget._ultrack_db_node_graph_came_from is None

    # Click 333 in the panel -> graph recenters on 333 and marks 222 as "from".
    widget._node_graph_select_node(333)
    assert widget._ultrack_db_selected_node_id == 333
    assert widget._ultrack_db_node_graph_came_from == 222

    widget.deleteLater()
    viewer.close()


def test_node_graph_panel_click_on_hidden_neighbor_reports_status(monkeypatch):
    pytest.importorskip("pyqtgraph")
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)

    monkeypatch.setattr(widget, "_current_t", lambda: 4)
    widget._ultrack_db_node_id_to_label = {}  # neighbor not painted at this frame
    widget._ultrack_db_node_graph_neighbor_t = {999: 4}

    called = []
    monkeypatch.setattr(
        widget, "_select_ultrack_db_preview_label", lambda *a, **k: called.append(a)
    )

    widget._node_graph_select_node(999)

    assert called == []
    assert "hidden" in widget.ultrack_db_section_status_lbl.text().lower()
    assert "999" in widget.ultrack_db_section_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_ultrack_db_size_slider_clamps_when_union_sizes_shrink(tmp_path, monkeypatch):
    db_path = tmp_path / "pos00" / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"sqlite placeholder")
    _app, viewer = _make_viewer()
    widget_class = _load_widget_class()
    widget = widget_class(viewer)
    calls = []

    widget._pos_dir = tmp_path / "pos00"
    widget._ultrack_db_browser_active = True
    widget._ultrack_db_frame_initialized = True
    widget.ultrack_db_hierarchy_slider.setRange(0, 3)
    widget.ultrack_db_hierarchy_slider.setValue(3)
    monkeypatch.setattr(widget, "_current_t", lambda: 0)
    monkeypatch.setattr(widget, "_ultrack_db_summary_text", lambda _path, _frame: "ok")
    size_sets = [(1, 2, 3, 4), (5, 6)]

    def _sizes(*_args):
        return size_sets[0]

    def _render(_db_path, _frame, color_node_ids, _union_size=None):
        calls.append(color_node_ids)
        return np.zeros((5, 5), dtype=np.uint32), "ok"

    monkeypatch.setattr(widget, "_query_union_sizes", _sizes)
    monkeypatch.setattr(widget, "_query_union_color_classes", lambda *_a: ((10,),))
    monkeypatch.setattr(widget, "_render_union_partition", _render)
    widget._refresh_ultrack_db_browser()

    size_sets.pop(0)
    widget._ultrack_db_preview_cache.clear()
    widget._refresh_ultrack_db_browser()

    assert widget.ultrack_db_hierarchy_slider.maximum() == 1
    assert widget.ultrack_db_hierarchy_slider.value() == 1
    assert "N=6" in widget.ultrack_db_height_lbl.text()
    assert "2/2" in widget.ultrack_db_height_lbl.text()
    assert calls == [(10,), (10,)]

    widget.deleteLater()
    viewer.close()
