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
from qtpy.QtGui import QKeySequence, QShortcut
from qtpy.QtWidgets import QApplication


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
    module = importlib.import_module("cellflow.napari.nucleus_correction_widget")
    return module.NucleusCorrectionWidget, module


def _frame_view_2d(arr: np.ndarray, t: int) -> np.ndarray | None:
    if arr.ndim < 3 or t < 0 or t >= arr.shape[0]:
        return None
    view = arr[t]
    while view.ndim > 2:
        if view.shape[0] != 1:
            return None
        view = view[0]
    return view


def _make_widget(viewer, pos_dir: Path | None = None):
    widget_class, module = _load_widget_class()
    widget = widget_class(viewer)
    widget._pos_dir = pos_dir

    def _path(*parts: str) -> Path | None:
        return pos_dir.joinpath(*parts) if pos_dir is not None else None

    widget._tracked_path = lambda: _path("2_nucleus", "tracked_labels.tif")
    widget._cell_zavg_path = lambda: _path("0_input", "cell_zavg.tif")
    widget._nucleus_zavg_path = lambda: _path("0_input", "nucleus_zavg.tif")
    widget._cell_prob_zavg_path = lambda: _path("1_cellpose", "cell_prob_zavg.tif")
    widget._nucleus_prob_zavg_path = lambda: _path("1_cellpose", "nucleus_prob_zavg.tif")
    widget._nls_zavg_path = lambda: _path("0_input", "NLS_zavg.tif")
    widget._ultrack_db_path = lambda: _path("2_nucleus", "ultrack_workdir", "data.db")
    widget._current_t = lambda: int(viewer.dims.current_step[0])
    widget._frame_view_2d = _frame_view_2d

    def _current_cell_ids(t: int) -> set[int]:
        layer = widget._correction_tracked_layer()
        if layer is None:
            return set()
        frame = widget._frame_view_2d(np.asarray(layer.data), t)
        if frame is None:
            return set()
        return set(int(value) for value in np.unique(frame)) - {0}

    widget._current_cell_ids = _current_cell_ids
    return widget, module


def test_correction_widget_uses_explicit_position_and_refresh_callbacks(tmp_path):
    _app, viewer = _make_viewer()
    widget_class, _module = _load_widget_class()
    current = {"pos_dir": tmp_path / "pos00"}
    refresh_calls = []
    widget = widget_class(
        viewer,
        pos_dir_provider=lambda: current["pos_dir"],
        refresh_refinement_callback=lambda: refresh_calls.append("refresh"),
    )

    assert widget._pos_dir == current["pos_dir"]
    current["pos_dir"] = tmp_path / "pos01"
    assert widget._pos_dir == current["pos_dir"]

    widget._refresh_refinement_widget()

    assert refresh_calls == ["refresh"]
    with pytest.raises(AttributeError):
        getattr(widget, "_missing_workflow_fallback_probe")

    widget.deleteLater()
    viewer.close()


def test_validated_overlay_uses_green_fill_at_default_opacity_below_spotlight():
    _app, viewer = _make_viewer()
    widget, _module = _make_widget(viewer)

    viewer.add_labels(np.array([[[0, 1], [1, 0]]], dtype=np.uint8), name="Tracked: Nucleus")
    viewer.add_image(np.zeros((2, 2, 4), dtype=np.float32), name="[Correction] CellSpotlight", rgb=True)
    widget._add_validated_overlay(np.array([[[0, 1], [0, 0]]], dtype=np.uint8))

    layer = viewer.layers["[Correction] Validated: Nucleus"]
    color = layer.get_color(1)

    assert layer.contour == 0
    assert layer.opacity == 0.4
    assert np.allclose(color[:3], [0.0, 1.0, 0.0], atol=1e-6)
    assert color[3] == 1.0
    assert viewer.layers.index("[Correction] Validated: Nucleus") < viewer.layers.index("[Correction] CellSpotlight")

    widget.deleteLater()
    viewer.close()


def test_nucleus_correction_button_removes_unvalidated_label_instances(tmp_path):
    from cellflow.database.validation import validate_track

    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos000"
    widget, _module = _make_widget(viewer, pos_dir)
    labels = np.zeros((2, 5, 5), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 7
    labels[0, 3:5, 3:5] = 9
    labels[1, 1:3, 1:3] = 7
    labels[1, 3:5, 3:5] = 11
    viewer.add_labels(labels, name="Tracked: Nucleus")
    validate_track(pos_dir, 7, [0])

    widget.remove_unvalidated_btn.click()

    edited = np.asarray(viewer.layers["Tracked: Nucleus"].data)
    assert np.all(edited[0, 1:3, 1:3] == 7)
    assert not np.any(edited[0, 3:5, 3:5] == 9)
    assert not np.any(edited[1] == 7)
    assert not np.any(edited[1] == 11)
    assert "Removed unvalidated labels" in widget.correction_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_validate_track_button_writes_validated_corrections_for_each_present_frame(tmp_path):
    from cellflow.database.validation import read_corrections, read_validated_tracks

    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget, _module = _make_widget(viewer, pos_dir)
    labels = np.zeros((3, 8, 8), dtype=np.uint32)
    labels[0, 1:3, 1:3] = 5
    labels[2, 4:6, 4:6] = 5
    viewer.add_labels(labels, name="Tracked: Nucleus")
    widget.correction_widget._selected_label = 5
    viewer.dims.current_step = (0, 0, 0)

    widget.validate_track_btn.click()

    corrections = read_corrections(pos_dir)
    assert [(c.cell_id, c.t, c.kind) for c in corrections] == [
        (5, 0, "validated"),
        (5, 2, "validated"),
    ]
    assert read_validated_tracks(pos_dir) == {5: {0, 2}}
    assert "validated track" in widget.correction_status_lbl.text().lower()

    widget.deleteLater()
    viewer.close()


def test_anchor_here_button_writes_anchor_correction_at_selected_cell_centroid(tmp_path):
    from cellflow.database.validation import read_corrections

    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget, _module = _make_widget(viewer, pos_dir)
    labels = np.zeros((2, 8, 8), dtype=np.uint32)
    labels[1, 4:6, 1:4] = 3
    viewer.add_labels(labels, name="Tracked: Nucleus")
    widget.correction_widget._selected_label = 3
    viewer.dims.current_step = (1, 0, 0)

    widget.anchor_here_btn.click()

    corrections = read_corrections(pos_dir)
    assert [(c.cell_id, c.t, c.kind) for c in corrections] == [(3, 1, "anchor")]
    assert corrections[0].y == pytest.approx(4.5)
    assert corrections[0].x == pytest.approx(2.0)
    assert "anchor" in widget.correction_status_lbl.text().lower()

    widget.deleteLater()
    viewer.close()


def test_b_shortcut_writes_anchor_correction_at_selected_cell_centroid(tmp_path):
    from cellflow.database.validation import read_corrections

    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget, _module = _make_widget(viewer, pos_dir)
    labels = np.zeros((2, 8, 8), dtype=np.uint32)
    labels[1, 4:6, 1:4] = 3
    viewer.add_labels(labels, name="Tracked: Nucleus")
    widget.correction_widget._selected_label = 3
    viewer.dims.current_step = (1, 0, 0)
    shortcut = next(
        shortcut
        for shortcut in widget.findChildren(QShortcut)
        if shortcut.key().toString(QKeySequence.SequenceFormat.PortableText) == "B"
    )

    shortcut.activated.emit()

    corrections = read_corrections(pos_dir)
    assert [(c.cell_id, c.t, c.kind) for c in corrections] == [(3, 1, "anchor")]
    assert corrections[0].y == pytest.approx(4.5)
    assert corrections[0].x == pytest.approx(2.0)

    widget.deleteLater()
    viewer.close()


def test_b_shortcut_toggles_existing_anchor_off(tmp_path):
    from cellflow.database.validation import read_corrections

    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget, _module = _make_widget(viewer, pos_dir)
    labels = np.zeros((2, 8, 8), dtype=np.uint32)
    labels[1, 4:6, 1:4] = 3
    viewer.add_labels(labels, name="Tracked: Nucleus")
    widget.correction_widget._selected_label = 3
    viewer.dims.current_step = (1, 0, 0)
    shortcut = next(
        shortcut
        for shortcut in widget.findChildren(QShortcut)
        if shortcut.key().toString(QKeySequence.SequenceFormat.PortableText) == "B"
    )

    shortcut.activated.emit()
    shortcut.activated.emit()

    assert read_corrections(pos_dir) == []
    assert "unanchored" in widget.correction_status_lbl.text().lower()

    widget.deleteLater()
    viewer.close()


def test_correction_activate_button_expands_activates_and_deactivates_layers(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "Position_1"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    tracked = np.zeros((1, 4, 4), dtype=np.uint32)
    tracked[0, 1:3, 1:3] = 1
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", tracked)
    widget, _module = _make_widget(viewer, pos_dir)

    assert widget.correction_mode_section.is_expanded is False
    assert widget.toolbar.isHidden() is True

    widget.correction_active_btn.setChecked(True)

    assert widget.correction_mode_section.is_expanded is True
    assert widget.toolbar.isHidden() is False
    assert widget.correction_widget._layer is viewer.layers["[Correction] Nucleus Labels"]
    assert "[Correction] CorrectionDraw" in viewer.layers
    assert "[Correction] CellHighlight" in viewer.layers
    assert "[Correction] CellSpotlight" in viewer.layers

    widget.correction_active_btn.setChecked(False)

    assert widget.correction_mode_section.is_expanded is False
    assert widget.toolbar.isHidden() is True
    assert widget.correction_widget._layer is None
    assert "[Correction] CorrectionDraw" not in viewer.layers
    assert "[Correction] CellHighlight" not in viewer.layers
    assert "[Correction] CellSpotlight" not in viewer.layers
    assert "[Correction] Nucleus Labels" not in viewer.layers

    widget.deleteLater()
    viewer.close()


def test_correction_activation_loads_owned_layers_from_disk(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "Position_1"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    (pos_dir / "0_input").mkdir(parents=True)
    (pos_dir / "1_cellpose").mkdir(parents=True)
    tracked = np.zeros((2, 4, 5), dtype=np.uint32)
    tracked[0, 1:3, 1:3] = 7
    tracked[1, 2:4, 2:4] = 7
    cell_raw_zavg = np.full((2, 4, 5), -99, dtype=np.float32)
    nucleus_raw_zavg = np.full((2, 4, 5), -77, dtype=np.float32)
    cell_prob_zavg = np.array(
        [
            np.full((4, 5), 0.25, dtype=np.float32),
            np.full((4, 5), 0.75, dtype=np.float32),
        ],
        dtype=np.float32,
    )
    nucleus_prob_zavg = np.array(
        [
            np.full((4, 5), 0.35, dtype=np.float32),
            np.full((4, 5), 0.65, dtype=np.float32),
        ],
        dtype=np.float32,
    )
    nls = np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5)
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", tracked)
    tifffile.imwrite(pos_dir / "0_input" / "cell_zavg.tif", cell_raw_zavg)
    tifffile.imwrite(pos_dir / "0_input" / "nucleus_zavg.tif", nucleus_raw_zavg)
    tifffile.imwrite(pos_dir / "1_cellpose" / "cell_prob_zavg.tif", cell_prob_zavg)
    tifffile.imwrite(pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif", nucleus_prob_zavg)
    tifffile.imwrite(pos_dir / "0_input" / "NLS_zavg.tif", nls)

    stale = viewer.add_labels(np.ones((1, 2, 2), dtype=np.uint32), name="[Correction] stale")
    stale.visible = True
    existing = viewer.add_image(np.ones((4, 5), dtype=np.float32), name="Existing")
    existing.visible = True
    widget, _module = _make_widget(viewer, pos_dir)

    widget.correction_active_btn.setChecked(True)

    assert "[Correction] stale" not in viewer.layers
    assert "[Correction] Nucleus Labels" in viewer.layers
    assert "[Correction] Nucleus tracks" in viewer.layers
    assert "[Correction] Cell z-avg" in viewer.layers
    assert "[Correction] Nucleus z-avg" in viewer.layers
    assert "[Correction] NLS z-avg" in viewer.layers
    assert viewer.layers["Existing"].visible is False
    assert widget.correction_widget._layer is viewer.layers["[Correction] Nucleus Labels"]
    assert widget.correction_mode_section.is_expanded is True
    assert set(widget._correction_owned_layers) == {
        "[Correction] Nucleus Labels",
        "[Correction] Nucleus tracks",
        "[Correction] Cell z-avg",
        "[Correction] Nucleus z-avg",
        "[Correction] NLS z-avg",
    }
    label_layer = viewer.layers["[Correction] Nucleus Labels"]
    np.testing.assert_array_equal(label_layer.data, tracked)
    assert label_layer.blending == "additive"
    track_layer = viewer.layers["[Correction] Nucleus tracks"]
    assert track_layer.rgb is True
    assert track_layer.blending == "additive"
    assert track_layer.opacity == 0.9
    assert track_layer.data.shape == (2, 4, 5, 4)
    assert np.count_nonzero(track_layer.data[1, :, :, 3]) > 0
    assert np.max(track_layer.data[..., :3]) < 255
    assert viewer.layers["[Correction] Cell z-avg"].blending == "minimum"
    assert viewer.layers["[Correction] Nucleus z-avg"].blending == "minimum"
    assert viewer.layers["[Correction] NLS z-avg"].blending == "minimum"
    assert viewer.layers["[Correction] Nucleus z-avg"].colormap.name == "I Orange"
    assert viewer.layers["[Correction] NLS z-avg"].colormap.name == "I Blue"
    image_indices = [
        viewer.layers.index("[Correction] Cell z-avg"),
        viewer.layers.index("[Correction] Nucleus z-avg"),
        viewer.layers.index("[Correction] NLS z-avg"),
    ]
    assert max(image_indices) < viewer.layers.index("[Correction] Nucleus Labels")
    assert max(image_indices) < viewer.layers.index("[Correction] Nucleus tracks")
    np.testing.assert_allclose(
        viewer.layers["[Correction] Cell z-avg"].data,
        cell_prob_zavg,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        viewer.layers["[Correction] Nucleus z-avg"].data,
        nucleus_prob_zavg,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(viewer.layers["[Correction] NLS z-avg"].data, nls)

    widget.deleteLater()
    viewer.close()


def test_correction_deactivation_removes_registered_layers_and_restores_viewer_state(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "Position_1"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    tracked = np.zeros((1, 4, 4), dtype=np.uint32)
    tracked[0, 1:3, 1:3] = 5
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", tracked)

    existing = viewer.add_image(np.ones((4, 4), dtype=np.float32), name="Existing")
    unrelated_prefixed = viewer.add_image(np.ones((4, 4), dtype=np.float32), name="[Correction] User Layer")
    existing.visible = True
    unrelated_prefixed.visible = True
    viewer.layers.selection.active = existing
    widget, _module = _make_widget(viewer, pos_dir)

    widget.correction_active_btn.setChecked(True)
    widget._add_validated_overlay(np.ones_like(tracked, dtype=np.uint8))
    assert "[Correction] Validated: Nucleus" in viewer.layers
    widget.correction_active_btn.setChecked(False)

    assert "[Correction] Nucleus Labels" not in viewer.layers
    assert "[Correction] Validated: Nucleus" not in viewer.layers
    assert "[Correction] User Layer" in viewer.layers
    assert viewer.layers["Existing"].visible is True
    assert viewer.layers["[Correction] User Layer"].visible is True
    assert viewer.layers.selection.active is existing
    assert widget._correction_owned_layers == set()
    assert widget.correction_widget._layer is None
    assert widget.correction_mode_section.is_expanded is False

    widget.deleteLater()
    viewer.close()


def test_correction_activation_without_tracked_file_restores_button_and_viewer(tmp_path):
    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "Position_1"
    pos_dir.mkdir()
    existing = viewer.add_image(np.ones((4, 4), dtype=np.float32), name="Existing")
    existing.visible = True
    widget, _module = _make_widget(viewer, pos_dir)

    widget.correction_active_btn.setChecked(True)

    assert widget.correction_active_btn.isChecked() is False
    assert widget.correction_mode_section.is_expanded is False
    assert widget.toolbar.isHidden() is True
    assert viewer.layers["Existing"].visible is True
    assert widget._correction_owned_layers == set()
    assert "No tracked labels file found" in widget.correction_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_correction_save_writes_correction_owned_tracked_layer(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "Position_1"
    (pos_dir / "2_nucleus").mkdir(parents=True)
    original = np.zeros((2, 4, 4), dtype=np.uint32)
    tifffile.imwrite(pos_dir / "2_nucleus" / "tracked_labels.tif", original)
    widget, _module = _make_widget(viewer, pos_dir)
    widget.correction_active_btn.setChecked(True)

    edited = np.zeros((2, 4, 4), dtype=np.uint32)
    edited[1, 1:3, 1:3] = 9
    viewer.layers["[Correction] Nucleus Labels"].data = edited
    widget.save_tracked_btn.click()

    np.testing.assert_array_equal(
        tifffile.imread(pos_dir / "2_nucleus" / "tracked_labels.tif"),
        edited,
    )
    assert "Saved 2 frame(s)" in widget.correction_status_lbl.text()

    widget.deleteLater()
    viewer.close()


def test_correction_shortcuts_are_installed_on_nucleus_correction_widget():
    _app, viewer = _make_viewer()
    widget, _module = _make_widget(viewer)

    shortcut_keys = {
        shortcut.key().toString(QKeySequence.SequenceFormat.PortableText)
        for shortcut in widget.findChildren(QShortcut)
    }
    assert {"A", "D", "Q", "E", "B"} <= shortcut_keys

    widget.deleteLater()
    viewer.close()


def test_extend_fails_clearly_if_db_missing(tmp_path):
    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    pos_dir.mkdir()
    widget, _module = _make_widget(viewer, pos_dir)
    viewer.add_labels(np.zeros((2, 8, 8), dtype=np.uint32), name="Tracked: Nucleus")

    widget._on_extend(direction="forward")

    text = widget.correction_status_lbl.text().lower()
    assert "data.db" in text or "database" in text

    widget.deleteLater()
    viewer.close()


def test_extend_passes_weight_parameters_to_db_tracker(tmp_path, monkeypatch):
    from cellflow.database.validation import validate_track

    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    db_path = pos_dir / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget, module = _make_widget(viewer, pos_dir)
    validate_track(pos_dir, 9, [1])

    labels = np.zeros((2, 12, 12), dtype=np.uint32)
    labels[0, 3:6, 3:6] = 7
    viewer.add_labels(labels, name="Tracked: Nucleus")
    widget.correction_widget._selected_label = 7

    widget.extend_max_dist_spin.setValue(31.0)
    widget.extend_area_weight_spin.setValue(0.7)
    widget.extend_iou_weight_spin.setValue(1.5)
    widget.extend_distance_weight_spin.setValue(0.2)
    widget.extend_overlap_penalty_spin.setValue(2.0)
    widget.extend_greedy_overwrite_check.setChecked(True)

    captured = {}

    def fake_extend_track_from_db(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setitem(
        module._DEFAULT_DEPENDENCIES,
        "extend_track_from_db",
        fake_extend_track_from_db,
    )

    widget._on_extend(direction="forward")

    assert captured["d_max"] == pytest.approx(31.0)
    assert captured["area_weight"] == pytest.approx(0.7)
    assert captured["iou_weight"] == pytest.approx(1.5)
    assert captured["distance_weight"] == pytest.approx(0.2)
    assert captured["overlap_penalty"] == pytest.approx(2.0)
    assert captured["greedy_overwrite"] is True
    assert captured["validated_tracks"] == {9: {1}}

    widget.deleteLater()
    viewer.close()


def test_extend_greedy_overwrite_paints_source_and_overwrites_non_validated(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    db_path = pos_dir / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget, module = _make_widget(viewer, pos_dir)

    labels = np.zeros((2, 32, 32), dtype=np.uint32)
    labels[0, 5:10, 5:10] = 7
    labels[1, 6:11, 6:11] = 9
    viewer.add_labels(labels, name="Tracked: Nucleus")
    widget.correction_widget._selected_label = 7
    widget.extend_greedy_overwrite_check.setChecked(True)

    source_mask = np.zeros((32, 32), dtype=bool)
    source_mask[6:11, 6:11] = True

    result = types.SimpleNamespace(
        target_frame=1,
        candidate_label=101,
        candidate_partition=0,
        mask_2d=source_mask,
        bbox=(6, 6, 11, 11),
        centroid_distance=1.0,
        area_ratio=1.0,
        centroid_corrected_iou=1.0,
        existing_overlap=1.0,
        assignments=(types.SimpleNamespace(cell_id=7, mask_2d=source_mask),),
    )
    monkeypatch.setitem(
        module._DEFAULT_DEPENDENCIES,
        "extend_track_from_db",
        lambda **_kwargs: result,
    )

    widget._on_extend(direction="forward")

    frame = viewer.layers["Tracked: Nucleus"].data[1]
    assert np.all(frame[6:11, 6:11] == 7)
    status = widget.correction_status_lbl.text()
    assert "reassigned" not in status

    widget.deleteLater()
    viewer.close()


def test_extend_greedy_overwrite_preserves_validated_cells(tmp_path, monkeypatch):
    _app, viewer = _make_viewer()
    pos_dir = tmp_path / "pos00"
    db_path = pos_dir / "2_nucleus" / "ultrack_workdir" / "data.db"
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(b"sqlite placeholder")
    widget, module = _make_widget(viewer, pos_dir)

    labels = np.zeros((2, 32, 32), dtype=np.uint32)
    labels[0, 5:10, 5:10] = 7
    labels[1, 6:11, 6:11] = 9
    viewer.add_labels(labels, name="Tracked: Nucleus")
    widget.correction_widget._selected_label = 7
    widget.extend_greedy_overwrite_check.setChecked(True)

    source_mask = np.zeros((32, 32), dtype=bool)
    source_mask[6:11, 6:11] = True

    result = types.SimpleNamespace(
        target_frame=1,
        candidate_label=101,
        candidate_partition=0,
        mask_2d=source_mask,
        bbox=(6, 6, 11, 11),
        centroid_distance=1.0,
        area_ratio=1.0,
        centroid_corrected_iou=1.0,
        existing_overlap=1.0,
        assignments=(types.SimpleNamespace(cell_id=7, mask_2d=source_mask),),
    )
    monkeypatch.setitem(
        module._DEFAULT_DEPENDENCIES,
        "extend_track_from_db",
        lambda **_kwargs: result,
    )
    monkeypatch.setattr(module, "read_validated_tracks", lambda _pos_dir: {9: {1}})

    widget._on_extend(direction="forward")

    frame = viewer.layers["Tracked: Nucleus"].data[1]
    assert np.all(frame[6:11, 6:11] == 9)

    widget.deleteLater()
    viewer.close()
