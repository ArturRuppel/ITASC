import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import h5py
import numpy as np
import tifffile
from qtpy.QtWidgets import QApplication

from cellflow.napari.meta_plugins import MetaContext, available_meta_plugins
from cellflow.napari.meta_plugins.nls_classification import (
    POSITIVE,
    NLSClassificationPlugin,
)


def _app():
    return QApplication.instance() or QApplication([])


class _FakeLayer:
    def __init__(self, data, name):
        self.data = data
        self.name = name
        self.contour = 0


class _FakeViewer:
    """Minimal viewer recording layer adds, enough for the plugin's overlays."""

    def __init__(self):
        self.layers = []

    def add_image(self, data, **kwargs):
        layer = _FakeLayer(data, kwargs.get("name"))
        self.layers.append(layer)
        return layer

    def add_labels(self, data, **kwargs):
        layer = _FakeLayer(data, kwargs.get("name"))
        self.layers.append(layer)
        return layer


def _write_minimal_position_h5(path, cell_ids):
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5:
        cells = h5.create_group("cells/table")
        cells.create_dataset("cell_id", data=np.asarray(cell_ids, dtype=np.int64))
        cells.create_dataset(
            "class_label",
            data=np.asarray(["old"] * len(cell_ids), dtype=object),
            dtype=string_dtype,
        )
        h5.create_group("cells/measurements")


def _make_position(tmp_path: Path) -> dict:
    labels = np.asarray([[[1, 1, 2, 2]]], dtype=np.uint16)
    nls = np.asarray([[[10.0, 11.0, 100.0, 110.0]]], dtype=np.float32)
    labels_path = tmp_path / "tracked_labels.tif"
    nls_path = tmp_path / "NLS_zavg.tif"
    h5_path = tmp_path / "contact_analysis.h5"
    tifffile.imwrite(labels_path, labels)
    tifffile.imwrite(nls_path, nls)
    _write_minimal_position_h5(h5_path, [1, 2])
    return {
        "id": "p1",
        "position_path": tmp_path,
        "nucleus_tracked_labels_path": labels_path,
        "contact_analysis_path": h5_path,
        "nls_path": nls_path,
    }


def test_plugin_is_registered():
    ids = {cls.plugin_id for cls in available_meta_plugins()}
    assert "nls_classification" in ids


def test_single_position_measures_and_auto_thresholds(tmp_path):
    app = _app()
    viewer = _FakeViewer()
    record = _make_position(tmp_path)

    plugin = NLSClassificationPlugin(viewer=viewer)
    plugin.set_context(MetaContext(records=[record], viewer=viewer))
    plugin._nls_edit.setText(str(record["nls_path"]))
    assert plugin._measure_btn.isEnabled()

    # Drive the post-measurement path directly (the worker just does I/O).
    from cellflow.aggregate_quantification import measure_track_nls_intensity

    labels = tifffile.imread(record["nucleus_tracked_labels_path"])
    measurements = measure_track_nls_intensity(tifffile.imread(record["nls_path"]), labels)
    plugin._on_measure_done((labels[np.newaxis, ...] if labels.ndim == 2 else labels, measurements))
    app.processEvents()

    # Auto threshold lands between the two clusters → track 2 positive.
    assert 11.0 < plugin.current_threshold() < 100.0
    assert plugin._assignments == {1: "negative", 2: POSITIVE}
    # Overlays added: NLS image + positive-nuclei outline of track 2.
    names = {layer.name for layer in viewer.layers}
    assert {"NLS image", "Positive nuclei"} <= names
    outline = next(layer for layer in viewer.layers if layer.name == "Positive nuclei")
    assert set(np.unique(outline.data)) == {0, 2}

    plugin.deleteLater()
    app.processEvents()


def test_dragging_threshold_reclassifies_and_updates_overlay(tmp_path):
    app = _app()
    viewer = _FakeViewer()
    record = _make_position(tmp_path)

    plugin = NLSClassificationPlugin(viewer=viewer)
    plugin.set_context(MetaContext(records=[record], viewer=viewer))
    plugin._nls_edit.setText(str(record["nls_path"]))

    from cellflow.aggregate_quantification import measure_track_nls_intensity

    labels = tifffile.imread(record["nucleus_tracked_labels_path"])
    measurements = measure_track_nls_intensity(tifffile.imread(record["nls_path"]), labels)
    plugin._on_measure_done((labels, measurements))

    # Raise the threshold above both medians → everyone negative.
    plugin._on_spin_changed(500.0)
    app.processEvents()
    assert all(status != POSITIVE for status in plugin._assignments.values())
    outline = next(layer for layer in viewer.layers if layer.name == "Positive nuclei")
    assert not np.any(outline.data)

    # Drop it below both → everyone positive.
    plugin._on_spin_changed(0.0)
    app.processEvents()
    assert all(status == POSITIVE for status in plugin._assignments.values())

    plugin.deleteLater()
    app.processEvents()


def test_apply_writes_classification_to_h5(tmp_path):
    app = _app()
    viewer = _FakeViewer()
    record = _make_position(tmp_path)

    plugin = NLSClassificationPlugin(viewer=viewer)
    plugin.set_context(MetaContext(records=[record], viewer=viewer))
    plugin._nls_edit.setText(str(record["nls_path"]))
    plugin._positive_edit.setText("GFP+")
    plugin._negative_edit.setText("GFP-")

    from cellflow.aggregate_quantification import measure_track_nls_intensity

    labels = tifffile.imread(record["nucleus_tracked_labels_path"])
    measurements = measure_track_nls_intensity(tifffile.imread(record["nls_path"]), labels)
    plugin._on_measure_done((labels, measurements))

    assert plugin._apply_btn.isEnabled()
    plugin._on_apply()

    with h5py.File(record["contact_analysis_path"], "r") as h5:
        cells = h5["cells/table"]
        assert cells["nls_status"].asstr()[:].tolist() == ["negative", "positive"]
        assert cells["class_label"].asstr()[:].tolist() == ["GFP-", "GFP+"]
        meta = h5["cells/measurements/nls_classification"].attrs
        assert meta["positive_label"] == "GFP+"

    plugin.deleteLater()
    app.processEvents()


def test_multiple_positions_disable_measure(tmp_path):
    app = _app()
    viewer = _FakeViewer()
    record = _make_position(tmp_path)

    plugin = NLSClassificationPlugin(viewer=viewer)
    plugin.set_context(MetaContext(records=[record, dict(record)], viewer=viewer))
    assert plugin._record is None
    assert not plugin._measure_btn.isEnabled()
    assert "exactly one" in plugin._scope_lbl.text()

    plugin.deleteLater()
    app.processEvents()
