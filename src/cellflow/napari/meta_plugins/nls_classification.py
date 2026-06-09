"""NLS subpopulation classification plugin for the Contact Analysis studio.

Classifies the cells of a *single* selected position into a labelled **positive**
subpopulation vs **negative** by aggregating a nucleus-localised marker image
(e.g. an NLS reporter) over each nuclear track. The plugin:

* measures one median intensity per nuclear track
  (:func:`cellflow.contact_analysis.measure_track_nls_intensity`);
* auto-places a threshold (:func:`~cellflow.contact_analysis.auto_threshold`) the
  user can drag on a per-track scatter to re-classify live;
* overlays the marker image and the **outlines of positive nuclei** in napari;
* writes the classification back into the contact-analysis ``.h5`` on *Apply*.

Heavy I/O (reading the TIFF stacks + measuring) runs in a ``thread_worker`` so the
UI stays responsive, mirroring the other CellFlow widgets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.contact_analysis import (
    NLSClassificationError,
    auto_threshold,
    classify_by_threshold,
    measure_track_nls_intensity,
    read_position_cell_ids,
    write_nls_classification,
)
from cellflow.contact_analysis.nls_classification import POSITIVE, _read_image_stack
from cellflow.napari.meta_plugins import MetaAnalysisPlugin, MetaContext
from cellflow.napari.ui_style import action_button, status_label

# pyqtgraph backs the interactive scatter + draggable threshold. Guard the import
# so a missing install degrades to a hint instead of breaking plugin discovery.
try:
    import pyqtgraph as pg

    _HAS_PYQTGRAPH = True
except Exception:  # pragma: no cover - exercised only when pyqtgraph is absent
    pg = None
    _HAS_PYQTGRAPH = False

# Colours for the two subpopulations (R, G, B). Positive pops; negative recedes.
_POSITIVE_RGB = (231, 76, 60)
_NEGATIVE_RGB = (120, 130, 140)


class NLSClassificationPlugin(MetaAnalysisPlugin):
    """Interactive NLS-marker subpopulation classification for one position."""

    plugin_id = "nls_classification"
    display_name = "NLS Classification"

    def __init__(self, viewer: object | None = None, parent: QWidget | None = None) -> None:
        super().__init__(viewer=viewer, parent=parent)

        #: The single in-scope catalog record, or None when scope != 1.
        self._record: dict | None = None
        #: Measurement state for the current position.
        self._measurements: dict[int, Any] = {}
        self._track_ids: np.ndarray = np.empty(0, dtype=int)
        self._medians: np.ndarray = np.empty(0, dtype=float)
        self._scatter_x: np.ndarray = np.empty(0, dtype=float)
        self._nucleus_labels: np.ndarray | None = None
        self._assignments: dict[int, str] = {}
        self._measure_worker = None
        #: napari layers we own, refreshed in place across measurements.
        self._image_layer = None
        self._outline_layer = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        self._scope_lbl = QLabel("")
        self._scope_lbl.setWordWrap(True)
        layout.addWidget(self._scope_lbl)

        self._nls_edit = QLineEdit()
        self._nls_edit.setPlaceholderText("NLS / marker image (matches nucleus labels)…")
        self._nls_edit.textChanged.connect(self._update_enabled)
        nls_browse = QPushButton("Browse…")
        action_button(nls_browse)
        nls_browse.clicked.connect(self._browse_nls)
        layout.addLayout(self._labelled_row("NLS image:", self._nls_edit, nls_browse))

        self._positive_edit = QLineEdit("positive")
        self._negative_edit = QLineEdit("negative")
        labels_row = QHBoxLayout()
        labels_row.setContentsMargins(0, 0, 0, 0)
        labels_row.setSpacing(4)
        labels_row.addWidget(QLabel("Positive:"))
        labels_row.addWidget(self._positive_edit, 1)
        labels_row.addWidget(QLabel("Negative:"))
        labels_row.addWidget(self._negative_edit, 1)
        layout.addLayout(labels_row)

        self._measure_btn = QPushButton("Measure & classify")
        action_button(self._measure_btn, expand=True)
        self._measure_btn.clicked.connect(self._on_measure)
        layout.addWidget(self._measure_btn)

        layout.addWidget(self._build_plot())

        thr_row = QHBoxLayout()
        thr_row.setContentsMargins(0, 0, 0, 0)
        thr_row.setSpacing(4)
        thr_row.addWidget(QLabel("Threshold:"))
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setDecimals(3)
        self._threshold_spin.setRange(-1e12, 1e12)
        self._threshold_spin.setEnabled(False)
        self._threshold_spin.valueChanged.connect(self._on_spin_changed)
        thr_row.addWidget(self._threshold_spin, 1)
        layout.addLayout(thr_row)

        self._counts_lbl = QLabel("")
        self._counts_lbl.setWordWrap(True)
        layout.addWidget(self._counts_lbl)

        self._apply_btn = QPushButton("Apply to H5")
        action_button(self._apply_btn, expand=True)
        self._apply_btn.clicked.connect(self._on_apply)
        layout.addWidget(self._apply_btn)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        status_label(self._status_lbl)
        layout.addWidget(self._status_lbl)

        layout.addStretch()

        self._update_enabled()

    # ----------------------------------------------------------------- UI helpers
    @staticmethod
    def _labelled_row(label: str, edit: QWidget, button: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        lbl = QLabel(label)
        lbl.setFixedWidth(70)
        row.addWidget(lbl)
        row.addWidget(edit, 1)
        row.addWidget(button)
        return row

    def _build_plot(self) -> QWidget:
        if not _HAS_PYQTGRAPH:  # pragma: no cover - only without pyqtgraph
            placeholder = QLabel("Scatter unavailable (pyqtgraph not installed).")
            status_label(placeholder, muted=True)
            return placeholder
        self._plot = pg.PlotWidget()
        self._plot.setMinimumHeight(220)
        self._plot.setLabel("left", "Median NLS intensity")
        self._plot.getAxis("bottom").setStyle(showValues=False)
        self._plot.setMouseEnabled(x=False, y=True)
        self._scatter = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None))
        self._plot.addItem(self._scatter)
        self._threshold_line = pg.InfiniteLine(
            angle=0, movable=True, pen=pg.mkPen("y", width=2)
        )
        self._threshold_line.setVisible(False)
        self._plot.addItem(self._threshold_line)
        # Live recolour/counts while dragging; rebuild the (heavier) outline
        # overlay only when the drag finishes.
        self._threshold_line.sigPositionChanged.connect(self._on_line_dragged)
        self._threshold_line.sigPositionChangeFinished.connect(self._on_line_released)
        return self._plot

    # -------------------------------------------------------------- plugin context
    def set_context(self, ctx: MetaContext) -> None:
        if ctx.viewer is not None:
            self.viewer = ctx.viewer
        records = list(ctx.records)
        record = records[0] if len(records) == 1 else None
        if record is not self._record:
            self._record = record
            self._reset_measurement()
            self._prefill_nls_path()
        self._update_scope_label(len(records))
        self._update_enabled()

    def _update_scope_label(self, scope_count: int) -> None:
        if self._record is not None:
            self._scope_lbl.setText(f"Position: {self._record.get('id', '?')}")
        elif scope_count == 0:
            self._scope_lbl.setText("Select a position to classify.")
        else:
            self._scope_lbl.setText("Select exactly one position to classify.")

    def _prefill_nls_path(self) -> None:
        """Default the NLS field to ``<position>/0_input/NLS_zavg.tif`` if present."""
        self._nls_edit.clear()
        if self._record is None:
            return
        position = self._record.get("position_path")
        if not position:
            return
        candidate = Path(position) / "0_input" / "NLS_zavg.tif"
        if candidate.is_file():
            self._nls_edit.setText(str(candidate))

    def _nucleus_labels_path(self) -> Path | None:
        if self._record is None:
            return None
        path = self._record.get("nucleus_tracked_labels_path")
        return Path(path) if path else None

    def _nls_path(self) -> Path | None:
        text = self._nls_edit.text().strip()
        return Path(text) if text else None

    def _update_enabled(self) -> None:
        running = self._measure_worker is not None
        labels_path = self._nucleus_labels_path()
        can_measure = (
            self._record is not None
            and labels_path is not None
            and labels_path.is_file()
            and self._nls_path() is not None
            and not running
        )
        self._measure_btn.setEnabled(bool(can_measure))
        self._apply_btn.setEnabled(bool(self._assignments) and not running)
        self._threshold_spin.setEnabled(self._medians.size > 0)

    # ------------------------------------------------------------------ measuring
    def _reset_measurement(self) -> None:
        self._measurements = {}
        self._track_ids = np.empty(0, dtype=int)
        self._medians = np.empty(0, dtype=float)
        self._scatter_x = np.empty(0, dtype=float)
        self._nucleus_labels = None
        self._assignments = {}
        if _HAS_PYQTGRAPH:
            self._scatter.clear()
            self._threshold_line.setVisible(False)
        self._counts_lbl.setText("")

    def _browse_nls(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select NLS / marker image", self._nls_edit.text(),
            "TIFF (*.tif *.tiff);;All Files (*)",
        )
        if path:
            self._nls_edit.setText(path)
            self._update_enabled()

    def _on_measure(self) -> None:
        nls_path = self._nls_path()
        labels_path = self._nucleus_labels_path()
        if nls_path is None or labels_path is None or not labels_path.is_file():
            self._status_lbl.setText("Status: need a position with nucleus labels and an NLS image.")
            return
        self._status_lbl.setText("Status: measuring per-track intensities…")
        self._measure_worker = object()
        self._update_enabled()

        @thread_worker(
            connect={"returned": self._on_measure_done, "errored": self._on_measure_error}
        )
        def _worker():
            nls = _read_image_stack(nls_path)
            labels = _read_image_stack(labels_path)
            measurements = measure_track_nls_intensity(nls, labels)
            return labels, measurements

        self._measure_worker = _worker()

    def _on_measure_error(self, exc: Exception) -> None:
        self._measure_worker = None
        self._status_lbl.setText(f"Status: error: {exc}")
        self._update_enabled()

    def _on_measure_done(self, result: tuple[np.ndarray, dict]) -> None:
        self._measure_worker = None
        labels, measurements = result
        if not measurements:
            self._status_lbl.setText("Status: no nonzero nuclear tracks found.")
            self._update_enabled()
            return

        self._nucleus_labels = labels
        self._measurements = measurements
        self._track_ids = np.asarray(sorted(measurements), dtype=int)
        self._medians = np.asarray(
            [measurements[int(t)].median_intensity for t in self._track_ids], dtype=float
        )
        rng = np.random.default_rng(0)
        self._scatter_x = rng.uniform(-0.35, 0.35, size=self._track_ids.size)

        medians_map = {int(t): float(m) for t, m in zip(self._track_ids, self._medians)}
        try:
            threshold = auto_threshold(medians_map)
        except NLSClassificationError as exc:
            self._status_lbl.setText(f"Status: cannot auto-threshold: {exc}")
            threshold = float(np.median(self._medians))

        self._ensure_image_layer()
        if _HAS_PYQTGRAPH:
            lo, hi = float(self._medians.min()), float(self._medians.max())
            pad = max((hi - lo) * 0.05, 1e-6)
            self._plot.setYRange(lo - pad, hi + pad)
            self._threshold_line.setBounds((lo - pad, hi + pad))
            self._threshold_line.setVisible(True)
            self._set_threshold(threshold)  # paints scatter + line + outline
        else:  # pragma: no cover - only without pyqtgraph
            self._apply_threshold_value(threshold, rebuild_outline=True)
        self._status_lbl.setText(
            f"Status: measured {self._track_ids.size} track(s); drag the line to tune."
        )
        self._update_enabled()

    # --------------------------------------------------------------- thresholding
    def _set_threshold(self, value: float) -> None:
        """Set the threshold from code, keeping line + spinbox in sync."""
        if _HAS_PYQTGRAPH:
            self._threshold_line.blockSignals(True)
            self._threshold_line.setValue(value)
            self._threshold_line.blockSignals(False)
        self._threshold_spin.blockSignals(True)
        self._threshold_spin.setValue(value)
        self._threshold_spin.blockSignals(False)
        self._apply_threshold_value(value, rebuild_outline=True)

    def current_threshold(self) -> float:
        return float(self._threshold_spin.value())

    def _on_line_dragged(self) -> None:
        value = float(self._threshold_line.value())
        self._threshold_spin.blockSignals(True)
        self._threshold_spin.setValue(value)
        self._threshold_spin.blockSignals(False)
        self._apply_threshold_value(value, rebuild_outline=False)

    def _on_line_released(self) -> None:
        self._apply_threshold_value(float(self._threshold_line.value()), rebuild_outline=True)

    def _on_spin_changed(self, value: float) -> None:
        if _HAS_PYQTGRAPH:
            self._threshold_line.blockSignals(True)
            self._threshold_line.setValue(value)
            self._threshold_line.blockSignals(False)
        self._apply_threshold_value(float(value), rebuild_outline=True)

    def _apply_threshold_value(self, threshold: float, *, rebuild_outline: bool) -> None:
        if not self._measurements:
            return
        self._assignments = classify_by_threshold(self._measurements, threshold)
        positive = sum(s == POSITIVE for s in self._assignments.values())
        negative = len(self._assignments) - positive
        self._counts_lbl.setText(
            f"{positive} {self._positive_edit.text().strip() or 'positive'} / "
            f"{negative} {self._negative_edit.text().strip() or 'negative'}   "
            f"(threshold = {threshold:.4g})"
        )
        self._paint_scatter()
        if rebuild_outline:
            self._refresh_outline_layer()
        self._apply_btn.setEnabled(bool(self._assignments) and self._measure_worker is None)

    def _paint_scatter(self) -> None:
        if not _HAS_PYQTGRAPH or self._track_ids.size == 0:
            return
        brushes = [
            pg.mkBrush(*(_POSITIVE_RGB if self._assignments.get(int(t)) == POSITIVE else _NEGATIVE_RGB))
            for t in self._track_ids
        ]
        self._scatter.setData(x=self._scatter_x, y=self._medians, brush=brushes)

    # ------------------------------------------------------------- napari overlays
    def _positive_track_ids(self) -> list[int]:
        return [int(t) for t, s in self._assignments.items() if s == POSITIVE]

    def _ensure_image_layer(self) -> None:
        if self.viewer is None or self._nucleus_labels is None:
            return
        nls_path = self._nls_path()
        if nls_path is None:
            return
        image = _read_image_stack(nls_path)
        if self._image_layer is not None and self._layer_alive(self._image_layer):
            self._image_layer.data = image
        else:
            self._image_layer = self.viewer.add_image(image, name="NLS image", blending="additive")

    def _refresh_outline_layer(self) -> None:
        if self.viewer is None or self._nucleus_labels is None:
            return
        positive_ids = self._positive_track_ids()
        if positive_ids:
            mask = np.isin(self._nucleus_labels, positive_ids)
            outline = np.where(mask, self._nucleus_labels, 0)
        else:
            outline = np.zeros_like(self._nucleus_labels)
        if self._outline_layer is not None and self._layer_alive(self._outline_layer):
            self._outline_layer.data = outline
        else:
            self._outline_layer = self.viewer.add_labels(
                outline, name="Positive nuclei", opacity=0.9
            )
            # Outline-only rendering; tolerate viewers/stubs without the property.
            try:
                self._outline_layer.contour = 2
            except Exception:  # pragma: no cover - non-napari stub viewers
                pass

    def _layer_alive(self, layer: Any) -> bool:
        """True if *layer* is still present in the viewer (not user-deleted)."""
        layers = getattr(self.viewer, "layers", None)
        if layers is None:
            return True
        try:
            return layer in layers
        except Exception:  # pragma: no cover - defensive
            return False

    # --------------------------------------------------------------------- apply
    def _on_apply(self) -> None:
        if not self._assignments or self._record is None:
            return
        h5_path = self._record.get("contact_analysis_path")
        if not h5_path or not Path(h5_path).is_file():
            self._status_lbl.setText("Status: contact analysis .h5 not found for this position.")
            return
        try:
            cell_ids = read_position_cell_ids(h5_path)
            if not set(int(c) for c in cell_ids).intersection(self._assignments):
                self._status_lbl.setText(
                    "Status: H5 cell IDs do not overlap the classified nuclear tracks."
                )
                return
            write_nls_classification(
                h5_path,
                cell_ids=cell_ids,
                measurements=self._measurements,
                assignments=self._assignments,
                threshold=self.current_threshold(),
                positive_label=self._positive_edit.text().strip() or "positive",
                negative_label=self._negative_edit.text().strip() or "negative",
                nls_path=self._nls_path() or "",
                labels_path=self._nucleus_labels_path() or "",
            )
        except Exception as exc:  # noqa: BLE001 - surface write errors in the UI
            self._status_lbl.setText(f"Status: apply failed: {exc}")
            return
        positive = sum(s == POSITIVE for s in self._assignments.values())
        negative = len(self._assignments) - positive
        self._status_lbl.setText(
            f"Status: wrote {positive} positive / {negative} negative to "
            f"{Path(h5_path).name}."
        )
