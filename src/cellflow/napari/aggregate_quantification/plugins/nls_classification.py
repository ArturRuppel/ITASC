"""NLS subpopulation classification plugin for the Aggregate Quantification studio.

Classifies cells into a labelled **positive** subpopulation vs **negative** by
aggregating a nucleus-localised marker image (e.g. an NLS reporter) over each
nuclear track. It runs in two modes, driven by the catalogue selection:

* **single position** — a single stateful action button first reads *Classify*:
  it measures one intensity scalar (90th pct of per-frame medians) per nuclear track
  (:func:`cellflow.aggregate_quantification.measure_track_nls_intensity`),
  auto-places a threshold (:func:`~cellflow.aggregate_quantification.auto_threshold`),
  and reveals a results pane with a per-track scatter the user can drag to
  re-classify live, overlaying the marker image and the **outlines of positive
  nuclei** in napari. The button then reads *Apply to H5* and writes the
  classification back into the contact-analysis ``.h5``;
* **multiple positions** — the interactive scatter does not apply, so the same
  button reads *Classify & apply to all H5* and batch-classifies every selected
  position with its own auto threshold
  (:func:`~cellflow.aggregate_quantification.patch_position_contact_analysis_nls_classes`).

The marker-image field accepts a path **relative to each position directory**
(e.g. ``0_input/NLS_zavg.tif``), so one entry resolves per-position across a
batch; absolute paths are used verbatim.

Heavy I/O (reading the TIFF stacks + measuring) runs in a ``thread_worker`` so the
UI stays responsive, mirroring the other CellFlow widgets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from cellflow.aggregate_quantification import (
    NLSClassificationError,
    auto_threshold,
    classify_by_threshold,
    measure_track_nls_intensity,
    patch_position_contact_analysis_nls_classes,
    read_position_cell_ids,
    write_nls_classification,
)
from cellflow.aggregate_quantification.contacts.nls_classification import POSITIVE, _read_image_stack
from cellflow.napari.aggregate_quantification.plugins import AnalysisContext, AnalysisPlugin
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


class NLSClassificationPlugin(AnalysisPlugin):
    """Interactive NLS-marker subpopulation classification for one position."""

    plugin_id = "nls_classification"
    display_name = "NLS Classification"
    # Per-position; needs nucleus labels to measure marker intensity over tracks.
    # Names a PositionInputs field (not the catalogue-record key).
    requires = ("nucleus_labels_path",)

    def __init__(self, viewer: object | None = None, parent: QWidget | None = None) -> None:
        super().__init__(viewer=viewer, parent=parent)

        #: All in-scope catalog records (>1 puts the plugin in batch mode).
        self._records: list[dict] = []
        #: The single in-scope catalog record, or None when scope != 1.
        self._record: dict | None = None
        #: Measurement state for the current position.
        self._measurements: dict[int, Any] = {}
        self._track_ids: np.ndarray = np.empty(0, dtype=int)
        self._intensities: np.ndarray = np.empty(0, dtype=float)
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

        # --------------------------------------------------------- top controls
        # Always-visible controls, docked at the top at their natural height so
        # the results window below can sit tightly beneath them.
        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)

        self._scope_lbl = QLabel("")
        self._scope_lbl.setWordWrap(True)
        top_layout.addWidget(self._scope_lbl)

        self._nls_edit = QLineEdit()
        self._nls_edit.setPlaceholderText(
            "NLS / marker image — absolute, or relative to each position (e.g. 0_input/NLS_zavg.tif)…"
        )
        self._nls_edit.textChanged.connect(self._on_nls_text_changed)
        nls_browse = QPushButton("Browse…")
        action_button(nls_browse)
        nls_browse.clicked.connect(self._browse_nls)
        top_layout.addLayout(self._labelled_row("NLS image:", self._nls_edit, nls_browse))

        self._positive_edit = QLineEdit("positive")
        self._negative_edit = QLineEdit("negative")
        labels_row = QHBoxLayout()
        labels_row.setContentsMargins(0, 0, 0, 0)
        labels_row.setSpacing(4)
        labels_row.addWidget(QLabel("Positive:"))
        labels_row.addWidget(self._positive_edit, 1)
        labels_row.addWidget(QLabel("Negative:"))
        labels_row.addWidget(self._negative_edit, 1)
        top_layout.addLayout(labels_row)

        # One stateful button: *Classify* → *Apply to H5* (single position) or
        # *Classify & apply to all H5* (batch). Label + action set in _update_enabled.
        self._action_btn = QPushButton("Classify")
        action_button(self._action_btn, expand=True)
        self._action_btn.clicked.connect(self._on_action)
        top_layout.addWidget(self._action_btn)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        status_label(self._status_lbl)
        top_layout.addWidget(self._status_lbl)

        layout.addWidget(top)

        # ----------------------------------------------------- results window
        # Threshold spinbox + scatter + counts: meaningful only after a
        # classification, so they live in a pane hidden until then. To dock that
        # pane tightly under the controls yet keep it resizable from its *bottom*
        # edge, it is the top child of a vertical splitter whose bottom child is
        # an empty spacer the scatter grows into — so the handle is the scatter's
        # own bottom edge.
        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.setChildrenCollapsible(False)
        layout.addWidget(self._splitter, 1)

        self._results_pane = QWidget()
        results_layout = QVBoxLayout(self._results_pane)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(6)

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
        results_layout.addLayout(thr_row)

        results_layout.addWidget(self._build_plot(), 1)

        self._counts_lbl = QLabel("")
        self._counts_lbl.setWordWrap(True)
        results_layout.addWidget(self._counts_lbl)

        self._splitter.addWidget(self._results_pane)
        self._results_pane.setVisible(False)

        # Empty bottom pane: dragging the handle above it (the scatter's bottom
        # edge) resizes the scatter; this spacer absorbs the freed space.
        self._splitter.addWidget(QWidget())

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
        self._plot.setMinimumHeight(140)
        self._plot.setLabel("left", "Track NLS intensity (90th pct of per-frame median)")
        # The x position is meaningless jitter, so the bottom axis carries no
        # information — hide it entirely rather than leave a clipped strip of
        # ticks/label, and give the scatter that vertical room back.
        self._plot.hideAxis("bottom")
        self._plot.setMouseEnabled(x=False, y=True)
        # Pin a padded x range so edge markers are never clipped against the border.
        self._plot.setXRange(-0.6, 0.6, padding=0)
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
    def set_context(self, ctx: AnalysisContext) -> None:
        if ctx.viewer is not None:
            self.viewer = ctx.viewer
        records = list(ctx.records)
        self._records = records
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
            self._scope_lbl.setText(
                f"{scope_count} positions selected — batch classify each with its "
                "own auto threshold (no interactive tuning)."
            )

    def _prefill_nls_path(self) -> None:
        """Default the NLS field to the *relative* ``0_input/NLS_zavg.tif`` if present.

        Keep any path the user already typed — a relative entry is what makes one
        field resolve across every position in a batch, so don't clobber it.
        """
        if self._nls_edit.text().strip():
            return
        if self._record is None:
            return
        position = self._record.get("position_path")
        if not position:
            return
        relative = Path("0_input") / "NLS_zavg.tif"
        if (Path(position) / relative).is_file():
            self._nls_edit.setText(str(relative))

    @staticmethod
    def _labels_path_for(record: dict) -> Path | None:
        path = record.get("nucleus_tracked_labels_path")
        return Path(path) if path else None

    def _nucleus_labels_path(self) -> Path | None:
        if self._record is None:
            return None
        return self._labels_path_for(self._record)

    def _resolve_nls_path(self, record: dict | None) -> Path | None:
        """Resolve the NLS field against *record*: absolute as-is, else per-position.

        A relative entry (e.g. ``0_input/NLS_zavg.tif``) is joined onto the
        record's ``position_path`` so one field resolves across a batch.
        """
        text = self._nls_edit.text().strip()
        if not text or record is None:
            return None
        path = Path(text)
        if path.is_absolute():
            return path
        position = record.get("position_path")
        return Path(position) / path if position else path

    def _nls_path(self) -> Path | None:
        return self._resolve_nls_path(self._record)

    def _is_batch(self) -> bool:
        return len(self._records) > 1

    def _batch_records(self) -> list[dict]:
        """Selected positions that have every input needed to classify headlessly."""
        out: list[dict] = []
        for record in self._records:
            nls = self._resolve_nls_path(record)
            labels = self._labels_path_for(record)
            h5 = record.get("contact_analysis_path")
            if (
                nls is not None and nls.is_file()
                and labels is not None and labels.is_file()
                and h5 and Path(h5).is_file()
            ):
                out.append(record)
        return out

    def _update_enabled(self) -> None:
        """Drive the single action button's label + enabled-ness per state.

        Batch → *Classify & apply to all H5*. Single position: *Apply to H5* once a
        classification exists (the *Classified* state), else *Classify* (the *Needs
        classify* state), enabled only when the inputs resolve.
        """
        running = self._measure_worker is not None
        if self._is_batch():
            self._action_btn.setText("Classify & apply to all H5")
            self._action_btn.setEnabled(bool(self._batch_records()) and not running)
            self._threshold_spin.setEnabled(False)
            return
        if self._assignments:  # Classified
            self._action_btn.setText("Apply to H5")
            self._action_btn.setEnabled(not running)
        else:  # Needs classify
            labels_path = self._nucleus_labels_path()
            can_measure = (
                self._record is not None
                and labels_path is not None
                and labels_path.is_file()
                and self._nls_path() is not None
                and not running
            )
            self._action_btn.setText("Classify")
            self._action_btn.setEnabled(bool(can_measure))
        self._threshold_spin.setEnabled(self._intensities.size > 0)

    def _on_action(self) -> None:
        """Route the one button to measure / write / batch by current state."""
        if self._is_batch():
            self._on_apply_batch()
        elif self._assignments:
            self._on_apply()
        else:
            self._on_measure()

    def _on_nls_text_changed(self) -> None:
        # Changing the marker image invalidates any existing classification:
        # revert to *Needs classify* and hide the now-stale results pane.
        if self._assignments or self._measurements:
            self._reset_measurement()
        self._update_enabled()

    def _show_results_pane(self) -> None:
        """Reveal the results window docked under the controls, sized for the scatter."""
        if not self._results_pane.isHidden():
            return
        self._results_pane.setVisible(True)
        total = self._splitter.height() or 360
        # Scatter takes the bulk; a thin spacer below keeps the bottom handle
        # grabbable and leaves the scatter room to grow downward.
        spacer = max(total // 6, 48)
        self._splitter.setSizes([max(total - spacer, 1), spacer])

    # ------------------------------------------------------------------ measuring
    def _reset_measurement(self) -> None:
        self._measurements = {}
        self._track_ids = np.empty(0, dtype=int)
        self._intensities = np.empty(0, dtype=float)
        self._scatter_x = np.empty(0, dtype=float)
        self._nucleus_labels = None
        self._assignments = {}
        if _HAS_PYQTGRAPH:
            self._scatter.clear()
            self._threshold_line.setVisible(False)
        self._counts_lbl.setText("")
        # Back to *Needs classify*: hide the results pane and relabel the button.
        self._results_pane.setVisible(False)
        self._action_btn.setText("Classify")

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
        self._intensities = np.asarray(
            [measurements[int(t)].intensity for t in self._track_ids], dtype=float
        )
        rng = np.random.default_rng(0)
        self._scatter_x = rng.uniform(-0.35, 0.35, size=self._track_ids.size)

        intensity_map = {int(t): float(v) for t, v in zip(self._track_ids, self._intensities)}
        try:
            threshold = auto_threshold(intensity_map)
        except NLSClassificationError as exc:
            self._status_lbl.setText(f"Status: cannot auto-threshold: {exc}")
            threshold = float(np.median(self._intensities))

        self._ensure_image_layer()
        # First successful classification → reveal the results pane (scatter + counts).
        self._show_results_pane()
        if _HAS_PYQTGRAPH:
            lo, hi = float(self._intensities.min()), float(self._intensities.max())
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
        # Now in *Classified*: refresh the button label/enable through one place.
        self._update_enabled()

    def _paint_scatter(self) -> None:
        if not _HAS_PYQTGRAPH or self._track_ids.size == 0:
            return
        brushes = [
            pg.mkBrush(*(_POSITIVE_RGB if self._assignments.get(int(t)) == POSITIVE else _NEGATIVE_RGB))
            for t in self._track_ids
        ]
        self._scatter.setData(x=self._scatter_x, y=self._intensities, brush=brushes)

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
        if self._is_batch():
            self._on_apply_batch()
            return
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

    # --------------------------------------------------------------- batch apply
    def _on_apply_batch(self) -> None:
        """Classify + write every selected position headlessly, each auto-thresholded."""
        records = self._batch_records()
        if not records:
            self._status_lbl.setText(
                "Status: no selected position has an NLS image, nucleus labels, and an .h5."
            )
            return
        positive_label = self._positive_edit.text().strip() or "positive"
        negative_label = self._negative_edit.text().strip() or "negative"
        # (id, h5, nls, labels) snapshots — read off the worker thread.
        jobs = [
            (
                str(record.get("id", "?")),
                Path(record["contact_analysis_path"]),
                self._resolve_nls_path(record),
                self._labels_path_for(record),
            )
            for record in records
        ]
        self._status_lbl.setText(f"Status: classifying {len(jobs)} position(s)…")
        self._measure_worker = object()
        self._update_enabled()

        @thread_worker(
            connect={
                "yielded": self._on_batch_progress,
                "returned": self._on_batch_done,
                "errored": self._on_measure_error,
            }
        )
        def _worker():
            results: list[tuple[str, bool, str]] = []
            for index, (record_id, h5_path, nls_path, labels_path) in enumerate(jobs, start=1):
                yield index, len(jobs), record_id
                try:
                    summary = patch_position_contact_analysis_nls_classes(
                        h5_path,
                        nls_path,
                        labels_path,
                        positive_label=positive_label,
                        negative_label=negative_label,
                    )
                    detail = (
                        f"{summary.positive_track_count} positive / "
                        f"{summary.negative_track_count} negative"
                    )
                    results.append((record_id, True, detail))
                except Exception as exc:  # noqa: BLE001 - per-position failures are non-fatal
                    results.append((record_id, False, str(exc)))
            return results

        self._measure_worker = _worker()

    def _on_batch_progress(self, info: tuple[int, int, str]) -> None:
        index, total, record_id = info
        self._status_lbl.setText(f"Status: [{index}/{total}] classifying {record_id}…")

    def _on_batch_done(self, results: list[tuple[str, bool, str]]) -> None:
        self._measure_worker = None
        succeeded = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]
        message = f"Status: classified {len(succeeded)}/{len(results)} position(s)."
        if failed:
            record_id, _, detail = failed[0]
            extra = "" if len(failed) == 1 else f" (+{len(failed) - 1} more)"
            message += f" Failed: {record_id}: {detail}{extra}."
        self._status_lbl.setText(message)
        self._update_enabled()
