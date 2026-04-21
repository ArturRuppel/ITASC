"""
Tracking tab for napariSegTrack.

Input  : segmentation labels from the Project Panel state, or the Segmentation
         tab's active Labels layer as fallback.
Output : tracked Labels written back into the same shared layer.

Tracking is performed with LapTrack (centroid-distance LAP with gap closing).
"""

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout,
    QSpinBox, QDoubleSpinBox, QComboBox,
    QPushButton, QLabel, QToolButton,
    QProgressBar, QScrollArea,
)
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info, show_error as _show_error
import napari

from .registry import get_state

# Optional dependency check
try:
    import laptrack as _laptrack  # noqa: F401
    _LAPTRACK_AVAILABLE = True
except ImportError:
    _LAPTRACK_AVAILABLE = False


# ── helpers ────────────────────────────────────────────────────────────

def _sep(title):
    lbl = QLabel(f"<b>{title}</b>")
    lbl.setStyleSheet("color: palette(mid); margin-top: 4px;")
    return lbl


# ── defaults ───────────────────────────────────────────────────────────

_METRICS = ["euclidean", "sqeuclidean", "cityblock", "cosine"]

TRACK_DEFAULTS = {
    "metric":                      "euclidean",
    "gap_closing_metric":          "euclidean",
    "max_link_dist":               20,
    "max_gap_dist":                25,
    "gap_closing_max_frame_count": 3,
    # 0.0 = Auto (let LapTrack compute from cost percentile)
    "track_start_cost":            0.0,
    "track_end_cost":              0.0,
    "alternative_cost_factor":     1.05,
    "alternative_cost_percentile": 90,
}



# ── widget ─────────────────────────────────────────────────────────────

class TrackingTab(QWidget):
    """Tracking tab: LapTrack-based cell tracking from a Labels layer."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer        = viewer
        self._worker       = None
        self._tracks_layer = None      # dedicated napari Tracks layer
        self._state        = get_state(viewer)
        self._preset_layer: napari.layers.Labels = None  # set by parent

        self._setup_ui()

    # ── UI construction ────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── Tracking parameters ──
        track_toggle = QToolButton()
        track_toggle.setText("LapTrack Parameters")
        track_toggle.setArrowType(Qt.RightArrow)
        track_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        track_toggle.setCheckable(True)
        track_toggle.setChecked(False)
        track_toggle.setStyleSheet("QToolButton { font-weight: bold; }")
        root.addWidget(track_toggle)

        track_inner = QWidget()
        self._track_form = QFormLayout(track_inner)
        self._track_form.setSpacing(4)
        self._build_track_params()
        t_scroll = QScrollArea()
        t_scroll.setWidget(track_inner)
        t_scroll.setWidgetResizable(True)
        t_scroll.setFixedHeight(240)
        t_scroll.setVisible(False)
        root.addWidget(t_scroll)

        def _toggle_track(checked):
            track_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
            t_scroll.setVisible(checked)
        track_toggle.toggled.connect(_toggle_track)

        # ── IoU weight ──
        iou_row = QFormLayout()
        iou_row.setSpacing(4)
        self._p_iou_weight = QDoubleSpinBox()
        self._p_iou_weight.setRange(0.0, 1.0)
        self._p_iou_weight.setDecimals(2)
        self._p_iou_weight.setSingleStep(0.1)
        self._p_iou_weight.setValue(0.0)
        self._p_iou_weight.setToolTip(
            "Blend between centroid distance (0) and IoU-based cost (1).\n"
            "Higher values penalise linking cells with low mask overlap.\n"
            "Gap closing always uses plain centroid distance."
        )
        iou_row.addRow("IoU weight:", self._p_iou_weight)
        iou_widget = QWidget()
        iou_widget.setLayout(iou_row)
        root.addWidget(iou_widget)

        # ── Run button ──
        self._run_btn = QPushButton("Run Tracking")
        self._run_btn.clicked.connect(self._on_run)
        if not _LAPTRACK_AVAILABLE:
            self._run_btn.setEnabled(False)
            self._run_btn.setToolTip(
                "LapTrack is not installed.\n"
                "Install with: pip install cellflow-napari[correction]"
            )
        root.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        root.addWidget(self._progress)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        root.addWidget(self._cancel_btn)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("font-size: 9pt; color: palette(mid);")
        root.addWidget(self._status_label)

        root.addStretch()

        # attribution
        attrib = QLabel(
            'Tracking powered by '
            '<a href="https://github.com/yfukai/laptrack">LapTrack</a>.'
            '<br>If you use tracking, please cite:<br>'
            '<a href="https://doi.org/10.1093/bioinformatics/btac799">'
            'doi:10.1093/bioinformatics/btac799</a>'
        )
        attrib.setOpenExternalLinks(True)
        attrib.setWordWrap(True)
        attrib.setStyleSheet("color: palette(mid); font-size: 9pt;")
        root.addWidget(attrib)

    def _build_track_params(self):
        add = self._track_form.addRow

        self._p_metric = QComboBox()
        self._p_metric.addItems(_METRICS)
        self._p_metric.setCurrentText(TRACK_DEFAULTS["metric"])
        self._p_metric.setToolTip("Distance metric used for frame-to-frame linking.")
        add("Link metric:", self._p_metric)

        self._p_gap_metric = QComboBox()
        self._p_gap_metric.addItems(_METRICS)
        self._p_gap_metric.setCurrentText(TRACK_DEFAULTS["gap_closing_metric"])
        self._p_gap_metric.setToolTip("Distance metric used for gap closing.")
        add("Gap metric:", self._p_gap_metric)

        self._p_link = QSpinBox()
        self._p_link.setRange(1, 500)
        self._p_link.setValue(TRACK_DEFAULTS["max_link_dist"])
        add("Max link dist (px):", self._p_link)

        self._p_gap = QSpinBox()
        self._p_gap.setRange(1, 500)
        self._p_gap.setValue(TRACK_DEFAULTS["max_gap_dist"])
        add("Max gap dist (px):", self._p_gap)

        self._p_gapf = QSpinBox()
        self._p_gapf.setRange(1, 20)
        self._p_gapf.setValue(TRACK_DEFAULTS["gap_closing_max_frame_count"])
        add("Gap closing frames:", self._p_gapf)

        self._p_start_cost = QDoubleSpinBox()
        self._p_start_cost.setRange(0.0, 100000.0)
        self._p_start_cost.setDecimals(1)
        self._p_start_cost.setSingleStep(10.0)
        self._p_start_cost.setSpecialValueText("Auto")
        self._p_start_cost.setValue(TRACK_DEFAULTS["track_start_cost"])
        self._p_start_cost.setToolTip(
            "Cost for starting a new track (no prior frame link).\n"
            "Higher = fewer spurious new tracks. Auto = derived from cost percentile."
        )
        add("Track start cost:", self._p_start_cost)

        self._p_end_cost = QDoubleSpinBox()
        self._p_end_cost.setRange(0.0, 100000.0)
        self._p_end_cost.setDecimals(1)
        self._p_end_cost.setSingleStep(10.0)
        self._p_end_cost.setSpecialValueText("Auto")
        self._p_end_cost.setValue(TRACK_DEFAULTS["track_end_cost"])
        self._p_end_cost.setToolTip(
            "Cost for ending a track (no subsequent frame link).\n"
            "Higher = fewer tracks that disappear prematurely. Auto = derived from cost percentile."
        )
        add("Track end cost:", self._p_end_cost)

        self._p_alt_factor = QDoubleSpinBox()
        self._p_alt_factor.setRange(1.0, 10.0)
        self._p_alt_factor.setDecimals(2)
        self._p_alt_factor.setSingleStep(0.05)
        self._p_alt_factor.setValue(TRACK_DEFAULTS["alternative_cost_factor"])
        self._p_alt_factor.setToolTip(
            "Multiplier on the cost percentile used to auto-compute start/end costs.\n"
            "Increase to make new-track / track-end events more expensive."
        )
        add("Alt cost factor:", self._p_alt_factor)

        self._p_alt_pct = QSpinBox()
        self._p_alt_pct.setRange(50, 100)
        self._p_alt_pct.setValue(TRACK_DEFAULTS["alternative_cost_percentile"])
        self._p_alt_pct.setToolTip(
            "Percentile of linking costs used to set the auto start/end cost baseline.\n"
            "Higher = auto costs scale with more expensive links in the dataset."
        )
        add("Alt cost percentile:", self._p_alt_pct)

    # ── Parameter collection ───────────────────────────────────────────

    def get_params(self) -> dict:
        return self._collect_track_params()

    def set_params(self, data: dict) -> None:
        if "metric" in data:
            idx = self._p_metric.findText(data["metric"])
            if idx >= 0:
                self._p_metric.setCurrentIndex(idx)
        if "gap_closing_metric" in data:
            idx = self._p_gap_metric.findText(data["gap_closing_metric"])
            if idx >= 0:
                self._p_gap_metric.setCurrentIndex(idx)
        if "max_link_dist" in data:
            self._p_link.setValue(int(data["max_link_dist"]))
        if "max_gap_dist" in data:
            self._p_gap.setValue(int(data["max_gap_dist"]))
        if "gap_closing_max_frame_count" in data:
            self._p_gapf.setValue(int(data["gap_closing_max_frame_count"]))
        if "track_start_cost" in data:
            self._p_start_cost.setValue(data["track_start_cost"] or 0.0)
        if "track_end_cost" in data:
            self._p_end_cost.setValue(data["track_end_cost"] or 0.0)
        if "alternative_cost_factor" in data:
            self._p_alt_factor.setValue(float(data["alternative_cost_factor"]))
        if "alternative_cost_percentile" in data:
            self._p_alt_pct.setValue(int(data["alternative_cost_percentile"]))
        if "iou_weight" in data:
            self._p_iou_weight.setValue(float(data["iou_weight"]))

    def _collect_track_params(self):
        start = self._p_start_cost.value()
        end   = self._p_end_cost.value()
        return {
            "metric":                      self._p_metric.currentText(),
            "gap_closing_metric":          self._p_gap_metric.currentText(),
            "max_link_dist":               self._p_link.value(),
            "max_gap_dist":                self._p_gap.value(),
            "gap_closing_max_frame_count": self._p_gapf.value(),
            "track_start_cost":            None if start == 0.0 else start,
            "track_end_cost":              None if end   == 0.0 else end,
            "alternative_cost_factor":     self._p_alt_factor.value(),
            "alternative_cost_percentile": self._p_alt_pct.value(),
            "iou_weight":                  self._p_iou_weight.value(),
        }

    # ── Run ────────────────────────────────────────────────────────────

    def _on_run(self):
        if not _LAPTRACK_AVAILABLE:
            self._log_append(
                "ERROR: LapTrack is not installed. "
                "Run: pip install cellflow-napari[correction]"
            )
            return
        if self._worker is not None and self._worker.is_running:
            self._log_append("Tracking already running.")
            return

        if self._preset_layer is None or self._preset_layer.name not in self.viewer.layers:
            self._log_append("ERROR: Load a layer first.")
            return

        # Take an explicit copy so the worker holds a snapshot and is isolated
        # from any further layer writes while it runs.
        nuc_data = np.array(self._preset_layer.data, dtype=np.int32)

        track_params = self._collect_track_params()

        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._cancel_btn.setVisible(True)

        @thread_worker(connect={
            "yielded":  lambda m: self._log_append(m),
            "returned": self._on_done,
            "errored":  self._on_error,
        })
        def _work():
            import queue
            from cellflow.backend.segmentation import track_nuclei_laptrack

            if nuc_data.ndim == 2:
                nuc_frames = [nuc_data]
            else:
                nuc_frames = [nuc_data[t] for t in range(nuc_data.shape[0])]

            # Progress messages from the backend are queued so we can yield
            # them here — yielding is the only point where cancel is checked.
            msg_queue = queue.SimpleQueue()
            if track_params["iou_weight"] > 0.0:
                yield "Precomputing IoU costs…"
            else:
                yield "Running LapTrack…"

            tracked_nuc, track_df = track_nuclei_laptrack(
                nuc_frames,
                metric                      = track_params["metric"],
                gap_closing_metric          = track_params["gap_closing_metric"],
                max_link_dist               = track_params["max_link_dist"],
                max_gap_dist                = track_params["max_gap_dist"],
                gap_closing_max_frame_count = track_params["gap_closing_max_frame_count"],
                track_start_cost            = track_params["track_start_cost"],
                track_end_cost              = track_params["track_end_cost"],
                alternative_cost_factor     = track_params["alternative_cost_factor"],
                alternative_cost_percentile = track_params["alternative_cost_percentile"],
                iou_weight                  = track_params["iou_weight"],
                progress_cb                 = msg_queue.put,
            )
            while not msg_queue.empty():
                yield msg_queue.get_nowait()
            n_tracks = track_df["track_id"].nunique() if len(track_df) > 0 else 0
            yield f"  {n_tracks} track(s) found"
            yield "Tracking complete!"
            return tracked_nuc

        self._worker = _work()
        self._worker.aborted.connect(self._on_aborted)

    def _on_done(self, tracked_nuc):
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._cancel_btn.setVisible(False)

        stacked = np.stack(tracked_nuc, axis=0).astype(np.uint16)

        seg_layer = self._preset_layer
        if seg_layer is not None and seg_layer.name in self.viewer.layers:
            seg_layer.data = stacked
            seg_layer.refresh()
            self._log_append("Done! Layer updated with tracked result.")
        else:
            seg_layer = self.viewer.add_labels(stacked, name="Nuclear Labels")
            self._log_append("Done! Tracked labels added as Nuclear Labels layer.")
            self._preset_layer = seg_layer

        self._state.set_tissue_nuclear_labels(np.asarray(seg_layer.data), seg_layer.name)
        self._rebuild_tracks_layer(stacked)

    # ── Tracks layer ───────────────────────────────────────────────────

    def rebuild_tracks_layer(self):
        """Public: rebuild the Tracks layer from the current segmentation.

        Call this after external label edits (e.g. corrections) to keep the
        Tracks layer in sync with the labels.
        """
        layer = self._preset_layer
        if layer is None or layer.name not in self.viewer.layers:
            return
        stacked = np.asarray(layer.data)
        if stacked.ndim < 3:
            return
        self._rebuild_tracks_layer(stacked)

    def set_data_layer(self, layer: "napari.layers.Labels") -> None:
        """Set the labels layer to track. Called by the parent widget after loading."""
        self._preset_layer = layer

    def _rebuild_tracks_layer(self, stacked):
        """Compute centroids per frame and update (or create) the Tracks layer."""
        from skimage.measure import regionprops_table

        if stacked.ndim < 3:
            return

        rows = []
        for t in range(stacked.shape[0]):
            frame = stacked[t]
            if frame.max() == 0:
                continue
            props = regionprops_table(frame, properties=["label", "centroid"])
            for lbl, y, x in zip(props["label"], props["centroid-0"], props["centroid-1"]):
                rows.append([int(lbl), t, y, x])

        if not rows:
            return

        track_data = np.array(rows, dtype=float)  # (N, 4): [track_id, t, y, x]
        # napari requires rows sorted by track_id then t
        idx = np.lexsort((track_data[:, 1], track_data[:, 0]))
        track_data = track_data[idx]

        if self._tracks_layer is not None and self._tracks_layer in self.viewer.layers:
            self._tracks_layer.data = track_data
        else:
            self._tracks_layer = self.viewer.add_tracks(track_data, name="Tracks")

    def _on_error(self, exc):
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._cancel_btn.setVisible(False)
        self._log_append(f"ERROR: {exc}")

    def _on_cancel(self):
        if self._worker is not None:
            self._worker.quit()

    def _on_aborted(self):
        self._run_btn.setEnabled(True)
        self._progress.setVisible(False)
        self._cancel_btn.setVisible(False)
        self._log_append("Cancelled.")

    def _log_append(self, msg: str) -> None:
        msg = str(msg)
        self._status_label.setText(msg)
        if msg.startswith("ERROR"):
            _show_error(msg)
        elif msg == "Tracking complete!":
            show_info(msg)
