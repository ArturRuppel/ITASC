"""
Tracking tab for napariSegTrack.

Input  : the Labels layer managed by the Segmentation tab (shared data manager).
         Load / Clear here and in the Segmentation tab are identical operations —
         there is exactly one segmentation layer at all times.
Output : tracked Labels written back into the same shared layer so both tabs
         always reflect the latest state.

Tracking is performed with LapTrack (centroid-distance LAP with gap closing).
"""

import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QSpinBox,
    QPushButton, QLabel,
    QTextEdit, QProgressBar, QScrollArea,
)
from napari.qt.threading import thread_worker
import napari


# ── helpers ────────────────────────────────────────────────────────────

def _sep(title):
    lbl = QLabel(f"<b>{title}</b>")
    lbl.setStyleSheet("color: palette(mid); margin-top: 4px;")
    return lbl


# ── defaults ───────────────────────────────────────────────────────────

TRACK_DEFAULTS = {
    "max_link_dist":               20,
    "max_gap_dist":                25,
    "gap_closing_max_frame_count": 3,
}


# ── widget ─────────────────────────────────────────────────────────────

class TrackingTab(QWidget):
    """Tracking tab: LapTrack-based cell tracking from a Labels layer."""

    def __init__(self, viewer: napari.Viewer, seg_tab):
        super().__init__()
        self.viewer   = viewer
        self._seg_tab = seg_tab   # SegmentationTab — single source of truth for labels data
        self._worker  = None

        self._setup_ui()

    # ── UI construction ────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # ── Nuclear input (mirrors seg tab's data manager) ──
        root.addWidget(self._build_nuc_input_panel())

        # ── Tracking parameters ──
        track_box = QGroupBox("LapTrack Parameters")
        track_box.setCheckable(True)
        track_box.setChecked(False)
        track_inner = QWidget()
        self._track_form = QFormLayout(track_inner)
        self._track_form.setSpacing(4)
        self._build_track_params()
        t_scroll = QScrollArea()
        t_scroll.setWidget(track_inner)
        t_scroll.setWidgetResizable(True)
        t_scroll.setFixedHeight(160)
        t_vlay = QVBoxLayout(track_box)
        t_vlay.addWidget(t_scroll)
        track_box.toggled.connect(t_scroll.setVisible)
        t_scroll.setVisible(False)
        root.addWidget(track_box)

        # ── Run button ──
        self._run_btn = QPushButton("Run Tracking")
        self._run_btn.clicked.connect(self._on_run)
        root.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        root.addWidget(self._progress)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._on_cancel)
        root.addWidget(self._cancel_btn)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(140)
        self._log.setPlaceholderText("Tracking log…")
        root.addWidget(self._log)

        root.addStretch()

    def _build_nuc_input_panel(self):
        box = QGroupBox("Nuclear Input")
        lay = QVBoxLayout(box)

        w = QWidget()
        hlay = QHBoxLayout(w)
        hlay.setContentsMargins(0, 0, 0, 0)

        self._load_nuc_btn = QPushButton("Load Nuclear Labels")
        self._load_nuc_btn.setFixedWidth(150)
        self._load_nuc_btn.setFixedHeight(25)
        self._load_nuc_btn.setToolTip(
            "Select a Labels layer in the napari layer list, then click this button.\n"
            "This is the same data as the Segmentation tab's 'Load Segmentation Layer'."
        )
        self._load_nuc_btn.clicked.connect(self._on_load_nuc)

        # Own label — synced from seg_tab rather than stealing its widget
        self._nuc_status = QLabel("Not loaded")
        self._nuc_status.setWordWrap(True)

        self._clear_nuc_btn = QPushButton("Clear")
        self._clear_nuc_btn.setFixedWidth(50)
        self._clear_nuc_btn.setFixedHeight(25)
        self._clear_nuc_btn.clicked.connect(self._on_clear_nuc)

        hlay.addWidget(self._load_nuc_btn)
        hlay.addWidget(self._nuc_status)
        hlay.addWidget(self._clear_nuc_btn)

        lay.addWidget(w)
        return box

    def showEvent(self, event):
        """Sync status whenever the tracking tab becomes visible."""
        super().showEvent(event)
        self._sync_status()

    def _sync_status(self):
        self._nuc_status.setText(self._seg_tab._seg_status.text())

    def _build_track_params(self):
        add = self._track_form.addRow

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

    # ── Data manager (delegates to seg tab — single source of truth) ───

    def _on_load_nuc(self):
        """Load active Labels layer as segmentation data (same as seg tab's Load)."""
        self._seg_tab._on_load_seg_layer()
        self._sync_status()

    def _on_clear_nuc(self):
        """Clear segmentation data (same as seg tab's Clear)."""
        self._seg_tab._on_clear_seg()
        self._sync_status()

    # ── Parameter collection ───────────────────────────────────────────

    def _collect_track_params(self):
        return {
            "max_link_dist":               self._p_link.value(),
            "max_gap_dist":                self._p_gap.value(),
            "gap_closing_max_frame_count": self._p_gapf.value(),
        }

    # ── Run ────────────────────────────────────────────────────────────

    def _on_run(self):
        if self._worker is not None and self._worker.is_running:
            self._log_append("Tracking already running.")
            return

        # Read directly from the live napari layer — the single source of truth.
        # Take an explicit copy so the worker holds a snapshot and is isolated
        # from any further layer writes while it runs.
        seg_layer = self._seg_tab._seg_layer
        if seg_layer is None or seg_layer not in self.viewer.layers:
            self._log_append("ERROR: Load a Nuclear Labels layer first.")
            return
        nuc_data = np.array(seg_layer.data, dtype=np.int32)  # explicit copy

        track_params = self._collect_track_params()

        self._log.clear()
        self._run_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._cancel_btn.setVisible(True)

        @thread_worker(connect={
            "yielded":  lambda m: self._log_append(m),
            "returned": self._on_done,
            "errored":  self._on_error,
        })
        def _work():
            from napariTissueFlow.segtrack._pipeline import track_nuclei_laptrack

            if nuc_data.ndim == 2:
                nuc_frames = [nuc_data]
            else:
                nuc_frames = [nuc_data[t] for t in range(nuc_data.shape[0])]

            yield "Running LapTrack…"
            tracked_nuc, track_df = track_nuclei_laptrack(
                nuc_frames,
                max_link_dist              = track_params["max_link_dist"],
                max_gap_dist               = track_params["max_gap_dist"],
                gap_closing_max_frame_count= track_params["gap_closing_max_frame_count"],
            )
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

        seg_layer = self._seg_tab._seg_layer
        if seg_layer is not None and seg_layer in self.viewer.layers:
            seg_layer.data = stacked
            seg_layer.refresh()
            self._log_append("Done! Segmentation layer updated with tracked result.")
        else:
            new_layer = self.viewer.add_labels(stacked, name="Segmentation")
            self._seg_tab._seg_layer = new_layer
            seg_layer = new_layer
            self._log_append("Done! Tracked labels added as Segmentation layer.")

        self._seg_tab._seg_status.setText(f"Loaded: {seg_layer.data.shape}")
        self._sync_status()

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

    def _log_append(self, msg):
        self._log.append(str(msg))
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())
