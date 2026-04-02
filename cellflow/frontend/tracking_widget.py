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
    QSpinBox,
    QPushButton, QLabel, QToolButton,
    QTextEdit, QProgressBar, QScrollArea,
)
from napari.qt.threading import thread_worker
import napari

from .registry import get_state


# ── helpers ────────────────────────────────────────────────────────────

def _sep(title):
    lbl = QLabel(f"<b>{title}</b>")
    lbl.setStyleSheet("color: palette(text); margin-top: 4px;")
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
        self.viewer        = viewer
        self._seg_tab      = seg_tab   # SegmentationTab — single source of truth for labels data
        self._worker       = None
        self._tracks_layer = None      # dedicated napari Tracks layer
        self._state        = get_state(viewer)

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
        t_scroll.setFixedHeight(160)
        t_scroll.setVisible(False)
        root.addWidget(t_scroll)

        def _toggle_track(checked):
            track_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
            t_scroll.setVisible(checked)
        track_toggle.toggled.connect(_toggle_track)

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
        attrib.setStyleSheet("color: palette(text); font-size: 9pt;")
        root.addWidget(attrib)

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

        # Take an explicit copy so the worker holds a snapshot and is isolated
        # from any further layer writes while it runs.
        if self._state.tissue.labels is None:
            self._log_append("ERROR: Load a tissue with segmentation in the Project Panel first.")
            return
        nuc_data = np.array(self._state.tissue.labels, dtype=np.int32)

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
            from cellflow.backend.segmentation import track_nuclei_laptrack

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
            scale = self._seg_tab._input_scale
            kw = {"scale": scale[-stacked.ndim:]} if scale is not None else {}
            new_layer = self.viewer.add_labels(stacked, name="Segmentation", **kw)
            self._seg_tab._seg_layer = new_layer
            seg_layer = new_layer
            self._log_append("Done! Tracked labels added as Segmentation layer.")

        self._state.set_tissue_labels(np.asarray(seg_layer.data), seg_layer.name)
        self._seg_tab._seg_status.setText(f"Tracked: {seg_layer.data.shape}")
        self._rebuild_tracks_layer(stacked)

    # ── Tracks layer ───────────────────────────────────────────────────

    def rebuild_tracks_layer(self):
        """Public: rebuild the Tracks layer from the current segmentation.

        Call this after external label edits (e.g. corrections) to keep the
        Tracks layer in sync with the labels.
        """
        seg_layer = self._seg_tab._seg_layer
        if seg_layer is None or seg_layer not in self.viewer.layers:
            return
        stacked = seg_layer.data
        if stacked.ndim < 3:
            return
        self._rebuild_tracks_layer(stacked)

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

    def _log_append(self, msg):
        self._log.append(str(msg))
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())
