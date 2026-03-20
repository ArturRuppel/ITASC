"""Nuclear Tracks widget — converts TrackMate XML + Voronoi into Labels layers.

This is a preprocessing widget: it takes nuclear tracking data (TrackMate XML),
computes a Voronoi tessellation, rasterizes it into a Labels layer, and assigns
track IDs so the main TissueGraphWidget can consume the result.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QProgressBar,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QFileDialog,
    QLineEdit,
)
from qtpy.QtCore import QThread, Qt, Signal, QObject

from ..structures import VoronoiMethod
from ..core.trackmate import TrackMateData, parse_trackmate_xml
from ..core.voronoi import voronoi_to_labels

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Worker
# ------------------------------------------------------------------

class VoronoiToLabelsWorker(QObject):
    """Build a 4-D label stack from TrackMate positions via Voronoi tessellation."""

    progress = Signal(int, str)
    finished = Signal(object)  # emits (label_stack, track_map)
    error = Signal(Exception)

    def __init__(self, trackmate_data: TrackMateData,
                 method: VoronoiMethod = VoronoiMethod.STANDARD,
                 lloyd_iterations: int = 10, lloyd_tol: float = 0.1,
                 image_shape: Optional[tuple] = None):
        super().__init__()
        self.trackmate_data = trackmate_data
        self.method = method
        self.lloyd_iterations = lloyd_iterations
        self.lloyd_tol = lloyd_tol
        self.image_shape = image_shape

    def run(self):
        try:
            td = self.trackmate_data
            image_shape = self.image_shape or td.image_shape
            if image_shape is None:
                raise ValueError(
                    "Image dimensions unknown — the TrackMate XML does not "
                    "contain image size metadata. Please set dimensions manually."
                )

            frames = sorted(td.spots_by_frame.keys())
            n_frames = len(frames)
            H, W = image_shape

            label_stack = np.zeros((n_frames, H, W), dtype=np.int32)
            # track_map: frame_index -> {label_value -> track_id}
            track_map = {}

            for i, frame in enumerate(frames):
                self.progress.emit(
                    int((i / n_frames) * 90),
                    f"Voronoi tessellation frame {i + 1}/{n_frames}...",
                )

                spots = td.spots_by_frame[frame]
                if len(spots) == 0:
                    continue

                spot_ids = [s[0] for s in spots]
                positions = np.array([[s[1], s[2]] for s in spots])  # (y, x)

                labels, _final_pos = voronoi_to_labels(
                    positions, image_shape,
                    method=self.method,
                    lloyd_iterations=self.lloyd_iterations,
                    lloyd_tol=self.lloyd_tol,
                )
                label_stack[i] = labels

                # Build track map for this frame: label (1-indexed) -> track_id
                frame_tracks = {}
                for cell_idx, spot_id in enumerate(spot_ids):
                    label_val = cell_idx + 1  # voronoi_to_labels is 1-indexed
                    track_id = td.spot_to_track.get(spot_id)
                    if track_id is not None:
                        frame_tracks[label_val] = track_id
                track_map[i] = frame_tracks

            self.progress.emit(100, "Voronoi tessellation complete.")
            self.finished.emit((label_stack, track_map))
        except Exception as e:
            self.error.emit(e)


# ------------------------------------------------------------------
# Widget
# ------------------------------------------------------------------

class NuclearTracksWidget(QWidget):
    """Load TrackMate XML, compute Voronoi, produce a Labels layer."""

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._trackmate_data: Optional[TrackMateData] = None
        self._thread: Optional[QThread] = None
        self._worker: Optional[VoronoiToLabelsWorker] = None

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # --- Load TrackMate XML ---
        load_group = QGroupBox("TrackMate XML")
        load_layout = QVBoxLayout()

        self.load_xml_btn = QPushButton("Load TrackMate XML...")
        load_layout.addWidget(self.load_xml_btn)

        self.xml_info_label = QLabel("No file loaded")
        self.xml_info_label.setWordWrap(True)
        self.xml_info_label.setStyleSheet("color: gray;")
        load_layout.addWidget(self.xml_info_label)

        load_group.setLayout(load_layout)
        layout.addWidget(load_group)

        # --- Image dimensions override ---
        dims_group = QGroupBox("Image Dimensions")
        dims_layout = QVBoxLayout()

        dims_hint = QLabel("Auto-detected from XML if available.")
        dims_hint.setStyleSheet("color: gray; font-size: 11px;")
        dims_layout.addWidget(dims_hint)

        h_row = QHBoxLayout()
        h_row.addWidget(QLabel("Height (px):"))
        self.height_spin = QSpinBox()
        self.height_spin.setMinimum(1)
        self.height_spin.setMaximum(100000)
        self.height_spin.setValue(512)
        h_row.addWidget(self.height_spin)
        dims_layout.addLayout(h_row)

        w_row = QHBoxLayout()
        w_row.addWidget(QLabel("Width (px):"))
        self.width_spin = QSpinBox()
        self.width_spin.setMinimum(1)
        self.width_spin.setMaximum(100000)
        self.width_spin.setValue(512)
        w_row.addWidget(self.width_spin)
        dims_layout.addLayout(w_row)

        dims_group.setLayout(dims_layout)
        layout.addWidget(dims_group)

        # --- Voronoi parameters ---
        voronoi_group = QGroupBox("Voronoi Tessellation")
        vor_layout = QVBoxLayout()

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Standard", "Lloyd's relaxation"])
        method_row.addWidget(self.method_combo)
        vor_layout.addLayout(method_row)

        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Lloyd iterations:"))
        self.lloyd_iter_spin = QSpinBox()
        self.lloyd_iter_spin.setMinimum(0)
        self.lloyd_iter_spin.setMaximum(200)
        self.lloyd_iter_spin.setValue(10)
        iter_row.addWidget(self.lloyd_iter_spin)
        vor_layout.addLayout(iter_row)

        tol_row = QHBoxLayout()
        tol_row.addWidget(QLabel("Lloyd tolerance:"))
        self.lloyd_tol_spin = QDoubleSpinBox()
        self.lloyd_tol_spin.setMinimum(0.0)
        self.lloyd_tol_spin.setMaximum(100.0)
        self.lloyd_tol_spin.setSingleStep(0.1)
        self.lloyd_tol_spin.setValue(0.1)
        tol_row.addWidget(self.lloyd_tol_spin)
        vor_layout.addLayout(tol_row)

        voronoi_group.setLayout(vor_layout)
        layout.addWidget(voronoi_group)

        self._update_lloyd_visibility()

        # --- Generate button ---
        self.generate_btn = QPushButton("Generate Labels")
        self.generate_btn.setStyleSheet(
            "QPushButton { font-weight: bold; padding: 6px; }"
        )
        self.generate_btn.setEnabled(False)
        layout.addWidget(self.generate_btn)

        # --- Progress ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.load_xml_btn.clicked.connect(self._on_load_xml)
        self.generate_btn.clicked.connect(self._on_generate)
        self.method_combo.currentIndexChanged.connect(self._update_lloyd_visibility)

    def _update_lloyd_visibility(self):
        is_lloyd = self.method_combo.currentIndex() == 1
        self.lloyd_iter_spin.setEnabled(is_lloyd)
        self.lloyd_tol_spin.setEnabled(is_lloyd)

    # ------------------------------------------------------------------
    # Load TrackMate XML
    # ------------------------------------------------------------------

    def _on_load_xml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load TrackMate XML", "", "XML files (*.xml);;All files (*)"
        )
        if not path:
            return

        try:
            self._trackmate_data = parse_trackmate_xml(path)
            td = self._trackmate_data
            filename = Path(path).name

            # Update image dimension spinboxes if XML has them
            if td.image_shape is not None:
                self.height_spin.setValue(td.image_shape[0])
                self.width_spin.setValue(td.image_shape[1])

            self.xml_info_label.setText(
                f"<b>{filename}</b><br>"
                f"{td.n_spots} spots, {td.n_tracks} tracks, "
                f"{len(td.spots_by_frame)} frames"
                + (f"<br>Image: {td.image_shape[1]}×{td.image_shape[0]} px"
                   if td.image_shape else "")
            )
            self.xml_info_label.setStyleSheet("")

            self.generate_btn.setEnabled(True)
            self.status_label.setText("")
        except Exception as e:
            self.xml_info_label.setText(f"Error: {e}")
            self.xml_info_label.setStyleSheet("color: red;")
            self._trackmate_data = None
            self.generate_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Generate Labels
    # ------------------------------------------------------------------

    def _on_generate(self):
        if self._trackmate_data is None:
            return

        image_shape = (self.height_spin.value(), self.width_spin.value())
        method = (VoronoiMethod.LLOYD if self.method_combo.currentIndex() == 1
                  else VoronoiMethod.STANDARD)

        worker = VoronoiToLabelsWorker(
            self._trackmate_data,
            method=method,
            lloyd_iterations=self.lloyd_iter_spin.value(),
            lloyd_tol=self.lloyd_tol_spin.value(),
            image_shape=image_shape,
        )
        self._run_worker(worker, self._on_generate_finished)

    def _on_generate_finished(self, result):
        label_stack, track_map = result
        self._finish_worker()

        # Add Labels layer to viewer
        layer_name = "Voronoi Labels"
        # Remove existing layer with the same name
        existing = [l for l in self.viewer.layers if l.name == layer_name]
        for l in existing:
            self.viewer.layers.remove(l)

        layer = self.viewer.add_labels(label_stack, name=layer_name)

        # Store track_map as layer metadata so the main widget can use it
        layer.metadata["track_map"] = track_map
        layer.metadata["source"] = "nuclear_tracks"

        n_frames = label_stack.shape[0]
        n_cells_per_frame = [len(np.unique(label_stack[i])) - (1 if 0 in label_stack[i] else 0)
                             for i in range(n_frames)]
        avg_cells = np.mean(n_cells_per_frame) if n_cells_per_frame else 0

        self.status_label.setText(
            f"Labels added to viewer: {n_frames} frames, "
            f"~{avg_cells:.0f} cells/frame. "
            f"Use TissueGraphWidget to run the pipeline."
        )

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    def _run_worker(self, worker, on_finished):
        self._thread = QThread()
        self._worker = worker
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.error.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self.generate_btn.setEnabled(False)
        self.load_xml_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Working...")

        self._thread.start()

    def _finish_worker(self):
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.load_xml_btn.setEnabled(True)

    def _on_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def _on_error(self, exc):
        self._finish_worker()
        self.status_label.setText(f"Error: {exc}")
        logger.exception("VoronoiToLabels worker failed", exc_info=exc)
