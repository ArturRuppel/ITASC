"""Minimal napari dock widget for napariTissueGraph."""
import logging
import numpy as np
from qtpy.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QLabel,
    QProgressBar,
)
from qtpy.QtCore import Signal, QThread, QObject

from ..structures import TissueGraphTimeSeries, InputType
from ..core.graph import build_from_labels, build_from_tracks
from .visualization import build_all_junction_lines, build_all_centroids

logger = logging.getLogger(__name__)


class GraphBuildWorker(QObject):
    """Worker to build tissue graph in a background thread."""
    progress = Signal(int, str)
    finished = Signal(object)
    error = Signal(Exception)

    def __init__(self, input_type: str, data: np.ndarray):
        super().__init__()
        self.input_type = input_type
        self.data = data

    def run(self):
        try:
            if self.input_type == "Segmentation Labels":
                series = self._build_labels_with_progress(self.data)
            else:
                series = build_from_tracks(self.data)
            self.finished.emit(series)
        except Exception as e:
            self.error.emit(e)

    def _build_labels_with_progress(self, label_stack):
        from ..core.labels import labels_to_graph
        from ..structures import TissueGraphFrame, TissueGraphTimeSeries, InputType

        n_frames = len(label_stack)
        frames_dict = {}
        for i in range(n_frames):
            self.progress.emit(
                int(100 * i / n_frames),
                f"Processing frame {i + 1}/{n_frames}",
            )
            cells, junctions, graph = labels_to_graph(label_stack[i])
            frames_dict[i] = TissueGraphFrame(
                frame=i,
                graph=graph,
                cells=cells,
                junctions=junctions,
                input_type=InputType.SEGMENTATION,
            )

        self.progress.emit(100, "Done")
        return TissueGraphTimeSeries(
            frames=frames_dict,
            input_type=InputType.SEGMENTATION,
        )


class TissueGraphWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.series: TissueGraphTimeSeries = None
        self._junction_layer = None
        self._centroid_layer = None
        self._thread = None
        self._worker = None

        self._build_ui()
        self._connect_signals()

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Input type selection
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Input type:"))
        self.input_type_combo = QComboBox()
        self.input_type_combo.addItems(["Segmentation Labels", "Nuclear Tracks"])
        type_row.addWidget(self.input_type_combo)
        layout.addLayout(type_row)

        # Layer selection
        layer_row = QHBoxLayout()
        layer_row.addWidget(QLabel("Layer:"))
        self.layer_combo = QComboBox()
        layer_row.addWidget(self.layer_combo)
        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setFixedWidth(30)
        layer_row.addWidget(self.refresh_btn)
        layout.addLayout(layer_row)

        # Build button
        self.build_btn = QPushButton("Build Graph")
        layout.addWidget(self.build_btn)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()

    def _connect_signals(self):
        self.refresh_btn.clicked.connect(self._refresh_layers)
        self.build_btn.clicked.connect(self._build_graph)
        self.input_type_combo.currentIndexChanged.connect(self._refresh_layers)
        self._refresh_layers()

    def _refresh_layers(self):
        self.layer_combo.clear()
        import napari
        input_type = self.input_type_combo.currentText()
        for layer in self.viewer.layers:
            if input_type == "Segmentation Labels" and isinstance(layer, napari.layers.Labels):
                self.layer_combo.addItem(layer.name)
            elif input_type == "Nuclear Tracks" and isinstance(layer, napari.layers.Points):
                self.layer_combo.addItem(layer.name)

    def _build_graph(self):
        layer_name = self.layer_combo.currentText()
        if not layer_name:
            self.status_label.setText("No layer selected.")
            return

        layer = self.viewer.layers[layer_name]
        input_type = self.input_type_combo.currentText()

        data = layer.data
        if input_type == "Segmentation Labels" and data.ndim == 2:
            data = data[np.newaxis, ...]

        # Set up worker and thread
        self._thread = QThread()
        self._worker = GraphBuildWorker(input_type, data)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_build_finished)
        self._worker.error.connect(self._on_build_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.error.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self.build_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Building graph...")

        self._thread.start()

    def _on_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def _on_build_finished(self, series):
        self.series = series
        self.build_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        n_frames = series.num_frames
        total_cells = sum(len(f.cells) for f in series.frames.values())
        total_junctions = sum(len(f.junctions) for f in series.frames.values())
        self.status_label.setText(
            f"Built graph: {n_frames} frames, "
            f"{total_cells} cells, {total_junctions} junctions total"
        )

        self._add_layers()

    def _on_build_error(self, exc):
        self.build_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error: {exc}")
        logger.exception("Graph build failed", exc_info=exc)

    def _add_layers(self):
        """Add junction and centroid layers with frame dimensions.

        Data is pre-built for all frames so napari's native dim slider
        handles frame scrubbing — no per-frame recomputation needed.
        """
        # Remove old layers
        for layer in (self._junction_layer, self._centroid_layer):
            if layer is not None and layer in self.viewer.layers:
                self.viewer.layers.remove(layer)

        # Junctions as shapes with (frame, y, x) coordinates
        lines, colors = build_all_junction_lines(self.series)
        if lines:
            self._junction_layer = self.viewer.add_shapes(
                lines,
                shape_type="path",
                edge_color=colors,
                edge_width=2,
                name="Junctions",
            )

        # Centroids as points with (frame, y, x) coordinates
        centroids = build_all_centroids(self.series)
        if len(centroids) > 0:
            self._centroid_layer = self.viewer.add_points(
                centroids,
                size=5,
                face_color="yellow",
                name="Cell Centroids",
            )

    def cleanup(self):
        """Clean up background thread if running."""
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait()
