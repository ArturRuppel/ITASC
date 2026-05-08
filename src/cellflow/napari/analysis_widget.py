"""Analysis widget for final processing and export in CellFlow v2."""
from __future__ import annotations

from pathlib import Path

from napari.qt.threading import thread_worker
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QCheckBox, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from cellflow.analysis import build_position_analysis_artifact
from cellflow.napari.ui_style import action_button, status_label

try:  # pragma: no cover - local branch compatibility
    from cellflow.analysis.artifact_reader import read_position_artifact
except ImportError:  # pragma: no cover - tests monkeypatch this when absent
    def read_position_artifact(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError("cellflow.analysis.artifact_reader is unavailable")


try:  # pragma: no cover - local branch compatibility
    from cellflow.napari.artifact_visualization import add_artifact_layers
except ImportError:  # pragma: no cover - tests monkeypatch this when absent
    def add_artifact_layers(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError("cellflow.napari.artifact_visualization is unavailable")


class _ProgressEmitter(QObject):
    progress = Signal(int, int, str)


class AnalysisWidget(QWidget):
    """Final analysis and export."""

    _artifact_layer_prefix = "[Artifact] "

    def __init__(self, viewer: object | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir: Path | None = None
        self._build_worker = None
        self._build_completion_pending = False
        self._build_error_pending = False
        self._progress_emitter = _ProgressEmitter(self)
        self._progress_emitter.progress.connect(self._on_build_progress)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        self.input_status_lbl = QLabel("")
        self.input_status_lbl.setWordWrap(True)
        status_label(self.input_status_lbl)
        layout.addWidget(self.input_status_lbl)

        self.artifact_path_lbl = QLabel("")
        self.artifact_path_lbl.setWordWrap(True)
        status_label(self.artifact_path_lbl)
        layout.addWidget(self.artifact_path_lbl)

        self.artifact_status_lbl = QLabel("")
        self.artifact_status_lbl.setWordWrap(True)
        status_label(self.artifact_status_lbl)
        layout.addWidget(self.artifact_status_lbl)

        self.artifact_progress_bar = QProgressBar()
        self.artifact_progress_bar.setRange(0, 100)
        self.artifact_progress_bar.setValue(0)
        self.artifact_progress_bar.setVisible(False)
        self.artifact_progress_bar.setTextVisible(True)
        layout.addWidget(self.artifact_progress_bar)

        self.build_artifact_btn = QPushButton("Build Position Artifact")
        action_button(self.build_artifact_btn, expand=True)
        layout.addWidget(self.build_artifact_btn)

        self.cancel_build_btn = QPushButton("Cancel")
        action_button(self.cancel_build_btn)
        self.cancel_build_btn.setEnabled(False)
        layout.addWidget(self.cancel_build_btn)

        self.show_artifact_btn = QPushButton("Show Artifact")
        action_button(self.show_artifact_btn, expand=True)
        layout.addWidget(self.show_artifact_btn)

        self.color_cells_by_label_cb = QCheckBox("Color cells by label")
        layout.addWidget(self.color_cells_by_label_cb)

        self.color_edges_by_id_cb = QCheckBox("Color edges by ID")
        layout.addWidget(self.color_edges_by_id_cb)

        self.color_edges_by_label_cb = QCheckBox("Color edges by label")
        layout.addWidget(self.color_edges_by_label_cb)

        self.hide_border_edges_cb = QCheckBox("Hide border edges")
        layout.addWidget(self.hide_border_edges_cb)

        self.clear_artifact_btn = QPushButton("Clear Artifact Layers")
        action_button(self.clear_artifact_btn, expand=True)
        layout.addWidget(self.clear_artifact_btn)

        layout.addStretch()

        self.build_artifact_btn.clicked.connect(self._on_build_artifact)
        self.cancel_build_btn.clicked.connect(self._on_cancel_build)
        self.show_artifact_btn.clicked.connect(self._on_show_artifact)
        self.clear_artifact_btn.clicked.connect(self._on_clear_artifact_layers)
        self.color_cells_by_label_cb.stateChanged.connect(self._on_visualizer_options_changed)
        self.color_edges_by_id_cb.stateChanged.connect(self._on_visualizer_options_changed)
        self.color_edges_by_label_cb.stateChanged.connect(self._on_visualizer_options_changed)
        self.hide_border_edges_cb.stateChanged.connect(self._on_visualizer_options_changed)
        self.refresh(None)

    @property
    def cell_labels_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "tracked_labels.tif" if self._pos_dir else None

    @property
    def nucleus_labels_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None

    @property
    def artifact_out_path(self) -> Path | None:
        return self._pos_dir / "4_analysis" / "position_analysis.h5" if self._pos_dir else None

    def refresh(self, pos_dir: Path | str | None) -> None:
        self._pos_dir = Path(pos_dir) if pos_dir is not None else None
        self._update_status()

    def _output_path_text(self) -> str:
        if self.artifact_out_path is None:
            return "Output: no project open."
        return f"Output: {self.artifact_out_path}"

    def _update_status(self) -> None:
        self.artifact_path_lbl.setText(self._output_path_text())
        self._update_input_status()
        self._update_action_states()
        if self._pos_dir is None:
            self._set_artifact_status("Status: no project open.")
        elif not self.artifact_status_lbl.text():
            self._set_artifact_status("Status: ready.")

    def _update_input_status(self) -> None:
        if self._pos_dir is None:
            self.input_status_lbl.setText("Inputs: no project open.")
            return

        cell_ok = self.cell_labels_path is not None and self.cell_labels_path.exists()
        nucleus_ok = self.nucleus_labels_path is not None and self.nucleus_labels_path.exists()
        artifact_ok = self.artifact_out_path is not None and self.artifact_out_path.exists()
        check = "✓"
        cross = "✗"
        self.input_status_lbl.setText(
            f"Inputs: {check if cell_ok else cross} cell labels  "
            f"{check if nucleus_ok else cross} nucleus labels  "
            f"{check if artifact_ok else cross} artifact"
        )

    def _update_action_states(self) -> None:
        inputs_ready = (
            self._pos_dir is not None
            and self.cell_labels_path is not None
            and self.cell_labels_path.exists()
            and self.nucleus_labels_path is not None
            and self.nucleus_labels_path.exists()
        )
        artifact_ready = (
            self.viewer is not None
            and self.artifact_out_path is not None
            and self.artifact_out_path.exists()
        )
        running = self._build_worker is not None
        self.build_artifact_btn.setEnabled(inputs_ready and not running)
        self.cancel_build_btn.setEnabled(running)
        self.show_artifact_btn.setEnabled(artifact_ready and not running)
        self.clear_artifact_btn.setEnabled(self.viewer is not None and not running)

    def _set_build_running(self, running: bool) -> None:
        self.artifact_progress_bar.setVisible(running)
        if running:
            self.artifact_progress_bar.setRange(0, 100)
            self.artifact_progress_bar.setValue(0)
        else:
            self.artifact_progress_bar.setValue(0)
            self.artifact_progress_bar.setRange(0, 100)
        self._update_action_states()

    def _set_artifact_status(self, message: str) -> None:
        self.artifact_status_lbl.setText(message)

    def _on_build_progress(self, done: int, total: int, message: str) -> None:
        if total > 0:
            self.artifact_progress_bar.setRange(0, total)
            self.artifact_progress_bar.setValue(done)
        self._set_artifact_status(f"Status: {message}")

    def _on_build_done(self, output_path: Path) -> None:
        self._build_completion_pending = True
        self._build_worker = None
        self._set_build_running(False)
        self._set_artifact_status(f"Status: Wrote {output_path}")
        self._update_status()

    def _on_build_error(self, exc: Exception) -> None:
        self._build_error_pending = True
        self._build_worker = None
        self._set_build_running(False)
        self._set_artifact_status(f"Status: error: {exc}")
        self._update_status()

    def _on_build_artifact(self) -> None:
        if self._pos_dir is None or self.artifact_out_path is None:
            self._set_artifact_status("Status: no project open.")
            self._update_action_states()
            return
        if self.cell_labels_path is None or not self.cell_labels_path.exists():
            self._set_artifact_status("Status: missing 3_cell/tracked_labels.tif.")
            self._update_status()
            return
        if self.nucleus_labels_path is None or not self.nucleus_labels_path.exists():
            self._set_artifact_status("Status: missing 2_nucleus/tracked_labels.tif.")
            self._update_status()
            return

        self._build_completion_pending = False
        self._build_error_pending = False
        self._set_artifact_status("Status: building position artifact...")
        self._set_build_running(True)

        @thread_worker(
            connect={
                "returned": self._on_build_done,
                "errored": self._on_build_error,
            }
        )
        def _worker():
            return build_position_analysis_artifact(
                self._pos_dir,
                self.artifact_out_path,
                cell_tracked_labels_path=self.cell_labels_path,
                nucleus_tracked_labels_path=self.nucleus_labels_path,
                progress_cb=self._progress_emitter.progress.emit,
            )

        worker = _worker()
        self._build_worker = worker
        if self._build_completion_pending or self._build_error_pending:
            self._build_worker = None
            self._build_completion_pending = False
            self._build_error_pending = False
            self._update_action_states()

    def _on_cancel_build(self) -> None:
        worker = self._build_worker
        if worker is not None:
            self._build_worker = None
            worker.quit()
        self._set_build_running(False)
        self._set_artifact_status("Status: build cancelled.")
        self._update_status()

    def _on_show_artifact(self) -> None:
        if self.viewer is None:
            self._set_artifact_status("Status: no viewer available.")
            self._update_action_states()
            return
        artifact_path = self.artifact_out_path
        if artifact_path is None or not artifact_path.exists():
            self._set_artifact_status("Status: artifact file not found.")
            self._update_action_states()
            return

        artifact = read_position_artifact(artifact_path)
        self._clear_artifact_layers(set_status=False)
        add_artifact_layers(
            self.viewer,
            artifact,
            prefix=self._artifact_layer_prefix,
            color_cells_by_label=self.color_cells_by_label_cb.isChecked(),
            color_edges_by_id=self.color_edges_by_id_cb.isChecked(),
            color_edges_by_label=self.color_edges_by_label_cb.isChecked(),
            hide_border_edges=self.hide_border_edges_cb.isChecked(),
        )
        self._set_artifact_status(f"Status: loaded {artifact_path.name}")
        self._update_action_states()

    def _on_visualizer_options_changed(self, _state: int) -> None:
        if self.viewer is None or not self._artifact_layer_names():
            return
        self._on_show_artifact()

    def _artifact_layer_names(self) -> list[str]:
        if self.viewer is None:
            return []
        names: list[str] = []
        for layer in list(self.viewer.layers):
            layer_name = getattr(layer, "name", layer)
            if isinstance(layer_name, str) and layer_name.startswith(self._artifact_layer_prefix):
                names.append(layer_name)
        return names

    def _on_clear_artifact_layers(self) -> None:
        if self.viewer is None:
            self._set_artifact_status("Status: no viewer available.")
            self._update_action_states()
            return

        self._clear_artifact_layers(set_status=True)
        self._update_action_states()

    def _clear_artifact_layers(self, *, set_status: bool) -> int:
        if self.viewer is None:
            return 0

        removed = 0
        names = self._artifact_layer_names()
        layers = self.viewer.layers
        for name in names:
            layer = None
            try:
                layer = layers[name]
            except Exception:
                layer = name
            try:
                layers.remove(layer)
            except Exception:
                try:
                    del layers[name]
                except Exception:
                    pass
            removed += 1

        if set_status:
            if removed:
                self._set_artifact_status(f"Status: cleared {removed} artifact layers.")
            else:
                self._set_artifact_status("Status: no artifact layers to clear.")
        return removed
