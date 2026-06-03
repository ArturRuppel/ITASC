"""Contact analysis widget for final processing and export in CellFlow v2."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from napari.qt.threading import thread_worker
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QCheckBox, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from cellflow.contact_analysis import build_position_contact_analysis
from cellflow.napari.ui_gate import ControlClass, UiGate
from cellflow.napari.ui_style import action_button, status_label
from cellflow.napari.widgets import (
    CollapsibleSection,
    PipelineFilesWidget,
    make_pipeline_files_header,
)

try:  # pragma: no cover - local branch compatibility
    from cellflow.contact_analysis.reader import read_position_contact_analysis
except ImportError:  # pragma: no cover - tests monkeypatch this when absent
    def read_position_contact_analysis(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError("cellflow.contact_analysis.reader is unavailable")


try:  # pragma: no cover - local branch compatibility
    from cellflow.napari.contact_analysis_visualization import (
        add_contact_analysis_layers,
        _nucleus_centroids_by_track,
    )
except ImportError:  # pragma: no cover - tests monkeypatch this when absent
    def add_contact_analysis_layers(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError("cellflow.napari.contact_analysis_visualization is unavailable")

    def _nucleus_centroids_by_track(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError("cellflow.napari.contact_analysis_visualization is unavailable")


class _ProgressEmitter(QObject):
    progress = Signal(int, int, str)


class ContactAnalysisWidget(QWidget):
    """Final contact analysis and export."""

    _contact_analysis_layer_prefix = "[Contact Analysis] "

    def __init__(
        self,
        viewer: object | None = None,
        parent: QWidget | None = None,
        gate: UiGate | None = None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        #: App-wide UI gate; a private one is created for standalone use.
        self.gate = gate if gate is not None else UiGate(self)
        self._pos_dir: Path | None = None
        self._build_worker = None
        self._build_completion_pending = False
        self._build_error_pending = False
        self._progress_emitter = _ProgressEmitter(self)
        self._progress_emitter.progress.connect(self._on_build_progress)
        self._cached_contact_analysis_path: Path | None = None
        self._cached_contact_analysis: Any = None
        self._cached_cell_labels: np.ndarray | None = None
        self._cached_nucleus_labels: np.ndarray | None = None
        self._cached_track_centroids: dict | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        self._files_widget = PipelineFilesWidget(
            [
                ("Inputs", [
                    ("2_nucleus/tracked_labels.tif", "Nucleus tracked labels"),
                    ("3_cell/tracked_labels.tif", "Cell tracked labels"),
                ]),
                ("Output", [
                    ("4_contact_analysis/contact_analysis.h5", "Contact analysis"),
                ]),
            ],
            viewer=self.viewer,
        )
        self._pipeline_files_section = CollapsibleSection(
            "Pipeline Files",
            self._files_widget,
            expanded=False,
        )
        (
            self.pipeline_files_header,
            self.pipeline_files_header_lbl,
            self.pipeline_files_toggle_btn,
        ) = make_pipeline_files_header(
            self._pipeline_files_section,
            stage_key="contact_analysis",
            parent=self,
        )
        layout.addWidget(self.pipeline_files_header)
        layout.addWidget(self._pipeline_files_section)

        self.contact_analysis_status_lbl = QLabel("")
        self.contact_analysis_status_lbl.setWordWrap(True)
        status_label(self.contact_analysis_status_lbl)
        layout.addWidget(self.contact_analysis_status_lbl)

        self.contact_analysis_progress_bar = QProgressBar()
        self.contact_analysis_progress_bar.setRange(0, 100)
        self.contact_analysis_progress_bar.setValue(0)
        self.contact_analysis_progress_bar.setVisible(False)
        self.contact_analysis_progress_bar.setTextVisible(True)
        layout.addWidget(self.contact_analysis_progress_bar)

        self.build_contact_analysis_btn = QPushButton("Build Contact Analysis")
        action_button(self.build_contact_analysis_btn, expand=True)
        layout.addWidget(self.build_contact_analysis_btn)

        self.cancel_build_btn = QPushButton("Cancel")
        action_button(self.cancel_build_btn)
        self.cancel_build_btn.setEnabled(False)
        layout.addWidget(self.cancel_build_btn)

        self.show_contact_analysis_btn = QPushButton("Show Contact Analysis")
        action_button(self.show_contact_analysis_btn, expand=True)
        layout.addWidget(self.show_contact_analysis_btn)

        self.color_cells_by_label_cb = QCheckBox("Color cells by label")
        layout.addWidget(self.color_cells_by_label_cb)

        self.color_edges_by_id_cb = QCheckBox("Color edges by ID")
        layout.addWidget(self.color_edges_by_id_cb)

        self.color_edges_by_label_cb = QCheckBox("Color edges by label")
        layout.addWidget(self.color_edges_by_label_cb)

        self.hide_border_edges_cb = QCheckBox("Hide border edges")
        layout.addWidget(self.hide_border_edges_cb)

        self.clear_contact_analysis_btn = QPushButton("Clear Contact Analysis Layers")
        action_button(self.clear_contact_analysis_btn, expand=True)
        layout.addWidget(self.clear_contact_analysis_btn)

        layout.addStretch()

        self.build_contact_analysis_btn.clicked.connect(self._on_build_contact_analysis)
        self.cancel_build_btn.clicked.connect(self._on_cancel_build)
        self.show_contact_analysis_btn.clicked.connect(self._on_show_contact_analysis)
        self.clear_contact_analysis_btn.clicked.connect(self._on_clear_contact_analysis_layers)
        self._register_gate_controls()
        self.refresh(None)

    def _register_gate_controls(self) -> None:
        """Register contact-analysis actions with the app-wide UI gate.

        Build/cancel are headless (disk only), so they run regardless of viewer
        ownership. Show/clear write the viewer, so they are blocked while a
        viewer owner (correction / live preview) is active.
        """
        g = self.gate
        running = lambda: self._build_worker is not None  # noqa: E731
        g.register(
            self.build_contact_analysis_btn,
            ControlClass.RUN_HEADLESS,
            when=lambda: self._inputs_ready() and not running(),
        )
        g.register(self.cancel_build_btn, ControlClass.RUN_HEADLESS, when=running)
        g.register(
            self.show_contact_analysis_btn,
            ControlClass.RUN_VIEWER,
            when=lambda: self._contact_analysis_ready() and not running(),
        )
        g.register(
            self.clear_contact_analysis_btn,
            ControlClass.RUN_VIEWER,
            when=lambda: self.viewer is not None and not running(),
        )
        g.recompute()

    @property
    def cell_labels_path(self) -> Path | None:
        return self._pos_dir / "3_cell" / "tracked_labels.tif" if self._pos_dir else None

    @property
    def nucleus_labels_path(self) -> Path | None:
        return self._pos_dir / "2_nucleus" / "tracked_labels.tif" if self._pos_dir else None

    @property
    def contact_analysis_out_path(self) -> Path | None:
        return self._pos_dir / "4_contact_analysis" / "contact_analysis.h5" if self._pos_dir else None

    def refresh(self, pos_dir: Path | str | None) -> None:
        new_pos_dir = Path(pos_dir) if pos_dir is not None else None
        if new_pos_dir != self._pos_dir:
            self._cached_contact_analysis_path = None
            self._cached_contact_analysis = None
            self._cached_cell_labels = None
            self._cached_nucleus_labels = None
            self._cached_track_centroids = None
        self._pos_dir = new_pos_dir
        self._files_widget.refresh(new_pos_dir)
        self._update_status()

    def _update_status(self) -> None:
        self._update_action_states()
        if self._pos_dir is None:
            self._set_contact_analysis_status("Status: no project open.")
        elif not self.contact_analysis_status_lbl.text():
            self._set_contact_analysis_status("Status: ready.")

    def _inputs_ready(self) -> bool:
        return (
            self._pos_dir is not None
            and self.cell_labels_path is not None
            and self.cell_labels_path.exists()
            and self.nucleus_labels_path is not None
            and self.nucleus_labels_path.exists()
        )

    def _contact_analysis_ready(self) -> bool:
        return (
            self.viewer is not None
            and self.contact_analysis_out_path is not None
            and self.contact_analysis_out_path.exists()
        )

    def _update_action_states(self) -> None:
        # Enablement is owned by the UI gate; its ``when`` predicates read the
        # readiness helpers and ``self._build_worker``.
        self.gate.recompute()

    def _set_build_running(self, running: bool) -> None:
        self.contact_analysis_progress_bar.setVisible(running)
        if running:
            self.contact_analysis_progress_bar.setRange(0, 100)
            self.contact_analysis_progress_bar.setValue(0)
        else:
            self.contact_analysis_progress_bar.setValue(0)
            self.contact_analysis_progress_bar.setRange(0, 100)
        self._update_action_states()

    def _set_contact_analysis_status(self, message: str) -> None:
        self.contact_analysis_status_lbl.setText(message)

    def _on_build_progress(self, done: int, total: int, message: str) -> None:
        if total > 0:
            self.contact_analysis_progress_bar.setRange(0, total)
            self.contact_analysis_progress_bar.setValue(done)
        self._set_contact_analysis_status(f"Status: {message}")

    def _on_build_done(self, output_path: Path) -> None:
        self._build_completion_pending = True
        self._build_worker = None
        self._set_build_running(False)
        self._set_contact_analysis_status(f"Status: Wrote {output_path}")
        self._update_status()

    def _on_build_error(self, exc: Exception) -> None:
        self._build_error_pending = True
        self._build_worker = None
        self._set_build_running(False)
        self._set_contact_analysis_status(f"Status: error: {exc}")
        self._update_status()

    def _on_build_contact_analysis(self) -> None:
        if self._pos_dir is None or self.contact_analysis_out_path is None:
            self._set_contact_analysis_status("Status: no project open.")
            self._update_action_states()
            return
        if self.cell_labels_path is None or not self.cell_labels_path.exists():
            self._set_contact_analysis_status("Status: missing 3_cell/tracked_labels.tif.")
            self._update_status()
            return
        if self.nucleus_labels_path is None or not self.nucleus_labels_path.exists():
            self._set_contact_analysis_status("Status: missing 2_nucleus/tracked_labels.tif.")
            self._update_status()
            return

        self._build_completion_pending = False
        self._build_error_pending = False
        self._set_contact_analysis_status("Status: building contact analysis...")
        self._set_build_running(True)

        @thread_worker(
            connect={
                "returned": self._on_build_done,
                "errored": self._on_build_error,
            }
        )
        def _worker():
            return build_position_contact_analysis(
                self._pos_dir,
                self.contact_analysis_out_path,
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
        self._set_contact_analysis_status("Status: build cancelled.")
        self._update_status()

    def _on_show_contact_analysis(self) -> None:
        if self.viewer is None:
            self._set_contact_analysis_status("Status: no viewer available.")
            self._update_action_states()
            return
        contact_analysis_path = self.contact_analysis_out_path
        if contact_analysis_path is None or not contact_analysis_path.exists():
            self._set_contact_analysis_status("Status: contact analysis file not found.")
            self._update_action_states()
            return

        # Cache contact analysis to avoid re-reading HDF5 on every Show click
        if self._cached_contact_analysis_path != contact_analysis_path:
            self._cached_contact_analysis = read_position_contact_analysis(contact_analysis_path)
            self._cached_contact_analysis_path = contact_analysis_path
            self._cached_cell_labels = None
            self._cached_nucleus_labels = None
            self._cached_track_centroids = None

        # Cache label TIFFs — these are large files whose repeated reading blocks
        # the Qt main thread and causes freezes + ghost layer artifacts
        if self._cached_cell_labels is None:
            if self.cell_labels_path is not None and self.cell_labels_path.exists():
                try:
                    self._cached_cell_labels = np.asarray(tifffile.imread(self.cell_labels_path))
                except Exception:
                    pass
        if self._cached_nucleus_labels is None:
            if self.nucleus_labels_path is not None and self.nucleus_labels_path.exists():
                try:
                    self._cached_nucleus_labels = np.asarray(tifffile.imread(self.nucleus_labels_path))
                except Exception:
                    pass

        # Cache nucleus track centroids — O(T*W*H*N) pixel iteration, very expensive
        if self._cached_track_centroids is None and self._cached_nucleus_labels is not None:
            try:
                self._cached_track_centroids = _nucleus_centroids_by_track(self._cached_nucleus_labels)
            except Exception:
                pass

        self._clear_contact_analysis_layers(set_status=False)
        show_kwargs: dict[str, Any] = {
            "prefix": self._contact_analysis_layer_prefix,
            "color_cells_by_label": self.color_cells_by_label_cb.isChecked(),
            "color_edges_by_id": self.color_edges_by_id_cb.isChecked(),
            "color_edges_by_label": self.color_edges_by_label_cb.isChecked(),
            "hide_border_edges": self.hide_border_edges_cb.isChecked(),
        }
        if self._cached_cell_labels is not None:
            show_kwargs["cell_labels"] = self._cached_cell_labels
        if self._cached_nucleus_labels is not None:
            show_kwargs["nucleus_labels"] = self._cached_nucleus_labels
        if self._cached_track_centroids is not None:
            show_kwargs["nucleus_track_centroids"] = self._cached_track_centroids
        add_contact_analysis_layers(self.viewer, self._cached_contact_analysis, **show_kwargs)
        self._set_contact_analysis_status(f"Status: loaded {contact_analysis_path.name}")
        self._update_action_states()

    def _contact_analysis_layer_names(self) -> list[str]:
        if self.viewer is None:
            return []
        names: list[str] = []
        for layer in list(self.viewer.layers):
            layer_name = getattr(layer, "name", layer)
            if isinstance(layer_name, str) and layer_name.startswith(self._contact_analysis_layer_prefix):
                names.append(layer_name)
        return names

    def _on_clear_contact_analysis_layers(self) -> None:
        if self.viewer is None:
            self._set_contact_analysis_status("Status: no viewer available.")
            self._update_action_states()
            return

        self._clear_contact_analysis_layers(set_status=True)
        self._update_action_states()

    def _clear_contact_analysis_layers(self, *, set_status: bool) -> int:
        if self.viewer is None:
            return 0

        removed = 0
        names = self._contact_analysis_layer_names()
        layers = self.viewer.layers
        for name in names:
            layer = None
            try:
                layer = layers[name]
            except Exception:
                layer = name
            cleanup = getattr(layer, "_cellflow_frame_shape_cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    pass
            # Hide before removal so napari clears the canvas visual first,
            # preventing a ghost frame from persisting in the viewport
            try:
                layer.visible = False
            except Exception:
                pass
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
                self._set_contact_analysis_status(f"Status: cleared {removed} contact analysis layers.")
            else:
                self._set_contact_analysis_status("Status: no contact analysis layers to clear.")
        return removed
