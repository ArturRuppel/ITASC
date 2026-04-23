"""Seeded tracker widget for building tracked labels frame by frame."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.core.paths import stage_dir
from cellflow.napari.registry import get_state
from cellflow.napari.widgets import PipelineFilesWidget
from cellflow.ultrack.stages.seeded_tracker import (
    _frame_to_2d,
    _match_frame,
    _relabel_sequential,
    build_seeded_tracker_inputs,
    match_frame_from_h5,
)


def _raw_import_dir(root_dir, pos):
    return stage_dir(root_dir, pos, "raw_import")


def _nucleus_ultrack_dir(root_dir, pos):
    return stage_dir(root_dir, pos, "nucleus_ultrack")


class SeededTrackerWidget(QWidget):
    """Compact bootstrap-and-advance tracker UI."""

    run_started = Signal()

    def __init__(self, viewer: "napari.Viewer", *, log_viewer=None) -> None:
        super().__init__()
        self.viewer = viewer
        self._state = get_state(viewer)
        self._log_viewer = log_viewer

        self._consensus_stack: np.ndarray | None = None
        self._tracked_stack: np.ndarray | None = None
        self._track_rows: list[dict[str, object]] = []
        self._current_t: int = -1
        self._seed_source: str = "consensus"
        self._tracked_layer = None
        self._h5_path: Path | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)
        lay.setAlignment(Qt.AlignTop)

        self._files_widget = PipelineFilesWidget([
            ("Input", [
                ("0_input/cell_zavg.tif", "Cell z-avg"),
                ("0_input/nucleus_zavg.tif", "Nucleus z-avg"),
                ("2_nucleus_ultrack/hypotheses.h5", "Hypotheses HDF5"),
            ]),
            ("Output", [
                ("2_nucleus_ultrack/tracked_labels.tif", "Tracked labels"),
                ("2_nucleus_ultrack/tracks.csv", "Tracks CSV"),
                ("2_nucleus_ultrack/seeded_tracker_state.json", "Tracker state"),
            ]),
        ])
        lay.addWidget(self._files_widget)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Max distance (px):"))
        self._max_dist_spin = QDoubleSpinBox()
        self._max_dist_spin.setRange(1.0, 9999.0)
        self._max_dist_spin.setValue(50.0)
        self._max_dist_spin.setSingleStep(5.0)
        param_row.addWidget(self._max_dist_spin)
        param_row.addWidget(QLabel("Max size dev:"))
        self._max_size_dev_spin = QDoubleSpinBox()
        self._max_size_dev_spin.setRange(0.0, 10.0)
        self._max_size_dev_spin.setValue(0.5)
        self._max_size_dev_spin.setSingleStep(0.05)
        param_row.addWidget(self._max_size_dev_spin)
        lay.addLayout(param_row)

        row = QHBoxLayout()
        self._load_btn = QPushButton("Load backgrounds")
        self._load_btn.clicked.connect(self._load_backgrounds)
        row.addWidget(self._load_btn)
        self._bootstrap_btn = QPushButton("Bootstrap seed")
        self._bootstrap_btn.clicked.connect(self._on_bootstrap)
        row.addWidget(self._bootstrap_btn)
        self._next_btn = QPushButton("Show best next")
        self._next_btn.clicked.connect(self._on_best_next)
        row.addWidget(self._next_btn)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.clicked.connect(self._on_reset)
        row.addWidget(self._reset_btn)
        lay.addLayout(row)

        self._status = QLabel("No seed loaded.")
        lay.addWidget(self._status)

        self._state.pipeline_schema_changed.connect(self._sync_project_dir)
        self._state.position_changed.connect(self._sync_project_dir)
        self._sync_project_dir()

    def _sync_project_dir(self) -> None:
        project_dir = self._state.project_dir
        if project_dir is None:
            self._files_widget.refresh(None)
            return
        self._files_widget.refresh(Path(project_dir) / f"pos{self._state.current_position:02d}")

    def _root_dir(self) -> Path | None:
        return Path(self._state.project_dir) if self._state.project_dir else None

    def _output_dir(self) -> Path | None:
        root = self._root_dir()
        if root is None:
            return None
        return _nucleus_ultrack_dir(root, self._state.current_position)

    def _load_backgrounds(self) -> None:
        root = self._root_dir()
        if root is None:
            self._status.setText("No project open.")
            return

        pos = int(self._state.current_position)
        raw_dir = _raw_import_dir(root, pos)
        cell_path = raw_dir / "cell_zavg.tif"
        nuc_path = raw_dir / "nucleus_zavg.tif"

        if cell_path.exists():
            cell_img = tifffile.imread(str(cell_path))
            layer_name = "Cell avg"
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].data = cell_img
            else:
                self.viewer.add_image(cell_img, name=layer_name, colormap="gray")

        if nuc_path.exists():
            nuc_img = tifffile.imread(str(nuc_path))
            layer_name = "Nucleus avg"
            if layer_name in self.viewer.layers:
                layer = self.viewer.layers[layer_name]
                layer.data = nuc_img
                layer.colormap = "bop orange"
                layer.blending = "additive"
            else:
                self.viewer.add_image(
                    nuc_img,
                    name=layer_name,
                    colormap="bop orange",
                    blending="additive",
                )

        self._status.setText("Backgrounds loaded.")

    def _bootstrap_stack(self) -> None:
        root = self._root_dir()
        if root is None:
            raise RuntimeError("No project open.")

        out_dir = self._output_dir()
        if out_dir is None:
            raise RuntimeError("Could not determine output directory.")

        self._load_backgrounds()

        h5_path = out_dir / "hypotheses.h5"
        if h5_path.exists():
            from cellflow.ultrack.hypotheses import load_medoid_stack
            medoid_yxt = load_medoid_stack(h5_path)  # (Y, X, T)
            consensus_stack = np.moveaxis(medoid_yxt, -1, 0).astype(np.uint32)  # (T, Y, X)
            seed = consensus_stack[0]
            seed_source = "medoid_stack:h5"
            self._h5_path = h5_path
        else:
            self._h5_path = None
            labelmaps, consensus_stack, seed, seed_source = build_seeded_tracker_inputs(out_dir)
            if not labelmaps:
                raise FileNotFoundError(f"No hypothesis labelmaps found in {out_dir}")
            seed = _frame_to_2d(seed)

        if consensus_stack.ndim == 2:
            n_frames = 1
            frame_shape = seed.shape
        else:
            n_frames = int(consensus_stack.shape[0])
            frame_shape = _frame_to_2d(consensus_stack[0]).shape

        if seed.shape != frame_shape:
            raise ValueError(f"Seed shape {seed.shape} does not match frame shape {frame_shape}")

        tracked_stack = np.zeros((n_frames,) + frame_shape, dtype=np.uint32)
        tracked_stack[0] = _relabel_sequential(seed)[0]

        self._consensus_stack = consensus_stack
        self._tracked_stack = tracked_stack
        self._current_t = 0
        self._seed_source = seed_source
        self._track_rows = []

        for src_label, track_id in _relabel_sequential(seed)[1].items():
            self._track_rows.append(
                {
                    "track_id": track_id,
                    "time": 0,
                    "source_track_id": track_id,
                    "source_label_id": src_label,
                    "candidate_label_id": src_label,
                    "iou": 1.0,
                }
            )

        self._write_outputs()
        self._set_layer_data()
        self._status.setText(
            f"Bootstrapped from {seed_source}; frame 0 ready for correction."
        )

    def _on_bootstrap(self) -> None:
        root = self._root_dir()
        if root is None:
            self._status.setText("No project open.")
            return

        self.run_started.emit()
        try:
            self._bootstrap_stack()
        except Exception as exc:
            self._status.setText(f"Bootstrap failed: {exc}")

    def _current_frame(self) -> np.ndarray:
        if self._tracked_stack is None:
            raise RuntimeError("Bootstrap the tracker first.")
        if self._tracked_layer is not None and self._tracked_layer in self.viewer.layers:
            data = np.asarray(self._tracked_layer.data, dtype=np.uint32)
            if data.ndim >= 3 and self._current_t < data.shape[0]:
                return np.asarray(data[self._current_t], dtype=np.uint32)
        return np.asarray(self._tracked_stack[self._current_t], dtype=np.uint32)

    def _candidate_frame(self, frame_index: int) -> np.ndarray:
        if self._consensus_stack is None:
            raise RuntimeError("Bootstrap the tracker first.")
        if self._consensus_stack.ndim == 2:
            return _frame_to_2d(self._consensus_stack)
        return _frame_to_2d(self._consensus_stack[frame_index])

    def _set_layer_data(self) -> None:
        if self._tracked_stack is None:
            return
        layer_name = "tracked_labels"
        data = self._tracked_stack
        if layer_name in self.viewer.layers:
            self.viewer.layers[layer_name].data = data
            self._tracked_layer = self.viewer.layers[layer_name]
        else:
            self._tracked_layer = self.viewer.add_labels(data, name=layer_name)

    def _write_outputs(self) -> None:
        out_dir = self._output_dir()
        if out_dir is None or self._tracked_stack is None:
            return

        out_dir.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(
            str(out_dir / "tracked_labels.tif"),
            self._tracked_stack.astype(np.uint32, copy=False),
            compression="zlib",
            photometric="minisblack",
        )

        with (out_dir / "tracks.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "track_id",
                    "time",
                    "source_track_id",
                    "source_label_id",
                    "candidate_label_id",
                    "iou",
                ],
            )
            writer.writeheader()
            for row in self._track_rows:
                writer.writerow(row)

        state = {
            "version": 1,
            "seed_source": self._seed_source,
            "current_time": self._current_t,
            "frame_count": int(self._tracked_stack.shape[0]),
            "track_count": int(self._tracked_stack.max()) if self._tracked_stack.size else 0,
            "tracked_labels": "tracked_labels.tif",
            "tracks": "tracks.csv",
        }
        (out_dir / "seeded_tracker_state.json").write_text(
            json.dumps(state, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _on_best_next(self) -> None:
        root = self._root_dir()
        if root is None:
            self._status.setText("No project open.")
            return
        if self._tracked_stack is None or self._consensus_stack is None:
            self._status.setText("Bootstrap the tracker first.")
            return

        # Synchronize current time with viewer's current step to allow re-tracking from any frame
        if len(self.viewer.dims.current_step) > 0:
            t = self.viewer.dims.current_step[0]
            if 0 <= t < self._tracked_stack.shape[0]:
                self._current_t = t

        if self._current_t >= self._tracked_stack.shape[0] - 1:
            self._status.setText("No next frame available.")
            return

        if self._tracked_layer is not None and self._tracked_layer in self.viewer.layers:
            self._tracked_stack = np.asarray(self._tracked_layer.data, dtype=np.uint32)

        current = self._current_frame()
        next_index = self._current_t + 1

        # Truncate track rows for this and future frames to keep CSV consistent when re-tracking
        self._track_rows = [row for row in self._track_rows if row["time"] < next_index]

        if self._h5_path is not None and self._h5_path.exists():
            self._status.setText(f"Matching frame {next_index} via H5 candidates…")
            next_frame, rows = match_frame_from_h5(
                current,
                self._h5_path,
                next_index,
                max_distance=self._max_dist_spin.value(),
                max_size_deviation=self._max_size_dev_spin.value(),
            )
        else:
            candidate = self._candidate_frame(next_index)
            next_frame, rows = _match_frame(current, _relabel_sequential(candidate)[0])

        self._tracked_stack[next_index] = next_frame
        for row in rows:
            row["time"] = next_index
            self._track_rows.append(row)
        self._current_t = next_index
        self._write_outputs()
        self._set_layer_data()

        # Advance viewer to the next frame
        if len(self.viewer.dims.current_step) > 0:
            self.viewer.dims.set_current_step(0, next_index)

        self._status.setText(f"Advanced to frame {self._current_t} ({len(rows)} labels matched).")

    def _on_reset(self) -> None:
        self._consensus_stack = None
        self._tracked_stack = None
        self._track_rows = []
        self._current_t = -1
        self._next_track_id = 1
        self._seed_source = "consensus"
        self._tracked_layer = None
        self._h5_path = None
        try:
            if "tracked_labels" in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers["tracked_labels"])
        except Exception:
            pass
        self._status.setText("Tracker reset.")
