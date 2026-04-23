"""Nucleus hypothesis preview widget.

This replaces the old nucleus Ultrack panel with a lightweight selector-driven
preview for watershed hypotheses built from nucleus DP/probability inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cellflow.core.paths import stage_dir
from cellflow.napari.registry import get_state
from cellflow.napari.widgets import PipelineFilesWidget
from cellflow.ultrack.hypotheses import load_hypotheses_h5_lazy


def _first_existing(dir_path: Path, *names: str) -> Path | None:
    for name in names:
        path = dir_path / name
        if path.exists():
            return path
    return None


def _ensure_time_axis(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim in (2, 3, 4, 5):
        return arr
    raise ValueError(f"Unsupported stack shape: {arr.shape}")


def _select_timepoint(arr: np.ndarray, t: int) -> np.ndarray:
    arr = _ensure_time_axis(arr)
    if arr.ndim in (2, 3):
        return arr
    if t < 0 or t >= arr.shape[0]:
        raise IndexError(f"timepoint {t} out of range for shape {arr.shape}")
    return arr[t]


def _flow_magnitude(dp: np.ndarray) -> np.ndarray:
    """Compute an L2 magnitude image from a DP stack."""
    dp = np.asarray(dp, dtype=np.float32)

    if dp.ndim == 2:
        return np.abs(dp)

    if dp.ndim == 3:
        if dp.shape[0] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=0)).astype(np.float32)
        if dp.shape[-1] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=-1)).astype(np.float32)
        return np.abs(dp).astype(np.float32)

    if dp.ndim == 4:
        if dp.shape[1] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=1)).astype(np.float32)
        if dp.shape[-1] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=-1)).astype(np.float32)

    if dp.ndim == 5:
        if dp.shape[1] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=1)).astype(np.float32)
        if dp.shape[-1] in (2, 3):
            return np.sqrt(np.sum(dp * dp, axis=-1)).astype(np.float32)

    if dp.ndim >= 3 and dp.shape[0] in (2, 3):
        return np.sqrt(np.sum(dp * dp, axis=0)).astype(np.float32)

    if dp.ndim >= 3 and dp.shape[-1] in (2, 3):
        return np.sqrt(np.sum(dp * dp, axis=-1)).astype(np.float32)

    raise ValueError(f"Unsupported DP shape for magnitude computation: {dp.shape}")


def _as_2d_slice(arr: np.ndarray, z: int) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if z < 0 or z >= arr.shape[0]:
            raise IndexError(f"z={z} out of range for shape {arr.shape}")
        return arr[z]
    raise ValueError(f"Expected a 2D or 3D array, got shape {arr.shape}")


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    """Normalize an image to [0, 1] for consistent thresholding and display."""
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    scaled = (arr - lo) / (hi - lo)
    # Keep the top end just below 1.0 so no pixel is fully saturated.
    scaled = np.minimum(scaled, np.nextafter(np.float32(1.0), np.float32(0.0)))
    return scaled.astype(np.float32)


def _to_5d(data: np.ndarray) -> np.ndarray:
    """Promote 2D/3D/4D data to 5D (t, z, p, y, x) for consistent napari alignment."""
    data = np.asarray(data)
    if data.ndim == 2:  # (y, x) -> (1, 1, 1, y, x)
        return data[np.newaxis, np.newaxis, np.newaxis, ...]
    if data.ndim == 3:  # (t, y, x) -> (t, 1, 1, y, x)
        return data[:, np.newaxis, np.newaxis, ...]
    if data.ndim == 4:  # (t, z, y, x) -> (t, z, 1, y, x)
        return data[:, :, np.newaxis, ...]
    if data.ndim == 5:
        return data
    return data


def _update_layer(viewer, name: str, data: np.ndarray, *, kind: str = "image", **kwargs):
    data = np.asarray(data).copy()
    if name in viewer.layers:
        viewer.layers.remove(name)

    if kind == "labels":
        layer = viewer.add_labels(data, name=name, **kwargs)
    else:
        layer = viewer.add_image(data, name=name, **kwargs)
    layer.refresh()
    return layer


def _is_preview_layer_name(name: str | None) -> bool:
    return bool(name) and str(name).startswith("Preview: ")


class UltrackAnalysisWidget(QWidget):
    """Selector-driven preview for nucleus watershed hypotheses."""

    labels_loaded = Signal(object)  # napari Labels layer
    run_started = Signal()

    def __init__(self, viewer: "napari.Viewer", *, log_viewer=None) -> None:
        super().__init__()
        self.viewer = viewer
        self._state = get_state(viewer)
        self._log_viewer = log_viewer

        self._dp_stack: np.ndarray | None = None
        self._prob_stack: np.ndarray | None = None
        self._seed_stack: np.ndarray | None = None
        self._n_frames: int = 1
        self._z_count: int = 1
        self._preview_cache_key: tuple[str | None, int] | None = None
        self._worker = None

        self._h5_params: list = []  # list[NucleusHypothesisParams]
        self._h5_path_loaded: Path | None = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(6)
        lay.setAlignment(Qt.AlignTop)

        self._files_widget = PipelineFilesWidget([
            ("Input", [
                ("1_cellpose/nucleus_dp_4d.tif", "Nucleus DP 4D"),
                ("1_cellpose/nucleus_prob_4d.tif", "Nucleus prob 4D"),
                ("3_correction/nuclear_labels_corrected.tif", "Corrected nuclear labels"),
            ]),
            ("Output", [
                ("2_nucleus_ultrack/hypotheses.h5", "Hypotheses HDF5"),
                ("2_nucleus_ultrack/hypotheses_manifest.json", "Hypotheses manifest"),
                ("2_nucleus_ultrack/labelmaps/labelmap_*.tif", "Hypothesis labelmaps"),
            ]),
        ])
        lay.addWidget(self._files_widget)

        row = QHBoxLayout()
        row.addWidget(QLabel("Frame"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setRange(0, 0)
        self._frame_spin.setValue(0)
        row.addWidget(self._frame_spin)
        row.addWidget(QLabel("Z"))
        self._z_spin = QSpinBox()
        self._z_spin.setRange(0, 0)
        self._z_spin.setValue(0)
        row.addWidget(self._z_spin)
        row.addStretch()
        lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Basin"))
        self._basin_combo = QComboBox()
        self._basin_combo.addItems(["prob", "flow_mag"])
        self._basin_combo.setToolTip("Select the watershed basin image.")
        row.addWidget(self._basin_combo)
        row.addWidget(QLabel("Seed source"))
        self._seed_combo = QComboBox()
        self._seed_combo.addItems([
            "Auto",
            "Peak local max",
            "Viewer active layer",
            "Viewer nuclear labels",
            "Disk corrected labels",
            "Disk nucleus labels 2D",
        ])
        self._seed_combo.setToolTip(
            "Choose the label map used as watershed markers. Auto prefers the current "
            "nuclear labels layer, then corrected labels on disk, then nucleus_labels_2d.tif. "
            "Peak local max generates seeds automatically from the basin image."
        )
        row.addWidget(self._seed_combo)
        lay.addLayout(row)

        self._seed_dist_widget = QWidget()
        seed_dist_row = QHBoxLayout(self._seed_dist_widget)
        seed_dist_row.setContentsMargins(0, 0, 0, 0)
        seed_dist_row.addWidget(QLabel("Seed distance"))
        self._seed_dist_min_spin = QSpinBox()
        self._seed_dist_min_spin.setRange(1, 500)
        self._seed_dist_min_spin.setValue(5)
        self._seed_dist_min_spin.setToolTip("Minimum seed distance for preview and sweep start.")
        self._seed_dist_max_spin = QSpinBox()
        self._seed_dist_max_spin.setRange(1, 500)
        self._seed_dist_max_spin.setValue(5)
        self._seed_dist_max_spin.setToolTip("Maximum seed distance for sweep end.")
        self._seed_dist_step_spin = QSpinBox()
        self._seed_dist_step_spin.setRange(1, 200)
        self._seed_dist_step_spin.setValue(5)
        self._seed_dist_step_spin.setToolTip("Step size between seed distance values.")
        seed_dist_row.addWidget(QLabel("min"))
        seed_dist_row.addWidget(self._seed_dist_min_spin)
        seed_dist_row.addWidget(QLabel("max"))
        seed_dist_row.addWidget(self._seed_dist_max_spin)
        seed_dist_row.addWidget(QLabel("step"))
        seed_dist_row.addWidget(self._seed_dist_step_spin)
        self._seed_dist_widget.setVisible(False)
        lay.addWidget(self._seed_dist_widget)

        # Keep alias so preview helper still works
        self._min_distance_spin = self._seed_dist_min_spin

        self._seed_combo.currentTextChanged.connect(self._on_seed_source_changed)

        # Basins row
        row = QHBoxLayout()
        row.addWidget(QLabel("Basins"))
        self._basin_prob_check = QCheckBox("prob")
        self._basin_prob_check.setChecked(True)
        self._basin_flow_check = QCheckBox("flow mag")
        self._basin_flow_check.setChecked(True)
        row.addWidget(self._basin_prob_check)
        row.addWidget(self._basin_flow_check)
        row.addStretch()
        lay.addLayout(row)

        # Sweep range rows: min / max / step for each parameter
        def _dspin(lo, hi, val, decimals, step) -> QDoubleSpinBox:
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(decimals)
            s.setSingleStep(step)
            s.setValue(val)
            return s

        def _param_row(label: str, min_w, max_w, step_w) -> QHBoxLayout:
            r = QHBoxLayout()
            r.addWidget(QLabel(label))
            r.addWidget(QLabel("min"))
            r.addWidget(min_w)
            r.addWidget(QLabel("max"))
            r.addWidget(max_w)
            r.addWidget(QLabel("step"))
            r.addWidget(step_w)
            return r

        self._threshold_min_spin = _dspin(0.0, 100.0, 0.0, 1, 1.0)
        self._threshold_max_spin = _dspin(0.0, 100.0, 0.0, 1, 1.0)
        self._threshold_step_spin = _dspin(0.1, 100.0, 1.0, 1, 0.5)
        self._threshold_min_spin.setToolTip("Minimum basin threshold (%) in the sweep.")
        self._threshold_max_spin.setToolTip("Maximum basin threshold (%) in the sweep.")
        self._threshold_step_spin.setToolTip("Step size between threshold values.")
        lay.addLayout(_param_row("Threshold (%)", self._threshold_min_spin, self._threshold_max_spin, self._threshold_step_spin))

        self._compactness_min_spin = _dspin(0.0, 100.0, 0.0, 4, 0.01)
        self._compactness_max_spin = _dspin(0.0, 100.0, 0.0, 4, 0.01)
        self._compactness_step_spin = _dspin(0.0001, 100.0, 0.01, 4, 0.005)
        self._compactness_min_spin.setToolTip("Minimum compactness in the sweep.")
        self._compactness_max_spin.setToolTip("Maximum compactness in the sweep.")
        self._compactness_step_spin.setToolTip("Step size between compactness values.")
        lay.addLayout(_param_row("Compactness", self._compactness_min_spin, self._compactness_max_spin, self._compactness_step_spin))

        self._smooth_min_spin = _dspin(0.0, 100.0, 0.0, 4, 0.25)
        self._smooth_max_spin = _dspin(0.0, 100.0, 0.0, 4, 0.25)
        self._smooth_step_spin = _dspin(0.0001, 100.0, 0.25, 4, 0.125)
        self._smooth_min_spin.setToolTip("Minimum smooth sigma in the sweep.")
        self._smooth_max_spin.setToolTip("Maximum smooth sigma in the sweep.")
        self._smooth_step_spin.setToolTip("Step size between smooth sigma values.")
        lay.addLayout(_param_row("Smooth sigma", self._smooth_min_spin, self._smooth_max_spin, self._smooth_step_spin))

        # Keep single-value aliases so preview uses the min value
        self._basin_threshold_spin = self._threshold_min_spin
        self._compactness_spin = self._compactness_min_spin
        self._smooth_sigma_spin = self._smooth_min_spin

        row = QHBoxLayout()
        row.addWidget(QLabel("Min region size"))
        self._min_size_spin = QSpinBox()
        self._min_size_spin.setRange(0, 100_000)
        self._min_size_spin.setValue(0)
        self._min_size_spin.setToolTip(
            "Remove connected regions with fewer pixels than this value (0 = disabled)."
        )
        row.addWidget(self._min_size_spin)
        row.addStretch()
        lay.addLayout(row)

        self._preview_btn = QPushButton("Preview")
        self._preview_btn.clicked.connect(self._on_preview)
        lay.addWidget(self._preview_btn)

        self._batch_btn = QPushButton("Write HDF5 Sweep")
        self._batch_btn.clicked.connect(self._on_write_hdf5_sweep)
        lay.addWidget(self._batch_btn)

        self._load_hdf5_btn = QPushButton("Load HDF5")
        self._load_hdf5_btn.clicked.connect(self._on_load_hdf5)
        lay.addWidget(self._load_hdf5_btn)

        self._load_medoids_btn = QPushButton("Load Medoids")
        self._load_medoids_btn.setToolTip(
            "Load the medoid label stack from hypotheses.h5 and display it as a labels layer."
        )
        self._load_medoids_btn.clicked.connect(self._on_load_medoids)
        lay.addWidget(self._load_medoids_btn)

        # Parameter browser (visible after loading)
        self._param_combo = QComboBox()
        self._param_combo.setToolTip("Select a parameter set to display.")
        self._param_combo.setVisible(False)
        lay.addWidget(self._param_combo)

        row = QHBoxLayout()
        self._show_slice_btn = QPushButton("Show slice")
        self._show_slice_btn.setToolTip("Display the selected parameter set at the current frame and z.")
        self._show_slice_btn.clicked.connect(self._on_show_slice)
        self._show_slice_btn.setVisible(False)
        row.addWidget(self._show_slice_btn)
        self._use_as_labels_btn = QPushButton("Use as labels")
        self._use_as_labels_btn.setToolTip("Promote the shown HDF5 slice to nuclear labels.")
        self._use_as_labels_btn.clicked.connect(self._on_use_as_labels)
        self._use_as_labels_btn.setVisible(False)
        row.addWidget(self._use_as_labels_btn)
        lay.addLayout(row)

        self._batch_progress = QProgressBar()
        self._batch_progress.setVisible(False)
        lay.addWidget(self._batch_progress)

        self._status = QLabel("")
        lay.addWidget(self._status)

        self._state.pipeline_schema_changed.connect(self._sync_project_dir)
        self._state.position_changed.connect(self._sync_project_dir)
        self._state.nuclear_labels_changed.connect(self._sync_seed_defaults)
        self._sync_project_dir()

    # ------------------------------------------------------------------
    # Project sync
    # ------------------------------------------------------------------

    def _root_dir(self) -> Path | None:
        return self._state.project_dir

    def _input_dir(self) -> Path | None:
        root = self._root_dir()
        if root is None:
            return None
        return stage_dir(root, self._state.current_position, "cellpose_cluster")

    def _output_dir(self) -> Path | None:
        root = self._root_dir()
        if root is None:
            return None
        return stage_dir(root, self._state.current_position, "nucleus_ultrack")

    def _hdf5_path(self) -> Path | None:
        out_dir = self._output_dir()
        if out_dir is None:
            return None
        return out_dir / "hypotheses.h5"

    def _sync_project_dir(self) -> None:
        root = self._root_dir()
        if root is None:
            self._files_widget.refresh(None)
            self._dp_stack = None
            self._prob_stack = None
            self._seed_stack = None
            self._n_frames = 1
            self._z_count = 1
            self._preview_cache_key = None
            self._status.setText("No project open.")
            return

        pos_dir = Path(root) / f"pos{self._state.current_position:02d}"
        self._files_widget.refresh(pos_dir)
        self._preview_cache_key = None
        self._h5_params = []
        self._h5_path_loaded = None
        self._param_combo.clear()
        self._param_combo.setVisible(False)
        self._show_slice_btn.setVisible(False)
        self._use_as_labels_btn.setVisible(False)
        self._sync_seed_defaults()
        self._load_input_shapes()

    def _sync_seed_defaults(self) -> None:
        return

    def _on_seed_source_changed(self, text: str) -> None:
        self._seed_dist_widget.setVisible(text == "Peak local max")

    def _seed_stack_data(self) -> np.ndarray | None:
        root = self._root_dir()
        if root is None:
            return None

        seed_choice = self._seed_combo.currentText()
        pos = self._state.current_position

        if seed_choice == "Peak local max":
            return None  # computed per-slice in the sweep worker

        if seed_choice == "Viewer active layer":
            return self._active_labels_layer_data()
        if seed_choice == "Viewer nuclear labels":
            if _is_preview_layer_name(self._state.tissue.nuclear_labels_layer):
                return None
            return self._state.tissue.nuclear_labels
        if seed_choice == "Disk corrected labels":
            path = stage_dir(root, pos, "correction") / "nuclear_labels_corrected.tif"
            return tifffile.imread(str(path)).astype(np.int32) if path.exists() else None
        if seed_choice == "Disk nucleus labels 2D":
            path = stage_dir(root, pos, "nucleus_ultrack") / "nuclear_labels_2d.tif"
            return tifffile.imread(str(path)).astype(np.int32) if path.exists() else None

        active_data = self._active_labels_layer_data()
        if active_data is not None:
            return active_data

        path = stage_dir(root, pos, "correction") / "nuclear_labels_corrected.tif"
        if path.exists():
            return tifffile.imread(str(path)).astype(np.int32)

        path = stage_dir(root, pos, "nucleus_ultrack") / "nuclear_labels_2d.tif"
        if path.exists():
            return tifffile.imread(str(path)).astype(np.int32)

        if not _is_preview_layer_name(self._state.tissue.nuclear_labels_layer):
            if self._state.tissue.nuclear_labels is not None:
                return self._state.tissue.nuclear_labels

        return None

    def _time_slice(self, arr: np.ndarray, t: int) -> np.ndarray:
        """Slice along the leading time axis when the loaded stack has one."""
        arr = np.asarray(arr)
        if arr.ndim <= 2:
            return arr
        if arr.ndim in (4, 5):
            return arr[t]
        if arr.ndim == 3 and self._n_frames > 1 and arr.shape[0] == self._n_frames:
            return arr[t]
        return arr

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_input_shapes(self) -> None:
        input_dir = self._input_dir()
        if input_dir is None:
            return

        dp_path = _first_existing(input_dir, "nucleus_dp_4d.tif", "nucleus_dp.tif")
        prob_path = _first_existing(input_dir, "nucleus_prob_4d.tif", "nucleus_prob.tif")
        if dp_path is None or prob_path is None:
            self._status.setText("Missing nucleus_dp_4d.tif or nucleus_prob_4d.tif.")
            return

        try:
            self._dp_stack = tifffile.imread(str(dp_path)).astype(np.float32)
            raw_prob = tifffile.imread(str(prob_path)).astype(np.float32)
            self._prob_stack = 1.0 / (1.0 + np.exp(-raw_prob))
        except Exception as exc:
            self._status.setText(f"Could not load nucleus inputs: {exc}")
            return

        n_frames = 1
        if self._dp_stack.ndim in (4, 5):
            n_frames = int(self._dp_stack.shape[0])
        elif self._prob_stack.ndim in (4, 5):
            n_frames = int(self._prob_stack.shape[0])
        elif (
            self._dp_stack.ndim == 3
            and self._prob_stack.ndim == 3
            and self._dp_stack.shape[0] == self._prob_stack.shape[0]
            and self._dp_stack.shape[0] > 1
        ):
            n_frames = int(self._dp_stack.shape[0])

        self._n_frames = max(1, n_frames)
        self._frame_spin.setRange(0, max(0, self._n_frames - 1))
        if self._n_frames == 1:
            self._frame_spin.setValue(0)

        sample_prob = self._time_slice(self._prob_stack, 0)
        self._z_count = sample_prob.shape[0] if sample_prob.ndim == 3 else 1
        self._z_spin.setRange(0, max(0, self._z_count - 1))
        self._z_spin.setEnabled(self._z_count > 1)
        if self._z_count == 1:
            self._z_spin.setValue(0)

        self._status.setText(
            f"Loaded inputs: dp={self._dp_stack.shape} prob={self._prob_stack.shape}"
        )

    def _load_seed_stack(self, t: int) -> np.ndarray | None:
        root = self._root_dir()
        if root is None:
            return None

        pos = self._state.current_position
        candidates: list[tuple[str, np.ndarray | None]] = []

        seed_choice = self._seed_combo.currentText()
        if seed_choice == "Viewer active layer":
            candidates = [("Viewer active layer", self._active_labels_layer_data())]
        elif seed_choice == "Viewer nuclear labels":
            nuc_layer_name = self._state.tissue.nuclear_labels_layer
            if _is_preview_layer_name(nuc_layer_name):
                candidates = [("Viewer nuclear labels", None)]
            else:
                candidates = [("Viewer nuclear labels", self._state.tissue.nuclear_labels)]
        elif seed_choice == "Disk corrected labels":
            path = stage_dir(root, pos, "correction") / "nuclear_labels_corrected.tif"
            candidates = [("Disk corrected labels", tifffile.imread(str(path)) if path.exists() else None)]
        elif seed_choice == "Disk nucleus labels 2D":
            path = stage_dir(root, pos, "nucleus_ultrack") / "nuclear_labels_2d.tif"
            candidates = [("Disk nucleus labels 2D", tifffile.imread(str(path)) if path.exists() else None)]
        else:
            active_data = self._active_labels_layer_data()
            if active_data is not None:
                candidates.append(("Viewer active layer", active_data))

            path = stage_dir(root, pos, "correction") / "nuclear_labels_corrected.tif"
            if path.exists():
                candidates.append(("Disk corrected labels", tifffile.imread(str(path))))

            path = stage_dir(root, pos, "nucleus_ultrack") / "nuclear_labels_2d.tif"
            if path.exists():
                candidates.append(("Disk nucleus labels 2D", tifffile.imread(str(path))))

            nuc_layer_name = self._state.tissue.nuclear_labels_layer
            if not _is_preview_layer_name(nuc_layer_name):
                candidates.append(("Viewer nuclear labels", self._state.tissue.nuclear_labels))

        for _label, arr in candidates:
            if arr is None:
                continue
            arr = np.asarray(arr)
            try:
                return _select_timepoint(arr, t)
            except Exception:
                continue

        return None

    def _peak_local_max_seeds_2d(self, basin_2d: np.ndarray) -> np.ndarray:
        from scipy.ndimage import label as nd_label
        from skimage.feature import peak_local_max

        min_dist = int(self._min_distance_spin.value())
        coords = peak_local_max(
            np.asarray(basin_2d, dtype=np.float32),
            min_distance=min_dist,
            exclude_border=False,
        )
        mask = np.zeros(basin_2d.shape, dtype=bool)
        if coords.size:
            mask[coords[:, 0], coords[:, 1]] = True
        markers, _ = nd_label(mask)
        return markers.astype(np.int32)

    def _compute_peak_seed_stack(self) -> np.ndarray | None:
        """Compute a full seed stack from the loaded prob stack using peak local max."""
        if self._prob_stack is None:
            return None
        prob = np.asarray(self._prob_stack)
        n_frames = self._n_frames

        def _seeds_from_volume(vol: np.ndarray) -> np.ndarray:
            vol = _normalize_01(np.asarray(vol, dtype=np.float32))
            if vol.ndim == 2:
                return self._peak_local_max_seeds_2d(vol)
            slices = [self._peak_local_max_seeds_2d(vol[z]) for z in range(vol.shape[0])]
            return np.stack(slices, axis=0)

        if n_frames == 1:
            vol = self._time_slice(prob, 0)
            return _seeds_from_volume(vol)

        frames = [_seeds_from_volume(self._time_slice(prob, t)) for t in range(n_frames)]
        return np.stack(frames, axis=0)

    def _active_labels_layer_data(self) -> np.ndarray | None:
        layer_name = self._state.tissue.nuclear_labels_layer
        if _is_preview_layer_name(layer_name):
            return None
        if layer_name and layer_name in self.viewer.layers:
            layer = self.viewer.layers[layer_name]
            if getattr(layer, "data", None) is not None:
                return np.asarray(layer.data)
        active = getattr(self.viewer.layers.selection.active, "data", None)
        if active is not None:
            from napari.layers import Labels

            if isinstance(self.viewer.layers.selection.active, Labels) and not _is_preview_layer_name(getattr(self.viewer.layers.selection.active, "name", None)):
                return np.asarray(active)
        return None

    def _build_sweep_spec(self):
        from cellflow.ultrack.hypotheses import NucleusHypothesisSweepSpec

        basins = tuple(
            b for b, chk in (("prob", self._basin_prob_check), ("flow_mag", self._basin_flow_check))
            if chk.isChecked()
        ) or ("prob",)

        thr_min = float(self._threshold_min_spin.value())
        thr_max = float(self._threshold_max_spin.value())
        thr_step = float(self._threshold_step_spin.value())

        cmp_min = float(self._compactness_min_spin.value())
        cmp_max = float(self._compactness_max_spin.value())
        cmp_step = float(self._compactness_step_spin.value())

        smo_min = float(self._smooth_min_spin.value())
        smo_max = float(self._smooth_max_spin.value())
        smo_step = float(self._smooth_step_spin.value())

        sd_min = int(self._seed_dist_min_spin.value())
        sd_max = int(self._seed_dist_max_spin.value())
        sd_step = int(self._seed_dist_step_spin.value())

        seed_source = self._seed_combo.currentText()
        return NucleusHypothesisSweepSpec(
            basins=basins,
            threshold=thr_min,
            threshold_min=thr_min,
            threshold_max=thr_max,
            threshold_step=thr_step,
            compactness=cmp_min,
            compactness_min=cmp_min,
            compactness_max=cmp_max,
            compactness_step=cmp_step,
            smooth_sigma=smo_min,
            smooth_min=smo_min,
            smooth_max=smo_max,
            smooth_step=smo_step,
            seed_source=seed_source,
            seed_distance=sd_min,
            seed_distance_min=sd_min,
            seed_distance_max=sd_max,
            seed_distance_step=sd_step,
            min_size=int(self._min_size_spin.value()),
        )

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _on_preview(self) -> None:
        root = self._root_dir()
        if root is None:
            self._status.setText("No project open.")
            return

        if self._dp_stack is None or self._prob_stack is None:
            self._load_input_shapes()
        if self._dp_stack is None or self._prob_stack is None:
            return

        t = int(self._frame_spin.value())
        z = int(self._z_spin.value())
        basin_name = self._basin_combo.currentText()

        try:
            prob_t = self._time_slice(self._prob_stack, t)
            dp_t = self._time_slice(self._dp_stack, t)
            flow_mag = _flow_magnitude(dp_t)

            if prob_t.ndim == 3:
                prob_2d = _as_2d_slice(prob_t, z)
            else:
                prob_2d = prob_t
                z = 0

            if flow_mag.ndim == 3:
                flow_2d = _as_2d_slice(flow_mag, z)
            else:
                flow_2d = flow_mag

            flow_2d = _normalize_01(flow_2d)
            basin_2d = _normalize_01(prob_2d if basin_name == "prob" else flow_2d)
            if self._smooth_sigma_spin.value() > 0:
                from scipy.ndimage import gaussian_filter

                basin_2d = gaussian_filter(
                    np.asarray(basin_2d, dtype=np.float32),
                    sigma=float(self._smooth_sigma_spin.value()),
                )
                basin_2d = _normalize_01(basin_2d)

            seed_choice = self._seed_combo.currentText()
            if seed_choice == "Peak local max":
                markers = self._peak_local_max_seeds_2d(basin_2d)
            else:
                seed_stack = self._load_seed_stack(t)
                if seed_stack is None:
                    self._status.setText(
                        "No seed labels available. Use 'Peak local max' seed source or load nuclear labels."
                    )
                    return

                seed_stack = self._time_slice(seed_stack, t)

                if seed_stack.ndim == 3:
                    markers = _as_2d_slice(seed_stack, z)
                elif seed_stack.ndim == 2:
                    markers = seed_stack
                else:
                    self._status.setText(f"Unsupported seed shape: {seed_stack.shape}")
                    return

            markers = np.asarray(markers, dtype=np.int32)
            threshold = float(self._basin_threshold_spin.value()) / 100.0
            mask = (np.asarray(basin_2d, dtype=np.float32) >= threshold) | (markers > 0)

            from skimage.segmentation import watershed

            from cellflow.ultrack.hypotheses import _remove_small_labels

            labels = watershed(
                -np.asarray(basin_2d, dtype=np.float32),
                markers=markers,
                mask=mask,
                compactness=float(self._compactness_spin.value()),
                watershed_line=False,
            ).astype(np.uint32)
            labels = _remove_small_labels(labels, int(self._min_size_spin.value()))

            for layer_name in ("Preview: Nucleus prob", "Preview: Nucleus flow mag", "Preview: Nucleus hypotheses"):
                if layer_name in self.viewer.layers:
                    self.viewer.layers.remove(layer_name)

            _update_layer(
                self.viewer,
                "Preview: Nucleus prob",
                _to_5d(np.asarray(prob_2d, dtype=np.float32)),
                kind="image",
                colormap="gray",
                blending="additive",
            )
            _update_layer(
                self.viewer,
                "Preview: Nucleus flow mag",
                _to_5d(np.asarray(flow_2d, dtype=np.float32)),
                kind="image",
                colormap="magma",
                blending="additive",
                visible=(basin_name == "flow_mag"),
            )
            labels_layer = _update_layer(
                self.viewer,
                "Preview: Nucleus hypotheses",
                _to_5d(labels),
                kind="labels",
            )
            self.labels_loaded.emit(labels_layer)

            n_labels = int(np.unique(labels[labels > 0]).size)
            self._status.setText(
                f"Previewed t={t}, z={z}, basin={basin_name}, threshold={self._basin_threshold_spin.value():.1f}%, labels={n_labels}"
            )
            self._preview_cache_key = (basin_name, t)
        except Exception as exc:
            self._status.setText(f"Preview error: {exc}")

    def _on_write_hdf5_sweep(self) -> None:
        root = self._root_dir()
        if root is None:
            self._status.setText("No project open.")
            return

        seed_stack = None if self._seed_combo.currentText() == "Peak local max" else self._seed_stack_data()
        spec = self._build_sweep_spec()
        out_dir = self._output_dir()
        if out_dir is None:
            self._status.setText("No project output directory.")
            return

        self._batch_btn.setEnabled(False)
        self._preview_btn.setEnabled(False)
        self._batch_progress.setVisible(True)
        self._batch_progress.setValue(0)
        self._status.setText("Writing HDF5 sweep…")

        @thread_worker(
            connect={
                "yielded": self._on_batch_progress,
                "finished": self._on_batch_finished,
                "errored": self._on_batch_error,
            }
        )
        def _work():
            from cellflow.ultrack.stages.tracking import run_nucleus_hypothesis_sweep

            yield from run_nucleus_hypothesis_sweep(
                root,
                int(self._state.current_position),
                spec,
                seed_labels=seed_stack,
                overwrite=True,
            )

        self.run_started.emit()
        self._worker = _work()
        self._worker.aborted.connect(self._on_batch_cancelled)

    def _on_load_hdf5(self) -> None:
        path = self._hdf5_path()
        if path is None:
            self._status.setText("No project open.")
            return
        if not path.exists():
            self._status.setText("No hypotheses.h5 found. Run the sweep first.")
            return

        # Read parameter metadata for the combo (t000/z000/p###)
        try:
            import h5py
            from cellflow.ultrack.hypotheses import NucleusHypothesisParams

            params = []
            with h5py.File(path, "r") as h5:
                root = h5["hypotheses"]
                t_keys = sorted(k for k in root.keys() if k.startswith("t"))
                if not t_keys:
                    self._status.setText("HDF5 contains no timepoints.")
                    return
                t0 = root[t_keys[0]]
                z_keys = sorted(k for k in t0.keys() if k.startswith("z"))
                z0 = t0[z_keys[0]] if z_keys else t0
                for p_name in sorted(k for k in z0.keys() if k.startswith("p")):
                    grp = z0[p_name]
                    params.append(NucleusHypothesisParams(
                        basin=str(grp.attrs.get("basin", "?")),
                        threshold_pct=float(grp.attrs.get("threshold_pct", 0.0)),
                        compactness=float(grp.attrs.get("compactness", 0.0)),
                        smooth_sigma=float(grp.attrs.get("smooth_sigma", 0.0)),
                        seed_source=str(grp.attrs.get("seed_source", "auto")),
                        seed_distance=int(grp.attrs.get("seed_distance", 5)),
                    ))
                n_t = len(t_keys)
                n_z = len(z_keys) if z_keys else 1
        except Exception as exc:
            self._status.setText(f"Load error: {exc}")
            return

        # Load the full dataset lazily as a 5-D dask array (t, z, p, y, x)
        try:
            from cellflow.ultrack.hypotheses import load_hypotheses_h5_lazy
            data = load_hypotheses_h5_lazy(path)
        except Exception as exc:
            self._status.setText(f"Lazy load error: {exc}")
            return

        layer_name = "HDF5: Nucleus hypotheses"
        if layer_name in self.viewer.layers:
            self.viewer.layers.remove(layer_name)
        layer = self.viewer.add_labels(data, name=layer_name)
        try:
            self.viewer.dims.axis_labels = ("t", "z", "param", "y", "x")
        except Exception:
            pass
        layer.refresh()

        self._h5_params = params
        self._h5_path_loaded = path

        self._param_combo.blockSignals(True)
        self._param_combo.clear()
        for i, p in enumerate(params):
            self._param_combo.addItem(
                f"p{i:03d}  basin={p.basin}  thr={p.threshold_pct:.1f}%  "
                f"cmpct={p.compactness:.4g}  smooth={p.smooth_sigma:.4g}"
                + (f"  sdist={p.seed_distance}" if p.seed_source == "Peak local max" else "")
            )
        self._param_combo.blockSignals(False)

        self._param_combo.setVisible(True)
        self._show_slice_btn.setVisible(False)  # napari sliders handle navigation
        self._use_as_labels_btn.setVisible(True)

        self._status.setText(
            f"Loaded {path.name} — {len(params)} param sets, "
            f"{n_t} frames, {n_z} z-slices. Use napari sliders to inspect."
        )

    def _on_load_medoids(self) -> None:
        path = self._hdf5_path()
        if path is None:
            self._status.setText("No project open.")
            return
        if not path.exists():
            self._status.setText("No hypotheses.h5 found. Run the sweep first.")
            return

        try:
            from cellflow.ultrack.hypotheses import load_medoid_stack

            medoid_yxt = load_medoid_stack(path)  # (Y, X, T)
            medoid_stack = _to_5d(np.moveaxis(medoid_yxt, -1, 0).astype(np.uint32))  # (T, 1, 1, Y, X)
        except Exception as exc:
            self._status.setText(f"Medoid load error: {exc}")
            return

        layer_name = "Medoids: Nucleus hypotheses"
        if layer_name in self.viewer.layers:
            self.viewer.layers.remove(layer_name)
        layer = self.viewer.add_labels(medoid_stack, name=layer_name)
        try:
            self.viewer.dims.axis_labels = ("t", "z", "param", "y", "x")
        except Exception:
            pass
        layer.refresh()
        self.labels_loaded.emit(layer)

        n_t = medoid_stack.shape[0]
        self._status.setText(f"Loaded medoid stack: {n_t} frames, shape={medoid_stack.shape}")

    def _on_show_slice(self) -> None:
        pass  # navigation is handled by napari's own sliders

    def _on_use_as_labels(self) -> None:
        """Extract the slice currently shown in napari and emit it as nuclear labels."""
        path = self._h5_path_loaded
        if path is None or not path.exists():
            self._status.setText("Load HDF5 first.")
            return

        layer_name = "HDF5: Nucleus hypotheses"
        if layer_name not in self.viewer.layers:
            self._status.setText("HDF5 layer not found in viewer.")
            return

        # Read the current slider position from napari dims.
        # For a 5-D layer (t, z, p, y, x) the first 3 dims are the sliders.
        step = tuple(self.viewer.dims.current_step)
        ndim = self.viewer.layers[layer_name].data.ndim
        slider_indices = step[:ndim - 2]  # strip spatial y, x
        t = int(slider_indices[0]) if len(slider_indices) > 0 else 0
        z = int(slider_indices[1]) if len(slider_indices) > 1 else 0
        p = int(slider_indices[2]) if len(slider_indices) > 2 else 0

        try:
            from cellflow.ultrack.hypotheses import _read_hypothesis_labels
            labels = _read_hypothesis_labels(str(path), t, z, p)
        except Exception as exc:
            self._status.setText(f"Use as labels error: {exc}")
            return

        promote_name = "Nuclear labels (from HDF5)"
        if promote_name in self.viewer.layers:
            self.viewer.layers.remove(promote_name)
        layer = self.viewer.add_labels(_to_5d(labels), name=promote_name)
        try:
            self.viewer.dims.axis_labels = ("t", "z", "param", "y", "x")
        except Exception:
            pass
        layer.refresh()
        self.labels_loaded.emit(layer)

        params = self._h5_params[p] if p < len(self._h5_params) else None
        desc = (
            f"basin={params.basin}  thr={params.threshold_pct:.1f}%  "
            f"cmpct={params.compactness:.4g}  smooth={params.smooth_sigma:.4g}"
            + (f"  sdist={params.seed_distance}" if params and params.seed_source == "Peak local max" else "")
            if params else ""
        )
        self._status.setText(f"Using t={t}, z={z}, p={p} as labels  {desc}")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        return {
            "preview_frame": int(self._frame_spin.value()),
            "preview_z": int(self._z_spin.value()),
            "basin": self._basin_combo.currentText(),
            "seed_source": self._seed_combo.currentText(),
            "seed_distance_min": int(self._seed_dist_min_spin.value()),
            "seed_distance_max": int(self._seed_dist_max_spin.value()),
            "seed_distance_step": int(self._seed_dist_step_spin.value()),
            "basins_prob": self._basin_prob_check.isChecked(),
            "basins_flow_mag": self._basin_flow_check.isChecked(),
            "threshold_min": float(self._threshold_min_spin.value()),
            "threshold_max": float(self._threshold_max_spin.value()),
            "threshold_step": float(self._threshold_step_spin.value()),
            "compactness_min": float(self._compactness_min_spin.value()),
            "compactness_max": float(self._compactness_max_spin.value()),
            "compactness_step": float(self._compactness_step_spin.value()),
            "smooth_min": float(self._smooth_min_spin.value()),
            "smooth_max": float(self._smooth_max_spin.value()),
            "smooth_step": float(self._smooth_step_spin.value()),
            "min_size": int(self._min_size_spin.value()),
        }

    def set_params(self, data: dict) -> None:
        if "preview_frame" in data:
            self._frame_spin.setValue(int(data["preview_frame"]))
        if "preview_z" in data:
            self._z_spin.setValue(int(data["preview_z"]))
        if "basin" in data:
            self._basin_combo.setCurrentText(str(data["basin"]))
        if "seed_source" in data:
            self._seed_combo.setCurrentText(str(data["seed_source"]))
        if "seed_distance_min" in data:
            self._seed_dist_min_spin.setValue(int(data["seed_distance_min"]))
        elif "min_seed_distance" in data:
            self._seed_dist_min_spin.setValue(int(data["min_seed_distance"]))
        if "seed_distance_max" in data:
            self._seed_dist_max_spin.setValue(int(data["seed_distance_max"]))
        if "seed_distance_step" in data:
            self._seed_dist_step_spin.setValue(int(data["seed_distance_step"]))
        if "basins_prob" in data:
            self._basin_prob_check.setChecked(bool(data["basins_prob"]))
        if "basins_flow_mag" in data:
            self._basin_flow_check.setChecked(bool(data["basins_flow_mag"]))
        if "threshold_min" in data:
            self._threshold_min_spin.setValue(float(data["threshold_min"]))
        elif "basin_threshold_pct" in data:
            self._threshold_min_spin.setValue(float(data["basin_threshold_pct"]))
        elif "basin_threshold" in data:
            v = float(data["basin_threshold"])
            self._threshold_min_spin.setValue(v * 100.0 if 0.0 <= v <= 1.0 else v)
        elif "prob_threshold" in data:
            v = float(data["prob_threshold"])
            self._threshold_min_spin.setValue(v * 100.0 if 0.0 <= v <= 1.0 else v)
        if "threshold_max" in data:
            self._threshold_max_spin.setValue(float(data["threshold_max"]))
        if "threshold_step" in data:
            self._threshold_step_spin.setValue(float(data["threshold_step"]))
        if "compactness_min" in data:
            self._compactness_min_spin.setValue(float(data["compactness_min"]))
        elif "compactness" in data:
            self._compactness_min_spin.setValue(float(data["compactness"]))
        if "compactness_max" in data:
            self._compactness_max_spin.setValue(float(data["compactness_max"]))
        if "compactness_step" in data:
            self._compactness_step_spin.setValue(float(data["compactness_step"]))
        if "smooth_min" in data:
            self._smooth_min_spin.setValue(float(data["smooth_min"]))
        elif "smooth_sigma" in data:
            self._smooth_min_spin.setValue(float(data["smooth_sigma"]))
        if "smooth_max" in data:
            self._smooth_max_spin.setValue(float(data["smooth_max"]))
        if "smooth_step" in data:
            self._smooth_step_spin.setValue(float(data["smooth_step"]))
        if "min_size" in data:
            self._min_size_spin.setValue(int(data["min_size"]))

    def _on_batch_progress(self, update: tuple) -> None:
        done, total, label = update
        self._batch_progress.setMaximum(max(total, 1))
        self._batch_progress.setValue(done)
        pct = int(100 * done / total) if total > 0 else 0
        self._status.setText(f"{done}/{total} ({pct}%) — {label}")

    def _on_batch_finished(self) -> None:
        self._batch_btn.setEnabled(True)
        self._preview_btn.setEnabled(True)
        self._batch_progress.setVisible(False)
        # Leave whatever status the last yield already set; don't clobber errors.
        self._worker = None

    def _on_batch_error(self, exc: Exception) -> None:
        self._batch_btn.setEnabled(True)
        self._preview_btn.setEnabled(True)
        self._batch_progress.setVisible(False)
        self._status.setText(f"Sweep error: {exc}")
        self._worker = None

    def _on_batch_cancelled(self) -> None:
        self._batch_btn.setEnabled(True)
        self._preview_btn.setEnabled(True)
        self._batch_progress.setVisible(False)
        self._status.setText("Sweep cancelled.")
        self._worker = None
