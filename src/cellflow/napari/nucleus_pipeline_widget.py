"""Pipeline action / worker-coordination widget for the nucleus workflow."""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._paths import NucleusArtifactPaths
from cellflow.napari._widget_helpers import (
    make_progress as _make_progress,
    make_status as _make_status,
    tool_btn as _tool_btn,
)
from cellflow.napari.ui_style import action_button as _action_button
from cellflow.database.validation import read_corrections, read_validated_tracks
from cellflow.segmentation import build_consensus_boundary, build_nucleus_averaged_maps
from cellflow.tracking_ultrack.db_build import apply_annotations_and_score
from cellflow.tracking_ultrack.export import export_tracked_labels
from cellflow.tracking_ultrack.multi_threshold import (
    build_ultrack_database_from_sources,
    preview_ultrack_source_stack_frame,
    write_ultrack_source_stacks,
)
from cellflow.tracking_ultrack.solve import run_solve

logger = logging.getLogger(__name__)

try:
    from ultrack.core.segmentation.processing import segment as _ultrack_segment
except ImportError:
    _ultrack_segment = None  # type: ignore[assignment]

# ── Layer name constants ──────────────────────────────────────────────────────
_TRACKED_LAYER = "Tracked: Nucleus"
_CONTOUR_LAYER = "Contour Map: Nucleus"
_FOREGROUND_SCORE_LAYER = "Foreground Score: Nucleus"
_NUC_ZAVG_LAYER = "Nucleus z-avg"


class NucleusPipelineWidget(QWidget):
    """Action buttons, workers, and coordination handlers for the nucleus pipeline."""

    def __init__(
        self,
        viewer: napari.Viewer,
        *,
        pos_dir_provider: Callable[[], Path | None],
        seg_inputs_provider: Callable,
        tracking_inputs_provider: Callable,
        refresh_files_callback: Callable[[Path | None], None],
        refresh_db_browser_callback: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._pos_dir_provider = pos_dir_provider
        self._seg_inputs_provider = seg_inputs_provider
        self._tracking_inputs_provider = tracking_inputs_provider
        self._refresh_files_callback = refresh_files_callback
        self._refresh_db_browser_callback = refresh_db_browser_callback

        self._contour_worker = None
        self._db_gen_worker = None
        self._ultrack_worker = None
        self._chain_remaining: list[str] = []

        self.stage_seg_check = QCheckBox("Segmentation inputs")
        self.stage_seg_check.setChecked(True)
        self.stage_seg_check.setToolTip(
            "Build averaged maps, contour and foreground source stacks."
        )
        self.stage_db_check = QCheckBox("Ultrack database")
        self.stage_db_check.setChecked(True)
        self.stage_db_check.setToolTip(
            "Build the Ultrack candidate database from source-stack artifacts."
        )
        self.stage_ultrack_check = QCheckBox("Ultrack solve")
        self.stage_ultrack_check.setChecked(True)
        self.stage_ultrack_check.setToolTip(
            "Solve ILP tracking and export tracked_labels.tif."
        )

        self.preview_contour_btn = _tool_btn(
            "\U0001F50D",
            "Preview the current frame's segmentation input source sweep "
            "without writing artifacts.",
        )

        self.run_btn = QPushButton("▶  Run")
        self.run_btn.setToolTip("Run the checked stages in order.")
        _action_button(self.run_btn, expand=True)
        self.cancel_btn = _tool_btn(
            "✕", "Cancel the currently running pipeline step."
        )
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setVisible(False)

        self.pipeline_status_lbl = _make_status()
        self.pipeline_progress_bar = _make_progress()

        self.preview_contour_btn.clicked.connect(self._on_preview_contour_maps)
        self.run_btn.clicked.connect(self._on_run_chain)
        self.cancel_btn.clicked.connect(self._on_cancel)

    # ── Layout helpers ────────────────────────────────────────────────────────

    def build_pipeline_block(self) -> QWidget:
        """Build the visible pipeline UI: stage checkboxes + Run / Cancel."""
        block = QWidget(self)
        lay = QVBoxLayout(block)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        def _stage_row(checkbox: QCheckBox, *trailing: QWidget) -> QHBoxLayout:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            row.addWidget(checkbox)
            row.addStretch(1)
            for w in trailing:
                row.addWidget(w)
            return row

        lay.addLayout(_stage_row(self.stage_seg_check, self.preview_contour_btn))
        lay.addLayout(_stage_row(self.stage_db_check))
        lay.addLayout(_stage_row(self.stage_ultrack_check))

        run_row = QHBoxLayout()
        run_row.setContentsMargins(0, 4, 0, 0)
        run_row.setSpacing(4)
        run_row.addWidget(self.run_btn, 1)
        run_row.addWidget(self.cancel_btn)
        lay.addLayout(run_row)

        return block

    # ── Chained run orchestration ────────────────────────────────────────────

    def _on_run_chain(self) -> None:
        queue: list[str] = []
        if self.stage_seg_check.isChecked():
            queue.append("seg")
        if self.stage_db_check.isChecked():
            queue.append("db")
        if self.stage_ultrack_check.isChecked():
            queue.append("ultrack")
        if not queue:
            self._status("No stages selected.")
            return
        self._chain_remaining = queue
        self._run_next_chain_step()

    _STAGE_DISPATCH = {
        "seg": ("_on_build_segmentation_inputs", "_contour_worker"),
        "db":  ("_on_run_db_generation",         "_db_gen_worker"),
        "ultrack": ("_on_run_ultrack",           "_ultrack_worker"),
    }

    def _run_next_chain_step(self) -> None:
        if not self._chain_remaining:
            self._set_pipeline_buttons_enabled(True)
            return
        stage = self._chain_remaining.pop(0)
        handler_name, worker_attr = self._STAGE_DISPATCH[stage]
        getattr(self, handler_name)()
        # If the handler early-returned (missing inputs, etc.) no worker is
        # in flight — stop chaining and restore button state.
        if getattr(self, worker_attr) is None:
            self._abort_chain()
            self._set_pipeline_buttons_enabled(True)

    def _chain_continue_or_finish(self) -> None:
        """Called by stage _done handlers; advances the chain if active."""
        if self._chain_remaining:
            self._run_next_chain_step()
        else:
            self._set_pipeline_buttons_enabled(True)

    def _abort_chain(self) -> None:
        self._chain_remaining = []

    # ── Path helpers ──────────────────────────────────────────────────────────

    @property
    def _pos_dir(self) -> Path | None:
        return self._pos_dir_provider()

    @property
    def _paths(self) -> NucleusArtifactPaths | None:
        pos = self._pos_dir
        return NucleusArtifactPaths(pos) if pos else None

    def _prob_path(self) -> Path | None:
        return self._paths.prob if self._paths else None

    def _dp_path(self) -> Path | None:
        return self._paths.dp if self._paths else None

    def _contours_path(self) -> Path | None:
        return self._paths.contours if self._paths else None

    def _contour_sources_path(self) -> Path | None:
        return self._paths.contour_sources if self._paths else None

    def _foreground_sources_path(self) -> Path | None:
        return self._paths.foreground_sources if self._paths else None

    def _foreground_scores_path(self) -> Path | None:
        return self._paths.foreground_scores if self._paths else None

    def _ultrack_workdir(self) -> Path | None:
        return self._paths.ultrack_workdir if self._paths else None

    def _ultrack_db_path(self) -> Path | None:
        return self._paths.ultrack_db if self._paths else None

    def _tracked_path(self) -> Path | None:
        return self._paths.tracked if self._paths else None

    def _nucleus_zavg_path(self) -> Path | None:
        return self._paths.nucleus_zavg if self._paths else None

    # ── Status / progress helpers ─────────────────────────────────────────────

    def _status(self, msg: str) -> None:
        self.pipeline_status_lbl.setText(msg)
        self.pipeline_status_lbl.setVisible(bool(msg))
        if msg:
            logger.info(msg)

    def _progress(self, done: int, total: int, msg: str) -> None:
        self.pipeline_progress_bar.setVisible(True)
        self.pipeline_progress_bar.setRange(0, total)
        self.pipeline_progress_bar.setValue(done)
        self._status(msg)

    def _on_progress(self, data) -> None:
        if isinstance(data, tuple):
            self._progress(*data)
        else:
            self._status(str(data))

    def _clear_progress(self) -> None:
        self.pipeline_progress_bar.setValue(0)
        self.pipeline_progress_bar.setVisible(False)

    def _set_pipeline_buttons_enabled(self, enabled: bool) -> None:
        for widget in (
            self.preview_contour_btn,
            self.run_btn,
            self.stage_seg_check,
            self.stage_db_check,
            self.stage_ultrack_check,
        ):
            widget.setEnabled(enabled)
        self.cancel_btn.setEnabled(not enabled)
        self.cancel_btn.setVisible(not enabled)

    # ── Threshold / config delegation ─────────────────────────────────────────

    def _map_cellprob_thresholds_from_controls(self):
        from cellflow.napari import _thresholds
        return _thresholds.map_cellprob_thresholds(self._seg_inputs_provider())

    def _map_z_indices_from_controls(self):
        from cellflow.napari import _thresholds
        return _thresholds.map_z_indices(self._seg_inputs_provider())

    def _source_contour_thresholds_from_controls(self):
        from cellflow.napari import _thresholds
        return _thresholds.source_contour_thresholds(self._seg_inputs_provider())

    def _source_foreground_thresholds_from_controls(self):
        from cellflow.napari import _thresholds
        return _thresholds.source_foreground_thresholds(self._seg_inputs_provider())

    def _db_gen_config_from_controls(self):
        return self._tracking_inputs_provider().db_gen_config()

    def _ultrack_config_from_controls(self):
        return self._tracking_inputs_provider().ultrack_config()

    # ── Viewer helpers ────────────────────────────────────────────────────────

    def _update_labels_layer(self, name: str, data: np.ndarray) -> None:
        from napari.layers import Labels
        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Labels):
            self.viewer.layers[name].data = data
            return
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        self.viewer.add_labels(data, name=name)

    def _update_tracked_display(
        self, labels: np.ndarray, t: int | None = None,
    ) -> None:
        if _TRACKED_LAYER in self.viewer.layers and t is not None:
            layer = self.viewer.layers[_TRACKED_LAYER]
            if layer.data.ndim == 3:
                if t < layer.data.shape[0]:
                    new_data = layer.data.copy()
                    new_data[t] = labels
                    layer.data = new_data
                    return
                new_data = np.concatenate(
                    [layer.data, labels[np.newaxis].astype(layer.data.dtype)], axis=0,
                )
                layer.data = new_data
                return
        display = labels[np.newaxis].copy() if labels.ndim == 2 else labels
        self._update_labels_layer(_TRACKED_LAYER, display)

    def _ensure_nucleus_zavg_layer(self) -> None:
        if _NUC_ZAVG_LAYER in self.viewer.layers:
            return
        zavg_path = self._nucleus_zavg_path()
        if zavg_path is None or not zavg_path.exists():
            return
        data = np.asarray(tifffile.imread(str(zavg_path)), dtype=np.float32)
        self.viewer.add_image(
            data,
            name=_NUC_ZAVG_LAYER,
            colormap="I Orange",
            blending="minimum",
            visible=True,
        )

    def _segmentation_preview_has_source_time_axes(self) -> bool:
        for name in (_CONTOUR_LAYER, _FOREGROUND_SCORE_LAYER):
            if name not in self.viewer.layers:
                continue
            data = np.asarray(self.viewer.layers[name].data)
            if data.ndim == 4:
                return True
        return False

    @staticmethod
    def _preview_frame_from_step(
        current_step: tuple[int, ...],
        frame_count: int,
        *,
        source_time_axes: bool,
    ) -> int:
        axis = 1 if source_time_axes and len(current_step) >= 2 else 0
        if not current_step:
            return 0
        return min(max(int(current_step[axis]), 0), frame_count - 1)

    def _ensure_tracked_layer_data(self) -> np.ndarray | None:
        """Return the tracked labelmap from the viewer layer if present, else
        read it from disk. Does not add anything to the viewer."""
        if _TRACKED_LAYER in self.viewer.layers:
            return np.asarray(self.viewer.layers[_TRACKED_LAYER].data)
        tracked_path = self._tracked_path()
        if tracked_path is None or not tracked_path.exists():
            return None
        self._status(f"Reading {tracked_path.name} from disk…")
        labels = np.asarray(tifffile.imread(str(tracked_path)), dtype=np.uint32)
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        return labels

    # ── Pipeline handlers — segmentation inputs ───────────────────────────────

    def _on_build_segmentation_inputs(self) -> None:
        pos_dir = self._pos_dir
        if pos_dir is None:
            self._status("No project open."); return
        prob_path = self._prob_path()
        dp_path = self._dp_path()
        contours_path = pos_dir / "2_nucleus" / "contours.tif"
        score_path = self._foreground_scores_path()
        contour_sources_path = self._contour_sources_path()
        foreground_sources_path = self._foreground_sources_path()
        if prob_path is None or not prob_path.exists():
            self._status(f"Missing: {prob_path}"); return
        if dp_path is None or not dp_path.exists():
            self._status(f"Missing: {dp_path}"); return
        if score_path is None or contour_sources_path is None or foreground_sources_path is None:
            self._status("No project open."); return
        try:
            map_thresholds = self._map_cellprob_thresholds_from_controls()
            z_indices = self._map_z_indices_from_controls()
            contour_thresholds = self._source_contour_thresholds_from_controls()
            foreground_thresholds = self._source_foreground_thresholds_from_controls()
        except ValueError as exc:
            self._status(str(exc)); return

        def _done(result):
            report, n_sources = result
            self._contour_worker = None
            self._clear_progress()
            self._refresh_files_callback(pos_dir)
            frames = int(getattr(report, "frames", 0))
            self._status(f"Segmentation inputs built ({frames} frames, {n_sources} sources).")
            self._chain_continue_or_finish()

        @thread_worker(connect={
            "yielded": self._on_progress,
            "returned": _done,
            "errored": self._on_contour_worker_error,
        })
        def _worker():
            import queue as _queue
            import threading

            msg_queue: _queue.SimpleQueue = _queue.SimpleQueue()
            result_holder: list = []
            exc_holder: list = []

            def _progress_cb(done: int, total: int, msg: str) -> None:
                msg_queue.put((done, total + 1, msg))

            def _run_maps() -> None:
                try:
                    result_holder.append(
                        build_nucleus_averaged_maps(
                            prob_path,
                            dp_path,
                            contours_path,
                            score_path,
                            cellprob_thresholds=map_thresholds,
                            z_indices=z_indices,
                            progress_cb=_progress_cb,
                        )
                    )
                except Exception as e:
                    exc_holder.append(e)

            t = threading.Thread(target=_run_maps, daemon=True)
            t.start()
            yield (0, 1, "Starting averaged-map build...")
            while t.is_alive() or not msg_queue.empty():
                try:
                    yield msg_queue.get_nowait()
                except _queue.Empty:
                    t.join(timeout=0.05)
            if exc_holder:
                raise exc_holder[0]
            report = result_holder[0]
            map_frames = max(1, int(getattr(report, "frames", 0)))
            yield (map_frames, map_frames + 1, "Building Ultrack source stacks...")
            metadata = write_ultrack_source_stacks(
                contours_path,
                score_path,
                contour_sources_path,
                foreground_sources_path,
                contour_thresholds=contour_thresholds,
                foreground_thresholds=foreground_thresholds,
            )
            yield (map_frames + 1, map_frames + 1, "Saved segmentation inputs.")
            return report, len(metadata)

        n_sources = len(contour_thresholds) * len(foreground_thresholds)
        self._status(
            f"Building segmentation inputs "
            f"({len(map_thresholds)} cellprob thresholds, {n_sources} sources)..."
        )
        self._set_pipeline_buttons_enabled(False)
        self._contour_worker = _worker()

    def _on_build_nucleus_maps(self) -> None:
        pos_dir = self._pos_dir
        if pos_dir is None:
            self._status("No project open."); return
        prob_path = self._prob_path()
        dp_path = self._dp_path()
        contours_path = pos_dir / "2_nucleus" / "contours.tif"
        score_path = self._foreground_scores_path()
        if prob_path is None or not prob_path.exists():
            self._status(f"Missing: {prob_path}"); return
        if dp_path is None or not dp_path.exists():
            self._status(f"Missing: {dp_path}"); return
        if score_path is None:
            self._status("No project open."); return
        try:
            thresholds = self._map_cellprob_thresholds_from_controls()
            z_indices = self._map_z_indices_from_controls()
        except ValueError as exc:
            self._status(str(exc)); return

        def _done(report):
            self._contour_worker = None
            self._set_pipeline_buttons_enabled(True)
            self._clear_progress()
            self._refresh_files_callback(pos_dir)
            frames = int(getattr(report, "frames", 0))
            self._status(f"Averaged maps built ({frames} frames).")

        @thread_worker(connect={
            "yielded": self._on_progress,
            "returned": _done,
            "errored": self._on_contour_worker_error,
        })
        def _worker():
            return build_nucleus_averaged_maps(
                prob_path,
                dp_path,
                contours_path,
                score_path,
                cellprob_thresholds=thresholds,
                z_indices=z_indices,
            )

        self._status(f"Building averaged maps ({len(thresholds)} cellprob thresholds)…")
        self._set_pipeline_buttons_enabled(False)
        self._contour_worker = _worker()

    def _on_build_contour_maps(self) -> None:
        pos_dir = self._pos_dir
        if pos_dir is None:
            self._status("No project open."); return
        contours_path = self._contours_path()
        score_path = self._foreground_scores_path()
        contour_sources_path = self._contour_sources_path()
        foreground_sources_path = self._foreground_sources_path()
        if contours_path is None or score_path is None:
            self._status("No project open."); return
        if not contours_path.exists():
            self._status("Missing: contours.tif — build segmentation inputs first."); return
        if not score_path.exists():
            self._status("Missing: foreground_scores.tif — build segmentation inputs first."); return
        if contour_sources_path is None or foreground_sources_path is None:
            self._status("No project open."); return

        try:
            contour_thresholds = self._source_contour_thresholds_from_controls()
            foreground_thresholds = self._source_foreground_thresholds_from_controls()
        except ValueError as exc:
            self._status(str(exc)); return

        def _done(result):
            pos_dir_result, n_sources = result
            self._contour_worker = None
            self._set_pipeline_buttons_enabled(True)
            self._clear_progress()
            self._refresh_files_callback(pos_dir_result)
            self._status(f"Ultrack source stacks built ({n_sources} sources).")

        @thread_worker(connect={
            "yielded": self._on_progress,
            "returned": _done,
            "errored": self._on_contour_worker_error,
        })
        def _worker():
            yield (0, 1, "Building Ultrack source stacks…")
            metadata = write_ultrack_source_stacks(
                contours_path,
                score_path,
                contour_sources_path,
                foreground_sources_path,
                contour_thresholds=contour_thresholds,
                foreground_thresholds=foreground_thresholds,
            )
            yield (1, 1, "Saved Ultrack source stacks.")
            return pos_dir, len(metadata)

        n_sources = len(contour_thresholds) * len(foreground_thresholds)
        self._status(f"Building Ultrack source stacks ({n_sources} sources)…")
        self._set_pipeline_buttons_enabled(False)
        self._contour_worker = _worker()

    def _on_preview_contour_maps(self) -> None:
        pos_dir = self._pos_dir
        if pos_dir is None:
            self._status("No project open."); return
        prob_path = self._prob_path()
        dp_path = self._dp_path()
        if prob_path is None or not prob_path.exists():
            self._status(f"Missing: {prob_path}"); return
        if dp_path is None or not dp_path.exists():
            self._status(f"Missing: {dp_path}"); return

        current_step = tuple(int(v) for v in self.viewer.dims.current_step)
        self._ensure_nucleus_zavg_layer()
        try:
            map_thresholds = self._map_cellprob_thresholds_from_controls()
            z_indices = self._map_z_indices_from_controls()
            contour_thresholds = self._source_contour_thresholds_from_controls()
            foreground_thresholds = self._source_foreground_thresholds_from_controls()
        except ValueError as exc:
            self._status(str(exc)); return

        def _done(result):
            self._contour_worker = None
            self._set_pipeline_buttons_enabled(True)
            self._clear_progress()
            contour_data, foreground_data, t_idx, frame_count, n_sources = result
            contour_data = np.asarray(contour_data)
            foreground_data = np.asarray(foreground_data)
            if contour_data.ndim == 2:
                contour_data = contour_data[np.newaxis, ...]
            if foreground_data.ndim == 2:
                foreground_data = foreground_data[np.newaxis, ...]
            if contour_data.ndim != 3 or foreground_data.ndim != 3:
                raise ValueError("Preview source frames must be PxYxX or YxX.")
            contour_stack = np.zeros(
                (contour_data.shape[0], frame_count) + contour_data.shape[1:],
                dtype=contour_data.dtype,
            )
            foreground_stack = np.zeros(
                (foreground_data.shape[0], frame_count) + foreground_data.shape[1:],
                dtype=foreground_data.dtype,
            )
            contour_stack[:, t_idx] = contour_data
            foreground_stack[:, t_idx] = foreground_data
            if _CONTOUR_LAYER in self.viewer.layers:
                self.viewer.layers[_CONTOUR_LAYER].data = contour_stack
            else:
                self.viewer.add_image(contour_stack, name=_CONTOUR_LAYER, colormap="magma", visible=True)
            self._update_labels_layer(_FOREGROUND_SCORE_LAYER, foreground_stack)
            self._refresh_files_callback(pos_dir)
            self._status(f"Preview segmentation inputs t={t_idx} — {n_sources} sources")

        @thread_worker(connect={
            "returned": _done, "errored": self._on_contour_worker_error,
        })
        def _worker():
            prob_stack = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
            dp_stack = np.asarray(tifffile.imread(str(dp_path)), dtype=np.float32)
            if prob_stack.ndim == 3:
                prob_stack = prob_stack[np.newaxis, ...]
            if dp_stack.ndim == 4:
                dp_stack = dp_stack[np.newaxis, ...]
            if prob_stack.ndim != 4:
                raise ValueError("nucleus_prob must be ZxYxX or TxZxYxX.")
            if dp_stack.ndim != 5 or dp_stack.shape[2] != 2:
                raise ValueError("nucleus_dp must be Zx2xYxX or TxZx2xYxX.")
            if prob_stack.shape[0] != dp_stack.shape[0]:
                raise ValueError("nucleus_prob and nucleus_dp must have the same frame count.")
            if prob_stack.shape[1] != dp_stack.shape[1]:
                raise ValueError("nucleus_prob and nucleus_dp must have the same z count.")
            if prob_stack.shape[2:] != dp_stack.shape[3:]:
                raise ValueError("nucleus_prob and nucleus_dp must have the same YxX shape.")

            preview_t = self._preview_frame_from_step(
                current_step,
                prob_stack.shape[0],
                source_time_axes=preview_has_source_time_axes,
            )
            if z_indices is None:
                z_sel = tuple(range(prob_stack.shape[1]))
            elif isinstance(z_indices, slice):
                start = 0 if z_indices.start is None else int(z_indices.start)
                stop = prob_stack.shape[1] if z_indices.stop is None else int(z_indices.stop)
                step = 1 if z_indices.step is None else int(z_indices.step)
                z_sel = tuple(range(start, stop, step))
            else:
                z_sel = tuple(int(z) for z in z_indices)
            bad_z = [z for z in z_sel if z < 0 or z >= prob_stack.shape[1]]
            if bad_z:
                raise ValueError(f"Z indices out of range for {prob_stack.shape[1]} z slices: {bad_z}")
            contours, foreground_scores = build_consensus_boundary(
                prob_stack[preview_t, z_sel],
                dp_stack[preview_t, z_sel],
                list(map_thresholds),
                gamma=1.0,
                flow_threshold=0.0,
            )
            contour_frame, foreground_frame, _, metadata = preview_ultrack_source_stack_frame(
                contours[np.newaxis, ...],
                foreground_scores[np.newaxis, ...],
                contour_thresholds=contour_thresholds,
                foreground_thresholds=foreground_thresholds,
                frame_index=0,
            )
            return contour_frame, foreground_frame, preview_t, prob_stack.shape[0], len(metadata)

        n_sources = len(contour_thresholds) * len(foreground_thresholds)
        preview_has_source_time_axes = self._segmentation_preview_has_source_time_axes()
        preview_axis = 1 if preview_has_source_time_axes and len(current_step) >= 2 else 0
        t_frame = int(current_step[preview_axis]) if current_step else 0
        self._status(
            f"Previewing segmentation inputs for frame t={t_frame} "
            f"({len(map_thresholds)} cellprob thresholds, {n_sources} sources)..."
        )
        self._set_pipeline_buttons_enabled(False)
        self._contour_worker = _worker()

    def _on_contour_worker_error(self, exc: Exception) -> None:
        self._contour_worker = None
        self._abort_chain()
        self._set_pipeline_buttons_enabled(True)
        self._clear_progress()
        self._status(f"Error: {exc}")
        logger.exception("Contour worker error", exc_info=exc)

    # ── Pipeline handlers — DB generation ────────────────────────────────────

    def _on_run_db_generation(self) -> None:
        pos_dir = self._pos_dir
        if pos_dir is None:
            self._status("No project open."); return
        contour_sources_path = self._contour_sources_path()
        foreground_sources_path = self._foreground_sources_path()
        if contour_sources_path is None or not contour_sources_path.exists():
            self._status("Missing: contour_sources.tif — run Build Sources first."); return
        if foreground_sources_path is None or not foreground_sources_path.exists():
            self._status("Missing: foreground_sources.tif — run Build Sources first."); return
        if _ultrack_segment is None:
            self._status("ultrack not installed — activate the cellflow conda environment."); return

        cfg = self._db_gen_config_from_controls()
        working_dir = self._ultrack_workdir()

        self.pipeline_progress_bar.setRange(0, 0)
        self.pipeline_progress_bar.setVisible(True)
        self._status("Starting DB generation…")
        self._set_pipeline_buttons_enabled(False)

        @thread_worker(connect={
            "yielded": self._on_progress,
            "returned": self._on_db_gen_done,
            "errored": self._on_db_gen_worker_error,
        })
        def _worker():
            import queue as _queue
            import threading

            msg_queue: _queue.SimpleQueue = _queue.SimpleQueue()
            result_holder: list = []
            exc_holder: list = []

            def _progress_cb(msg: str) -> None:
                msg_queue.put(msg)

            def _run() -> None:
                try:
                    build_ultrack_database_from_sources(
                        contour_sources_path=contour_sources_path,
                        foreground_sources_path=foreground_sources_path,
                        working_dir=working_dir,
                        cfg=cfg,
                        progress_cb=_progress_cb,
                    )
                    _progress_cb("Scoring node probabilities...")
                    score_path = self._foreground_scores_path()
                    apply_annotations_and_score(
                        working_dir=working_dir,
                        cfg=cfg,
                        score_signal_path=score_path,
                        corrections=None,
                        validated_tracks=None,
                        tracked_labels=None,
                    )
                    result_holder.append(pos_dir)
                except Exception as e:
                    exc_holder.append(e)

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            while t.is_alive() or not msg_queue.empty():
                try:
                    yield msg_queue.get_nowait()
                except _queue.Empty:
                    t.join(timeout=0.05)
            if exc_holder:
                raise exc_holder[0]
            return pos_dir

        self._db_gen_worker = _worker()

    def _on_db_gen_done(self, pos_dir: Path) -> None:
        self._db_gen_worker = None
        self._clear_progress()
        self._status("DB generation complete.")
        self._refresh_files_callback(pos_dir)
        self._refresh_db_browser_callback()
        self._chain_continue_or_finish()

    def _on_db_gen_worker_error(self, exc: Exception) -> None:
        self._db_gen_worker = None
        self._abort_chain()
        self._set_pipeline_buttons_enabled(True)
        self._clear_progress()
        self._status(f"Error: {exc}")
        logger.exception("DB generation worker error", exc_info=exc)

    # ── Pipeline handlers — Ultrack tracking ─────────────────────────────────

    def _on_run_ultrack(self) -> None:
        pos_dir = self._pos_dir
        if pos_dir is None:
            self._status("No project open."); return
        db_path = self._ultrack_db_path()
        if db_path is None or not db_path.exists():
            self._status("data.db not found — run DB Generation first."); return
        score_path = self._foreground_scores_path()
        if score_path is None or not score_path.exists():
            self._status("Missing: foreground_scores.tif — build segmentation inputs first."); return
        working_dir = self._ultrack_workdir()
        tracked_path = self._tracked_path()

        cfg = self._ultrack_config_from_controls()
        use_corrections = self._tracking_inputs_provider().db_gen_use_validated_check.isChecked()
        corrections = read_corrections(pos_dir) if use_corrections else None
        validated_tracks = (
            read_validated_tracks(pos_dir)
            if use_corrections and not corrections
            else None
        )
        tracked_labels = None
        if corrections or validated_tracks:
            tracked_labels = self._ensure_tracked_layer_data()
            if tracked_labels is None:
                self._status(
                    "Correction-aware solve requires tracked_labels.tif "
                    "(layer not loaded and file not on disk)."
                ); return

        self.pipeline_progress_bar.setRange(0, 100)
        self.pipeline_progress_bar.setVisible(True)
        self.pipeline_progress_bar.setValue(0)
        self._status("Starting Ultrack solve…")
        self._set_pipeline_buttons_enabled(False)

        @thread_worker(connect={
            "yielded": self._on_ultrack_progress,
            "returned": self._on_run_ultrack_done,
            "errored": self._on_ultrack_worker_error,
        })
        def _worker():
            yield "Applying annotations and scoring…"
            apply_annotations_and_score(
                working_dir=working_dir,
                cfg=cfg,
                score_signal_path=score_path,
                corrections=corrections,
                validated_tracks=validated_tracks,
                tracked_labels=tracked_labels,
            )
            for step, total, label in run_solve(working_dir, cfg, overwrite=True):
                yield (step, total, f"[solve] {label}")
            yield "Exporting tracked labels…"
            return export_tracked_labels(
                working_dir, cfg, tracked_path,
                corrections=corrections,
                validated_tracks=validated_tracks,
                tracked_labels=tracked_labels,
            )

        self._ultrack_worker = _worker()

    def _on_ultrack_progress(self, data) -> None:
        if isinstance(data, tuple):
            step, total, msg = data
            self._status(msg)
            if total > 0:
                self.pipeline_progress_bar.setRange(0, total)
                self.pipeline_progress_bar.setValue(step)
        else:
            self._status(str(data))

    def _on_run_ultrack_done(self, labels: np.ndarray | None) -> None:
        self._ultrack_worker = None
        self._clear_progress()
        if labels is None:
            self._abort_chain()
            self._set_pipeline_buttons_enabled(True)
            self._status("Ultrack tracking failed (no output).")
            return
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        nt = labels.shape[0]
        self._update_tracked_display(labels)
        self._refresh_files_callback(self._pos_dir)
        self._status(f"Tracking done: {nt} frame(s).")
        self._chain_continue_or_finish()

    def _on_ultrack_worker_error(self, exc: Exception) -> None:
        self._ultrack_worker = None
        self._abort_chain()
        self._set_pipeline_buttons_enabled(True)
        self._clear_progress()
        self._status(f"Error: {exc}")
        logger.exception("Ultrack worker error", exc_info=exc)

    # ── Cancel ────────────────────────────────────────────────────────────────

    def _on_cancel(self) -> None:
        cancelled = False
        for attr in ("_contour_worker", "_db_gen_worker", "_ultrack_worker"):
            worker = getattr(self, attr, None)
            if worker is not None:
                worker.quit()
                setattr(self, attr, None)
                cancelled = True
        self._abort_chain()
        self._set_pipeline_buttons_enabled(True)
        self._clear_progress()
        self._status("Cancelled." if cancelled else "Nothing running.")
