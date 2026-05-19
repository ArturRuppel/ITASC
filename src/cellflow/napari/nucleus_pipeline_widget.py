"""Pipeline action / worker-coordination widget for the nucleus workflow."""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._paths import NucleusArtifactPaths
from cellflow.napari._widget_helpers import (
    make_progress as _make_progress,
    make_status as _make_status,
    tool_btn as _tool_btn,
)
from cellflow.napari.ui_style import stage_header_label as _stage_header_label
from cellflow.database.validation import read_corrections, read_validated_tracks
from cellflow.segmentation import CancelledError
from cellflow.tracking_ultrack.db_build import apply_annotations_and_score
from cellflow.tracking_ultrack.export import export_tracked_labels
from cellflow.tracking_ultrack.multi_threshold import (
    build_ultrack_database_from_threshold_pairs,
    build_ultrack_source_stacks_from_pairs,
)
from cellflow.tracking_ultrack.solve import run_solve

logger = logging.getLogger(__name__)


def _ultrack_available() -> bool:
    """Return True if the ultrack package is importable.

    Kept as a function so the (slow) ultrack import only happens when an
    action actually needs it, not at widget construction time.
    """
    try:
        import ultrack.core.segmentation.processing  # noqa: F401
    except ImportError:
        return False
    return True


# ── Layer name constants ──────────────────────────────────────────────────────
_TRACKED_LAYER = "Tracked: Nucleus"
_PREVIEW_CONTOURS_LAYER = "Ultrack Preview: Contours"
_PREVIEW_FOREGROUND_LAYER = "Ultrack Preview: Foreground"


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
        self._contour_cancel: threading.Event | None = None
        self._running_stage: str | None = None

        # ── Per-stage buttons ──────────────────────────────────────────
        self.seg_params_btn = _tool_btn("⚙", "Show parameters for this stage.", checkable=True)
        self.seg_run_btn = _tool_btn("▶", "Run segmentation inputs.")

        self.db_params_btn = _tool_btn("⚙", "Show parameters for this stage.", checkable=True)
        self.db_run_btn = _tool_btn("▶", "Run Ultrack database build.")

        self.solve_params_btn = _tool_btn("⚙", "Show parameters for this stage.", checkable=True)
        self.solve_run_btn = _tool_btn("▶", "Run Ultrack solve.")

        self.pipeline_status_lbl = _make_status()
        self.pipeline_progress_bar = _make_progress()

        self.seg_run_btn.clicked.connect(self._on_seg_run_btn_clicked)
        self.db_run_btn.clicked.connect(self._on_db_run_btn_clicked)
        self.solve_run_btn.clicked.connect(self._on_solve_run_btn_clicked)

    # ── Layout helpers ────────────────────────────────────────────────────────

    # ── Per-row run/cancel dispatchers ───────────────────────────────────────

    def _on_seg_run_btn_clicked(self) -> None:
        if self._running_stage is not None:
            self._on_cancel()
        else:
            self._on_build_segmentation_inputs()

    def _on_db_run_btn_clicked(self) -> None:
        if self._running_stage is not None:
            self._on_cancel()
        else:
            self._on_run_db_generation()

    def _on_solve_run_btn_clicked(self) -> None:
        if self._running_stage is not None:
            self._on_cancel()
        else:
            self._on_run_ultrack()

    def build_pipeline_block(
        self,
        seg_section=None,
        db_section=None,
        solve_section=None,
    ) -> QWidget:
        """Build the three per-stage rows with inline params blocks."""
        block = QWidget(self)
        lay = QVBoxLayout(block)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        def _stage_label(text: str) -> QLabel:
            lbl = QLabel(text)
            return _stage_header_label(lbl, "nucleus")

        def _stage_row(label: QLabel, *trailing: QWidget) -> QHBoxLayout:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            row.addWidget(label)
            row.addStretch(1)
            for w in trailing:
                row.addWidget(w)
            return row

        # ── Ultrack database ─────────────────────────────────────────
        lay.addLayout(_stage_row(
            _stage_label("Ultrack database"),
            self.db_params_btn,
            self.db_run_btn,
        ))
        if db_section is not None:
            self.db_params_btn.toggled.connect(
                lambda checked: db_section._toggle.setChecked(checked)
            )
            lay.addWidget(db_section)

        # ── Ultrack solve ────────────────────────────────────────────
        lay.addLayout(_stage_row(
            _stage_label("Ultrack solve"),
            self.solve_params_btn,
            self.solve_run_btn,
        ))
        if solve_section is not None:
            self.solve_params_btn.toggled.connect(
                lambda checked: solve_section._toggle.setChecked(checked)
            )
            lay.addWidget(solve_section)

        return block

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
        return self._paths.nucleus_contours if self._paths else None

    def _contour_sources_path(self) -> Path | None:
        return self._paths.contour_sources if self._paths else None

    def _foreground_sources_path(self) -> Path | None:
        return self._paths.foreground_sources if self._paths else None

    def _foreground_path(self) -> Path | None:
        return self._paths.nucleus_foreground if self._paths else None

    def _ultrack_workdir(self) -> Path | None:
        return self._paths.ultrack_workdir if self._paths else None

    def _ultrack_db_path(self) -> Path | None:
        return self._paths.ultrack_db if self._paths else None

    def _tracked_path(self) -> Path | None:
        return self._paths.tracked if self._paths else None

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
        if self._running_stage is None:
            return
        if isinstance(data, tuple):
            self._progress(*data)
        else:
            self._status(str(data))

    def _clear_progress(self) -> None:
        self.pipeline_progress_bar.setValue(0)
        self.pipeline_progress_bar.setVisible(False)

    def _set_running_stage(self, stage_key: str | None) -> None:
        """Update run/cancel button states based on which stage is running.

        ``None`` means idle: all ▶ enabled, no ✕.
        ``"seg" | "db" | "ultrack"`` means that stage shows ✕, others disabled.
        """
        self._running_stage = stage_key
        _rows = {
            "seg":    (self.seg_params_btn,   self.seg_run_btn),
            "db":     (self.db_params_btn,    self.db_run_btn),
            "ultrack":(self.solve_params_btn, self.solve_run_btn),
        }
        if stage_key is None:
            for params_btn, run_btn in _rows.values():
                params_btn.setEnabled(True)
                run_btn.setEnabled(True)
            self.seg_run_btn.setText("▶")
            self.seg_run_btn.setToolTip("Run segmentation inputs.")
            self.db_run_btn.setText("▶")
            self.db_run_btn.setToolTip("Run Ultrack database build.")
            self.solve_run_btn.setText("▶")
            self.solve_run_btn.setToolTip("Run Ultrack solve.")
        else:
            for key, (params_btn, run_btn) in _rows.items():
                if key == stage_key:
                    run_btn.setText("✕")
                    run_btn.setToolTip("Cancel.")
                    run_btn.setEnabled(True)
                    params_btn.setEnabled(True)
                else:
                    params_btn.setEnabled(False)
                    run_btn.setEnabled(False)

    def _set_pipeline_buttons_enabled(self, enabled: bool) -> None:
        """Backward-compat shim — delegates to _set_running_stage."""
        if enabled:
            self._set_running_stage(None)
        else:
            # Disable all run buttons without showing any ✕
            for btn in (self.seg_run_btn, self.db_run_btn, self.solve_run_btn,
                        self.seg_params_btn, self.db_params_btn, self.solve_params_btn):
                btn.setEnabled(False)

    # ── Threshold / config delegation ─────────────────────────────────────────

    def _current_threshold_pair_from_controls(self):
        return self._tracking_inputs_provider().current_threshold_pair()

    def _threshold_pairs_from_controls(self):
        return self._tracking_inputs_provider().threshold_pairs()

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

    def _update_image_layer(
        self,
        name: str,
        data: np.ndarray,
        *,
        metadata: dict | None = None,
    ) -> None:
        from napari.layers import Image

        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Image):
            layer = self.viewer.layers[name]
            layer.data = data
            layer.metadata = dict(metadata or {})
            return
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        self.viewer.add_image(data, name=name, metadata=dict(metadata or {}))

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
        self._on_preview_threshold_pair()

    def _on_preview_threshold_pair(self) -> None:
        pos_dir = self._pos_dir
        if pos_dir is None:
            self._status("No project open."); return
        paths = self._paths
        if paths is None:
            self._status("No project open."); return
        contours_path = paths.nucleus_contours
        score_path = paths.nucleus_foreground
        if not contours_path.exists():
            self._status(
                "Missing: nucleus_contours.tif — build divergence maps first."
            ); return
        if not score_path.exists():
            self._status(
                "Missing: nucleus_foreground.tif — build divergence maps first."
            ); return
        threshold_pair = self._current_threshold_pair_from_controls()

        cancel_event = threading.Event()
        self._contour_cancel = cancel_event

        def _done(result):
            contour_preview, foreground_preview, metadata = result
            self._contour_worker = None
            self._contour_cancel = None
            self._clear_progress()
            # Core source stacks are P x T x Y x X. Display them as
            # T x P x Y x X so the viewer's leading axis remains time for
            # DB browser and correction actions that read current_step[0].
            contour_display = np.moveaxis(contour_preview, 0, 1)
            foreground_display = np.moveaxis(foreground_preview, 0, 1)
            layer_metadata = {
                "thresholds": metadata,
                "axis_order": ("time", "source", "y", "x"),
            }
            self._update_image_layer(
                _PREVIEW_CONTOURS_LAYER,
                contour_display,
                metadata=layer_metadata,
            )
            self._update_image_layer(
                _PREVIEW_FOREGROUND_LAYER,
                foreground_display,
                metadata=layer_metadata,
            )
            self._status(f"Ultrack threshold preview ready ({len(metadata)} source).")
            self._set_running_stage(None)

        @thread_worker(connect={
            "yielded": self._on_progress,
            "returned": _done,
            "errored": self._on_contour_worker_error,
        })
        def _worker():
            from cellflow.segmentation.nucleus_segmentation import _check_cancel

            yield (0, 3, "Loading Ultrack input maps...")
            contours = np.asarray(tifffile.imread(str(contours_path)), dtype=np.float32)
            _check_cancel(cancel_event.is_set)
            foreground_scores = np.asarray(
                tifffile.imread(str(score_path)),
                dtype=np.float32,
            )
            _check_cancel(cancel_event.is_set)
            yield (1, 3, "Building Ultrack threshold preview...")
            contour_preview, foreground_preview, metadata = (
                build_ultrack_source_stacks_from_pairs(
                    contours,
                    foreground_scores,
                    threshold_pairs=[threshold_pair],
                )
            )
            _check_cancel(cancel_event.is_set)
            yield (3, 3, "Loaded Ultrack threshold preview.")
            return contour_preview, foreground_preview, metadata

        self._status("Building Ultrack threshold preview...")
        self._set_running_stage("seg")
        self._contour_worker = _worker()

    def _on_contour_worker_error(self, exc: Exception) -> None:
        self._contour_worker = None
        self._contour_cancel = None
        self._set_running_stage(None)
        self._clear_progress()
        if isinstance(exc, CancelledError):
            self._status("Cancelled.")
            return
        self._status(f"Error: {exc}")
        logger.exception("Contour worker error", exc_info=exc)

    # ── Pipeline handlers — DB generation ────────────────────────────────────

    def _on_run_db_generation(self) -> None:
        pos_dir = self._pos_dir
        if pos_dir is None:
            self._status("No project open."); return
        paths = self._paths
        if paths is None:
            self._status("No project open."); return
        contours_path = paths.nucleus_contours
        score_path = paths.nucleus_foreground
        if not contours_path.exists():
            self._status(
                "Missing: nucleus_contours.tif — build divergence maps first."
            ); return
        if not score_path.exists():
            self._status(
                "Missing: nucleus_foreground.tif — build divergence maps first."
            ); return
        if not _ultrack_available():
            self._status("ultrack not installed — activate the cellflow conda environment."); return

        threshold_pairs = self._threshold_pairs_from_controls()
        if not threshold_pairs:
            self._status("Add at least one threshold pair before DB generation.")
            return

        cfg = self._db_gen_config_from_controls()
        working_dir = self._ultrack_workdir()

        self.pipeline_progress_bar.setRange(0, 0)
        self.pipeline_progress_bar.setVisible(True)
        self._status("Starting DB generation…")
        self._set_running_stage("db")

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
                    build_ultrack_database_from_threshold_pairs(
                        contours_path=contours_path,
                        foreground_scores_path=score_path,
                        working_dir=working_dir,
                        cfg=cfg,
                        threshold_pairs=threshold_pairs,
                        progress_cb=_progress_cb,
                    )
                    _progress_cb("Scoring node probabilities...")
                    score_signal_path = self._foreground_path()
                    apply_annotations_and_score(
                        working_dir=working_dir,
                        cfg=cfg,
                        score_signal_path=score_signal_path,
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
        self._set_running_stage(None)

    def _on_db_gen_worker_error(self, exc: Exception) -> None:
        self._db_gen_worker = None
        self._set_running_stage(None)
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
        score_path = self._foreground_path()
        if score_path is None or not score_path.exists():
            self._status("Missing: nucleus_foreground.tif — build divergence maps first."); return
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
        self._set_running_stage("ultrack")

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
        if self._running_stage is None:
            return
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
            self._set_running_stage(None)
            self._status("Ultrack tracking failed (no output).")
            return
        if labels.ndim == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        nt = labels.shape[0]
        self._update_tracked_display(labels)
        self._refresh_files_callback(self._pos_dir)
        self._status(f"Tracking done: {nt} frame(s).")
        self._set_running_stage(None)

    def _on_ultrack_worker_error(self, exc: Exception) -> None:
        self._ultrack_worker = None
        self._set_running_stage(None)
        self._clear_progress()
        self._status(f"Error: {exc}")
        logger.exception("Ultrack worker error", exc_info=exc)

    # ── Cancel ────────────────────────────────────────────────────────────────

    def _on_cancel(self) -> None:
        cancelled = False
        if self._contour_cancel is not None:
            self._contour_cancel.set()
        for attr in ("_contour_worker", "_db_gen_worker", "_ultrack_worker"):
            worker = getattr(self, attr, None)
            if worker is not None:
                worker.quit()
                setattr(self, attr, None)
                cancelled = True
        self._contour_cancel = None
        self._set_running_stage(None)
        self._clear_progress()
        self._status("Cancelled." if cancelled else "Nothing running.")
