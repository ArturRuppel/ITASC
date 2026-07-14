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

from itasc.core.tiff import imwrite_grayscale
from itasc.napari.correction._correction_utils import reorder_stack_by_quality
from itasc.napari._paths import NucleusWorkspace
from itasc.napari._widget_helpers import (
    make_progress as _make_progress,
    make_status as _make_status,
    tool_btn as _tool_btn,
)
from itasc.napari.ui_style import (
    stage_header_action_button as _stage_header_action_button,
    stage_header_label as _stage_header_label,
)
from itasc.tracking_ultrack.validation_state import read_corrections, read_validated_tracks
from itasc.core.cancellation import CancelledError
from itasc.tracking_ultrack.db_build import (
    apply_annotations_and_score,
    build_atom_union_database,
)
from itasc.tracking_ultrack.corrections import corrections_from_validated_tracks
from itasc.tracking_ultrack.export import export_tracked_labels
from itasc.tracking_ultrack.solve import run_solve
from itasc.tracking_ultrack.track_quality import track_quality_scores

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


class NucleusPipelineWidget(QWidget):
    """Action buttons, workers, and coordination handlers for the nucleus pipeline."""

    def __init__(
        self,
        viewer: napari.Viewer,
        *,
        workspace_provider: Callable[[], NucleusWorkspace | None],
        seg_inputs_provider: Callable,
        tracking_inputs_provider: Callable,
        refresh_files_callback: Callable[[], None],
        refresh_db_browser_callback: Callable[[], None],
        sync_viewer_activity_callback: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._workspace_provider = workspace_provider
        self._seg_inputs_provider = seg_inputs_provider
        self._tracking_inputs_provider = tracking_inputs_provider
        self._refresh_files_callback = refresh_files_callback
        self._refresh_db_browser_callback = refresh_db_browser_callback
        self._sync_viewer_activity_callback = sync_viewer_activity_callback

        self._db_gen_worker = None
        self._ultrack_worker = None
        self._db_gen_cancel: threading.Event | None = None
        self._running_stage: str | None = None

        # ── Per-stage buttons ──────────────────────────────────────────
        self.seg_params_btn = _tool_btn("⚙", "Show parameters for this stage.", checkable=True)
        self.seg_run_btn = _tool_btn("▶", "Run segmentation inputs.")

        self.db_params_btn = _tool_btn("⚙", "Show parameters for this stage.", checkable=True)
        self.db_run_btn = _tool_btn("▶", "Run Ultrack database build.")

        self.solve_params_btn = _tool_btn("⚙", "Show parameters for this stage.", checkable=True)
        self.solve_run_btn = _tool_btn("▶", "Run Ultrack solve.")

        for button in (
            self.seg_params_btn,
            self.seg_run_btn,
            self.db_params_btn,
            self.db_run_btn,
            self.solve_params_btn,
            self.solve_run_btn,
        ):
            _stage_header_action_button(button, "nucleus")

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
            for w in trailing:
                row.addWidget(w)
            row.addStretch(1)
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
    def _workspace(self) -> NucleusWorkspace | None:
        return self._workspace_provider()

    @property
    def _pos_dir(self) -> Path | None:
        """The nucleus annotation/store directory (validation JSONs live here)."""
        ws = self._workspace
        return ws.nucleus_dir if ws is not None else None

    def _contours_path(self) -> Path | None:
        ws = self._workspace
        return ws.contours if ws is not None else None

    def _foreground_path(self) -> Path | None:
        ws = self._workspace
        return ws.foreground if ws is not None else None

    def _ultrack_workdir(self) -> Path | None:
        ws = self._workspace
        return ws.ultrack_workdir if ws is not None else None

    def _ultrack_db_path(self) -> Path | None:
        ws = self._workspace
        return ws.ultrack_db if ws is not None else None

    def _tracked_path(self) -> Path | None:
        ws = self._workspace
        return ws.tracked if ws is not None else None

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
        """Swap the run buttons' ▶/✕ affordance for the active stage.

        ``None`` means idle (all ▶); ``"seg" | "db" | "ultrack"`` shows ✕ on
        that row. Button *enablement* is owned by the UI gate — this only
        updates the cancel/run glyph and tooltip, then notifies the gate to
        recompute (the gate's predicates read ``self._running_stage``).
        """
        self._running_stage = stage_key
        if stage_key is None:
            self.seg_run_btn.setText("▶")
            self.seg_run_btn.setToolTip("Run segmentation inputs.")
            self.db_run_btn.setText("▶")
            self.db_run_btn.setToolTip("Run Ultrack database build.")
            self.solve_run_btn.setText("▶")
            self.solve_run_btn.setToolTip("Run Ultrack solve.")
        else:
            run_btn = {
                "seg": self.seg_run_btn,
                "db": self.db_run_btn,
                "ultrack": self.solve_run_btn,
            }[stage_key]
            run_btn.setText("✕")
            run_btn.setToolTip("Cancel.")
        if self._sync_viewer_activity_callback is not None:
            self._sync_viewer_activity_callback()

    # ── Config delegation ─────────────────────────────────────────────────────

    def _db_gen_config_from_controls(self):
        return self._tracking_inputs_provider().db_gen_config()

    def _ultrack_config_from_controls(self):
        return self._tracking_inputs_provider().ultrack_config()

    # ── Viewer helpers ────────────────────────────────────────────────────────

    def _update_labels_layer(
        self,
        name: str,
        data: np.ndarray,
        *,
        metadata: dict | None = None,
    ) -> None:
        from napari.layers import Labels
        if name in self.viewer.layers and isinstance(self.viewer.layers[name], Labels):
            layer = self.viewer.layers[name]
            layer.data = data
            layer.metadata = dict(metadata or {})
            return
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        self.viewer.add_labels(data, name=name, metadata=dict(metadata or {}))

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
        # Cellpose maps and atom extraction now produce the candidate inputs;
        # there is no separate source-threshold build step here.
        self._status("Segmentation inputs are produced by Atom Extraction.")

    # ── Pipeline handlers — DB generation ────────────────────────────────────

    def _atoms_path(self) -> Path | None:
        ws = self._workspace
        return ws.atoms if ws is not None else None

    def _on_run_db_generation(self) -> None:
        pos_dir = self._pos_dir
        if pos_dir is None:
            self._status("No project open."); return
        atoms_path = self._atoms_path()
        if atoms_path is None or not atoms_path.exists():
            self._status("Missing: atoms.tif — run Atom Extraction first.")
            return
        if not _ultrack_available():
            self._status("ultrack not installed — activate the itasc conda environment."); return

        cfg = self._db_gen_config_from_controls()
        working_dir = self._ultrack_workdir()
        foreground_path = self._foreground_path()
        contour_path = self._contours_path()

        self.pipeline_progress_bar.setRange(0, 0)
        self.pipeline_progress_bar.setVisible(True)
        self._status("Starting DB generation…")
        self._set_running_stage("db")
        cancel_event = threading.Event()
        self._db_gen_cancel = cancel_event

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
                    report = build_atom_union_database(
                        atoms_path,
                        working_dir,
                        cfg,
                        _progress_cb,
                        contour_maps_path=contour_path,
                        cancel=cancel_event.is_set,
                    )
                    if cancel_event.is_set():
                        raise CancelledError("Operation cancelled.")
                    if foreground_path is not None and foreground_path.exists():
                        _progress_cb("Scoring node probabilities...")
                        apply_annotations_and_score(
                            working_dir=working_dir,
                            cfg=cfg,
                            score_signal_path=foreground_path,
                            corrections=None,
                            validated_tracks=None,
                            tracked_labels=None,
                        )
                    if cancel_event.is_set():
                        raise CancelledError("Operation cancelled.")
                    result_holder.append(report)
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
            return (pos_dir, result_holder[0] if result_holder else None)

        self._db_gen_worker = _worker()

    def _on_db_gen_done(self, result) -> None:
        self._db_gen_worker = None
        self._db_gen_cancel = None
        self._clear_progress()
        pos_dir, _ = result
        self._status("DB generation complete.")
        self._refresh_files_callback()
        self._refresh_db_browser_callback()
        self._set_running_stage(None)

    def _on_db_gen_worker_error(self, exc: Exception) -> None:
        self._db_gen_worker = None
        self._db_gen_cancel = None
        self._set_running_stage(None)
        self._clear_progress()
        if isinstance(exc, CancelledError):
            self._status("Cancelled.")
            return
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
        working_dir = self._ultrack_workdir()
        tracked_path = self._tracked_path()

        cfg = self._ultrack_config_from_controls()
        corrections = read_corrections(pos_dir)
        validated_tracks = read_validated_tracks(pos_dir) or None
        tracked_labels = None
        if corrections or validated_tracks:
            tracked_labels = self._ensure_tracked_layer_data()
            if tracked_labels is None:
                self._status(
                    "Validated-aware export requires tracked_labels.tif "
                    "(layer not loaded and file not on disk)."
                ); return
        if corrections and validated_tracks and tracked_labels is not None:
            existing = {
                (int(c.cell_id), int(c.t))
                for c in corrections
                if getattr(c, "kind", None) == "validated"
            }
            corrections = list(corrections) + [
                c for c in corrections_from_validated_tracks(validated_tracks, tracked_labels)
                if (int(c.cell_id), int(c.t)) not in existing
            ]
            validated_tracks = None

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
            for step, total, label in run_solve(working_dir, cfg, overwrite=True):
                yield (step, total, f"[solve] {label}")
            yield "Exporting tracked labels…"
            labels = export_tracked_labels(
                working_dir, cfg, tracked_path,
                corrections=corrections,
                validated_tracks=validated_tracks,
                tracked_labels=tracked_labels,
            )
            yield "Ordering track IDs by quality…"
            try:
                scores = track_quality_scores(db_path, cfg)
            except Exception:
                logger.exception("Quality scoring failed; leaving track IDs unordered.")
                scores = {}
            if scores:
                relabeled, old_to_new = reorder_stack_by_quality(labels, scores, pos_dir)
                if old_to_new:
                    labels = relabeled
                    imwrite_grayscale(tracked_path, labels, compression="zlib")
            return labels

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
        self._refresh_files_callback()
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
        db_worker = self._db_gen_worker
        if db_worker is not None:
            if self._db_gen_cancel is not None:
                self._db_gen_cancel.set()
                self._status(
                    "Cancelling DB generation after the current frame..."
                )
                return
            db_worker.quit()
            self._db_gen_worker = None
            cancelled = True
        ultrack_worker = self._ultrack_worker
        if ultrack_worker is not None:
            ultrack_worker.quit()
            self._ultrack_worker = None
            cancelled = True
        self._set_running_stage(None)
        self._clear_progress()
        self._status("Cancelled." if cancelled else "Nothing running.")
