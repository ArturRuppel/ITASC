"""Contact Analysis widget: per-position quantity view (currently contacts)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tifffile

import napari
from napari.qt.threading import thread_worker
from qtpy.QtCore import QObject, QSettings, QTimer, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from itasc.contact_analysis import (
    build_contacts,  # noqa: F401 - re-exported for tests that build directly
    discover_contact_batch_jobs,
    ensure_contacts,
    run_contact_batch,
)
from itasc.contact_analysis.catalog import CONTACT_ANALYSIS_RELPATH
from itasc.napari.ui_gate import ControlClass, UiGate
from itasc.napari.ui_style import action_button, status_label
from itasc.napari.widgets import (
    CollapsibleSection,
    PipelineFilesWidget,
    make_pipeline_files_header,
)

try:  # pragma: no cover - local branch compatibility
    from itasc.contact_analysis.contacts.reader import read_position_contacts
except ImportError:  # pragma: no cover - tests monkeypatch this when absent
    def read_position_contacts(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError("itasc.contact_analysis.contacts.reader is unavailable")


try:  # pragma: no cover - local branch compatibility
    from itasc.napari.contact_visualization import (
        add_contact_analysis_layers,
        _nucleus_centroids_by_track,
    )
except ImportError:  # pragma: no cover - tests monkeypatch this when absent
    def add_contact_analysis_layers(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError("itasc.napari.contact_visualization is unavailable")

    def _nucleus_centroids_by_track(*_args, **_kwargs):  # type: ignore[no-redef]
        raise ImportError("itasc.napari.contact_visualization is unavailable")


def make_contact_analysis_widget(napari_viewer=None):
    """napari plugin entry point: the per-position Contact Analysis dock widget.

    Returns the bare per-position :class:`ContactAnalysisWidget` in standalone
    mode (own file pickers + config). The cross-position aggregate role now lives
    in the full app's Aggregate capstone (``itasc.napari.aggregate_widget``),
    so there is no separate interactive studio to serve here. Runs the napari
    layer-delegate patch (normally done by the orchestrator package).
    """
    try:
        from itasc.napari._napari_compat import patch_napari_layer_delegate

        patch_napari_layer_delegate()
    except Exception:  # pragma: no cover - patch is best-effort
        pass
    if napari_viewer is None:
        napari_viewer = napari.current_viewer()
    return ContactAnalysisWidget(viewer=napari_viewer, standalone=True)


class _ProgressEmitter(QObject):
    progress = Signal(int, int, str)


class ContactAnalysisWidget(QWidget):
    """Final contact analysis and export."""

    _contact_analysis_layer_prefix = "[Contact Analysis] "

    #: QSettings key used to persist display options in standalone mode.
    _SETTINGS_KEY = "itasc_contact/state"

    def __init__(
        self,
        viewer: object | None = None,
        parent: QWidget | None = None,
        gate: UiGate | None = None,
        standalone: bool = False,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        #: App-wide UI gate; a private one is created for standalone use.
        self.gate = gate if gate is not None else UiGate(self)
        #: When standalone, the widget owns its own file pickers + config and
        #: hides the orchestrator's staged "Pipeline Files" panel. When
        #: orchestrated, the parent injects paths via :meth:`set_context`.
        self._standalone = standalone
        #: Explicit working context (set via :meth:`set_context` or the pickers).
        self._cell_labels_path: Path | None = None
        self._nucleus_labels_path: Path | None = None
        self._out_path: Path | None = None
        #: The position dir driving the Pipeline Files panel (orchestrated only).
        self._status_root: Path | None = None
        self._build_worker = None
        self._batch_worker = None
        self._build_completion_pending = False
        self._build_error_pending = False
        self._progress_emitter = _ProgressEmitter(self)
        self._progress_emitter.progress.connect(self._on_build_progress)
        self._batch_progress_emitter = _ProgressEmitter(self)
        self._batch_progress_emitter.progress.connect(self._on_batch_progress)
        self._batch_completion_pending = False
        self._batch_cancel = False
        self._cached_contact_analysis_path: Path | None = None
        self._cached_contact_analysis: Any = None
        self._cached_cell_labels: np.ndarray | None = None
        self._cached_nucleus_labels: np.ndarray | None = None
        self._cached_track_centroids: dict | None = None
        #: (path, *display-option flags) of the contact-analysis overlay currently
        #: on screen, so a re-Show of the same position+options can skip the
        #: remove/re-add churn (see ``_show_from_disk``). ``None`` when nothing shown.
        self._displayed_contact_analysis_signature: tuple | None = None
        #: Pending deferred overlay-add timer, cancelled if a new Show supersedes it.
        self._pending_show_timer: QTimer | None = None
        #: Positions discovered under the standalone top folder (row index ↔ job).
        self._discovered_jobs: list = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        self._files_widget = PipelineFilesWidget(
            [
                ("Inputs", [
                    ("nucleus_labels.tif", "Nucleus labels (committed)"),
                    ("cell_labels.tif", "Cell labels (committed)"),
                ]),
                ("Output", [
                    (CONTACT_ANALYSIS_RELPATH, "Contact analysis"),
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
        # The staged "Pipeline Files" panel is an orchestrator concept; standalone
        # use replaces it with the discovery panel below.
        self.pipeline_files_header.setVisible(not self._standalone)
        self._pipeline_files_section.setVisible(not self._standalone)
        layout.addWidget(self.pipeline_files_header)
        layout.addWidget(self._pipeline_files_section)

        # Standalone: enter a top folder + file names → discovered positions list.
        self._build_discovery_section(layout)

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

        # Display options, laid out as a two-column grid of toggles.
        self.color_edges_by_id_cb = QCheckBox("Color edges by ID")
        self.color_edges_by_label_cb = QCheckBox("Color edges by label")
        self.hide_border_edges_cb = QCheckBox("Hide border edges")
        options_grid = QGridLayout()
        options_grid.setContentsMargins(0, 0, 0, 0)
        options_grid.setHorizontalSpacing(12)
        options_grid.addWidget(self.color_edges_by_id_cb, 0, 0)
        options_grid.addWidget(self.color_edges_by_label_cb, 0, 1)
        options_grid.addWidget(self.hide_border_edges_cb, 1, 0)
        layout.addLayout(options_grid)

        # Two build/show actions sit directly below the display options (no
        # trailing stretch, so the panel hugs its content). Visualize computes the
        # .h5 on demand only if it is missing, then shows the overlays; Recompute
        # forces a rebuild before showing, for when the .h5 is stale relative to
        # its inputs.
        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(6)
        self.recompute_btn = QPushButton("Run Contact Analysis")
        self.recompute_btn.setToolTip(
            "Rebuild the contact-analysis .h5 from the current inputs, then show "
            "the overlays. Use this when the result is stale."
        )
        action_button(self.recompute_btn, expand=True)
        actions_row.addWidget(self.recompute_btn, 1)

        self.visualize_btn = QPushButton("Visualize Contact Analysis")
        self.visualize_btn.setToolTip(
            "Show contact-analysis overlays for the current position. "
            "If the analysis has not been computed yet, it is computed first; "
            "an existing result is shown as-is."
        )
        action_button(self.visualize_btn, expand=True)
        actions_row.addWidget(self.visualize_btn, 1)
        layout.addLayout(actions_row)

        self.visualize_btn.clicked.connect(lambda: self._on_visualize(overwrite=False))
        self.recompute_btn.clicked.connect(lambda: self._on_visualize(overwrite=True))
        self._register_gate_controls()
        if self._standalone:
            self._load_standalone_settings()
        self._update_status()

    # --------------------------------------------------------------- discovery UI
    def _make_picker_row(
        self, layout, label: str, on_browse, *, read_only: bool = True
    ) -> QLineEdit:
        """Add a ``label / line-edit / Browse`` row to *layout*; return the edit."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        lbl = QLabel(label)
        lbl.setFixedWidth(150)
        edit = QLineEdit()
        edit.setReadOnly(read_only)
        browse = QPushButton("Browse...")
        action_button(browse)
        browse.clicked.connect(on_browse)
        row.addWidget(lbl)
        row.addWidget(edit, 1)
        row.addWidget(browse)
        layout.addLayout(row)
        return edit

    def _build_discovery_section(self, layout) -> None:
        """Standalone discovery panel: a top folder + file names → a list of
        discovered positions to pick from. Hidden when embedded in the
        orchestrator (which drives the widget per-position via ``set_context``).
        """
        container = QWidget()
        col = QVBoxLayout(container)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)

        self._batch_root_edit = self._make_picker_row(
            col, "Top folder:", self._on_browse_batch_root, read_only=False
        )
        self._batch_root_edit.editingFinished.connect(self._rediscover)
        self._batch_cell_name_edit = self._make_name_row(
            col, "Cell labels name:", "cell_labels.tif"
        )
        self._batch_nucleus_name_edit = self._make_name_row(
            col, "Nucleus name (optional):", "nucleus_labels.tif"
        )
        self._batch_h5_name_edit = self._make_name_row(
            col, "Output .h5 name:", "contact_analysis.h5"
        )
        for edit in (
            self._batch_cell_name_edit,
            self._batch_nucleus_name_edit,
            self._batch_h5_name_edit,
        ):
            edit.editingFinished.connect(self._rediscover)

        self._discovery_list = QListWidget()
        self._discovery_list.itemSelectionChanged.connect(self._on_job_selected)
        self._discovery_list.itemDoubleClicked.connect(self._on_job_activated)
        col.addWidget(self._discovery_list, 1)

        self.batch_overwrite_cb = QCheckBox("Overwrite existing")
        col.addWidget(self.batch_overwrite_cb)

        self.run_batch_btn = QPushButton("Process all")
        action_button(self.run_batch_btn, expand=True)
        col.addWidget(self.run_batch_btn)

        self.cancel_batch_btn = QPushButton("Cancel batch")
        action_button(self.cancel_batch_btn)
        self.cancel_batch_btn.setEnabled(False)
        col.addWidget(self.cancel_batch_btn)

        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setVisible(False)
        self.batch_progress_bar.setTextVisible(True)
        col.addWidget(self.batch_progress_bar)

        self.batch_status_lbl = QLabel("")
        self.batch_status_lbl.setWordWrap(True)
        status_label(self.batch_status_lbl)
        col.addWidget(self.batch_status_lbl)

        self._discovery_container = container
        container.setVisible(self._standalone)
        layout.addWidget(container)

        self.run_batch_btn.clicked.connect(self._on_run_batch)
        self.cancel_batch_btn.clicked.connect(self._on_cancel_batch)

    def _make_name_row(self, layout, label: str, default: str) -> QLineEdit:
        """Add a ``label / editable filename`` row pre-filled with *default*."""
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        lbl = QLabel(label)
        lbl.setFixedWidth(150)
        edit = QLineEdit()
        edit.setText(default)
        row.addWidget(lbl)
        row.addWidget(edit, 1)
        layout.addLayout(row)
        return edit

    # ------------------------------------------------------------ discovery logic
    def _discovery_fields(self) -> tuple[str, str, str, str | None]:
        return (
            self._batch_root_edit.text().strip(),
            self._batch_cell_name_edit.text().strip(),
            self._batch_h5_name_edit.text().strip(),
            self._batch_nucleus_name_edit.text().strip() or None,
        )

    def _job_label(self, job) -> str:
        kind = "cell+nucleus" if job.nucleus_labels else "cell only"
        built = "built" if job.output.exists() else "missing"
        return f"{job.group_dir.name}    {kind}    [{built}]"

    def _populate_discovery_list(self) -> None:
        self._discovery_list.clear()
        for job in self._discovered_jobs:
            self._discovery_list.addItem(self._job_label(job))

    def _refresh_discovery_status(self) -> None:
        """Re-evaluate built/missing badges (outputs may have appeared)."""
        for row, job in enumerate(self._discovered_jobs):
            item = self._discovery_list.item(row)
            if item is not None:
                item.setText(self._job_label(job))

    def _rediscover(self) -> None:
        """Re-scan the top folder and repopulate the discovered-positions list."""
        root, cell_name, h5_name, nucleus_name = self._discovery_fields()
        self._discovered_jobs = []
        self._discovery_list.clear()
        if not (root and cell_name and h5_name):
            self.batch_status_lbl.setText(
                "Enter a top folder and file names to discover positions."
            )
            return
        try:
            jobs = discover_contact_batch_jobs(
                root, cell_name=cell_name, h5_name=h5_name, nucleus_name=nucleus_name
            )
        except Exception as exc:
            self.batch_status_lbl.setText(f"Discovery error: {exc}")
            return
        self._discovered_jobs = jobs
        self._populate_discovery_list()
        self.batch_status_lbl.setText(
            f"Discovered {len(jobs)} position(s); double-click one to visualize."
            if jobs
            else f"No '{cell_name}' files found under {root}."
        )
        self._save_standalone_settings()

    def _selected_job(self):
        row = self._discovery_list.currentRow()
        if 0 <= row < len(self._discovered_jobs):
            return self._discovered_jobs[row]
        return None

    def _on_job_selected(self) -> None:
        job = self._selected_job()
        if job is None:
            return
        # Target the selected position so Visualize/Recompute act on it.
        self.set_context(
            cell_labels=job.cell_labels,
            nucleus_labels=job.nucleus_labels,
            out_path=job.output,
        )

    def _on_job_activated(self, item) -> None:
        row = self._discovery_list.row(item)
        if not (0 <= row < len(self._discovered_jobs)):
            return
        job = self._discovered_jobs[row]
        self.set_context(
            cell_labels=job.cell_labels,
            nucleus_labels=job.nucleus_labels,
            out_path=job.output,
        )
        self._on_visualize(overwrite=False)

    # ------------------------------------------------------------------- config
    def get_state(self) -> dict:
        """Serialize display options (the seam shared by orchestrator + standalone)."""
        return {
            "color_edges_by_id": self.color_edges_by_id_cb.isChecked(),
            "color_edges_by_label": self.color_edges_by_label_cb.isChecked(),
            "hide_border_edges": self.hide_border_edges_cb.isChecked(),
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        if "color_edges_by_id" in state:
            self.color_edges_by_id_cb.setChecked(bool(state["color_edges_by_id"]))
        if "color_edges_by_label" in state:
            self.color_edges_by_label_cb.setChecked(bool(state["color_edges_by_label"]))
        if "hide_border_edges" in state:
            self.hide_border_edges_cb.setChecked(bool(state["hide_border_edges"]))

    def _load_standalone_settings(self) -> None:
        s = QSettings()
        raw = s.value(self._SETTINGS_KEY)
        if isinstance(raw, dict):
            self.set_state(raw)
        cell = s.value("itasc_contact/cell_name", "", type=str)
        nucleus = s.value("itasc_contact/nucleus_name", "", type=str)
        h5 = s.value("itasc_contact/h5_name", "", type=str)
        root = s.value("itasc_contact/root", "", type=str)
        if cell:
            self._batch_cell_name_edit.setText(cell)
        if nucleus:
            self._batch_nucleus_name_edit.setText(nucleus)
        if h5:
            self._batch_h5_name_edit.setText(h5)
        if root:
            self._batch_root_edit.setText(root)
        self._rediscover()

    def _save_standalone_settings(self) -> None:
        if not self._standalone:
            return
        s = QSettings()
        s.setValue(self._SETTINGS_KEY, self.get_state())
        root, cell_name, h5_name, nucleus_name = self._discovery_fields()
        s.setValue("itasc_contact/root", root)
        s.setValue("itasc_contact/cell_name", cell_name)
        s.setValue("itasc_contact/nucleus_name", nucleus_name or "")
        s.setValue("itasc_contact/h5_name", h5_name)

    def _register_gate_controls(self) -> None:
        """Register contact-analysis actions with the app-wide UI gate.

        Build/cancel are headless (disk only), so they run regardless of viewer
        ownership. Visualize/Recompute write the viewer, so they are blocked
        while a viewer owner (correction / live preview) is active.
        """
        g = self.gate
        running = lambda: self._build_worker is not None
        batch_running = lambda: self._batch_worker is not None
        # Visualize/Recompute write the viewer (they show overlays) and build
        # first; both need inputs and no in-flight build. Recompute additionally
        # forces a rebuild, but its enablement gates the same way.
        g.register(
            self.visualize_btn,
            ControlClass.RUN_VIEWER,
            when=lambda: self._inputs_ready() and not running(),
        )
        g.register(
            self.recompute_btn,
            ControlClass.RUN_VIEWER,
            when=lambda: self._inputs_ready() and not running(),
        )
        # Batch is headless (disk only); it runs regardless of viewer ownership.
        g.register(
            self.run_batch_btn, ControlClass.RUN_HEADLESS, when=lambda: not batch_running()
        )
        g.register(self.cancel_batch_btn, ControlClass.RUN_HEADLESS, when=batch_running)
        g.recompute()

    @property
    def cell_labels_path(self) -> Path | None:
        return self._cell_labels_path

    @property
    def nucleus_labels_path(self) -> Path | None:
        return self._nucleus_labels_path

    @property
    def contact_analysis_out_path(self) -> Path | None:
        return self._out_path

    def set_context(
        self,
        *,
        cell_labels: Path | str | None,
        nucleus_labels: Path | str | None = None,
        out_path: Path | str | None = None,
        status_root: Path | str | None = None,
    ) -> None:
        """Set the working context for both orchestrated and standalone use.

        The orchestrator supplies explicit staged paths plus ``status_root``
        (the position directory) to drive the "Pipeline Files" panel. Standalone
        use targets a position by selecting it in the discovered-positions list,
        which calls this without ``status_root``.
        """
        cell = Path(cell_labels) if cell_labels else None
        nucleus = Path(nucleus_labels) if nucleus_labels else None
        out = Path(out_path) if out_path else None
        # The .h5 is a derived artifact; when no explicit output is given, default
        # it next to the cell-labels file. Orchestrated callers always pass an
        # explicit out_path, so this only fires for standalone selections.
        if out is None and cell is not None:
            out = cell.parent / "contact_analysis.h5"
        if (cell, nucleus, out) != (
            self._cell_labels_path,
            self._nucleus_labels_path,
            self._out_path,
        ):
            self._invalidate_caches()
        self._cell_labels_path = cell
        self._nucleus_labels_path = nucleus
        self._out_path = out
        self._status_root = Path(status_root) if status_root else None
        if not self._standalone:
            self._files_widget.refresh(self._status_root)
        self._update_status()

    def _refresh_files_tracker(self) -> None:
        """Re-read the Pipeline Files panel after a run wrote a new ``.h5``.

        This fires the tracker's ``refreshed`` signal, which is how the host
        repaints the stage dots and catalog rail without a manual Refresh.
        Standalone use has no such panel, so it is a no-op there.
        """
        if not self._standalone:
            self._files_widget.refresh(self._status_root)

    def _invalidate_caches(self) -> None:
        self._cached_contact_analysis_path = None
        self._cached_contact_analysis = None
        self._cached_cell_labels = None
        self._cached_nucleus_labels = None
        self._cached_track_centroids = None

    def _update_status(self) -> None:
        self._update_action_states()
        if self._cell_labels_path is None:
            self._set_contact_analysis_status(
                "Status: pick a position from the list."
                if self._standalone
                else "Status: no project open."
            )
        elif not self.contact_analysis_status_lbl.text():
            self._set_contact_analysis_status("Status: ready.")

    def _effective_nucleus_path(self) -> Path | None:
        """Nucleus labels to use, or ``None`` if absent.

        The orchestrator always wires a nucleus path, but the file may not exist
        yet (e.g. only cell labels have been produced). Nucleus labels are
        optional for contact analysis, so a missing file is treated as "no
        nucleus" rather than a blocking error.
        """
        nucleus = self._nucleus_labels_path
        if nucleus is not None and nucleus.exists():
            return nucleus
        return None

    def _inputs_ready(self) -> bool:
        # Cell labels are the only required input; nucleus is optional, so a
        # set-but-missing nucleus path must not gate Visualize.
        cell = self._cell_labels_path
        if cell is None or not cell.exists():
            return False
        return self._out_path is not None

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

    def _on_compute_done(self, result: tuple[Path, bool]) -> None:
        output_path, built = result
        self._build_completion_pending = True
        self._build_worker = None
        self._set_build_running(False)
        self._set_contact_analysis_status(
            f"Status: {'computed' if built else 'using existing'} contact analysis "
            f"({output_path.name})."
        )
        # The .h5 cache is keyed by path, but a build rewrites the file in place
        # (same path), so a Recompute would otherwise serve the stale cached read.
        # Drop just the cached analysis so _show_from_disk re-reads the fresh file;
        # the label TIFFs are inputs the build does not touch, so they stay cached.
        if built:
            self._cached_contact_analysis = None
            # A rebuild overwrites the .h5 in place, so the display signature
            # (path + display options) is unchanged and _show_from_disk's
            # "already shown" fast path would skip re-adding the fresh overlays,
            # leaving stale edges/tracks/labels on screen. Invalidate the shown
            # signature so the fresh data is actually re-rendered.
            self._displayed_contact_analysis_signature = None
        self._update_status()
        self._refresh_discovery_status()
        self._refresh_files_tracker()
        self._show_from_disk()

    def _on_build_error(self, exc: Exception) -> None:
        self._build_error_pending = True
        self._build_worker = None
        self._set_build_running(False)
        self._set_contact_analysis_status(f"Status: error: {exc}")
        self._update_status()

    def _on_visualize(self, *, overwrite: bool = False) -> None:
        """Visualize the contact analysis, computing the .h5 only if needed.

        With ``overwrite=False`` (Visualize) an existing .h5 is shown as-is and a
        missing one is built first. With ``overwrite=True`` (Recompute) the .h5 is
        always rebuilt. The build runs in a worker; the show happens afterwards.
        """
        cell = self._cell_labels_path
        out = self._out_path
        if cell is None or out is None:
            self._set_contact_analysis_status(
                "Status: pick a position from the list."
                if self._standalone
                else "Status: no project open."
            )
            self._update_action_states()
            return
        if not cell.exists():
            self._set_contact_analysis_status(f"Status: missing cell labels: {cell}")
            self._update_status()
            return
        # Nucleus is optional: if its file is not present yet, compute contacts
        # from cell labels alone instead of refusing to run.
        nucleus = self._effective_nucleus_path()

        # Fast path: the artifact already exists and no rebuild was requested, so
        # skip the worker and show immediately. ensure_contacts remains the
        # authority on the missing-only policy for the build path below.
        if out.exists() and not overwrite:
            self._set_contact_analysis_status(
                f"Status: showing existing contact analysis ({out.name})."
            )
            self._show_from_disk()
            return

        self._build_completion_pending = False
        self._build_error_pending = False
        self._set_contact_analysis_status(
            "Status: computing contact analysis (not present yet)..."
            if not overwrite
            else "Status: recomputing contact analysis..."
        )
        self._set_build_running(True)

        @thread_worker(
            connect={
                "returned": self._on_compute_done,
                "errored": self._on_build_error,
            }
        )
        def _worker():
            return ensure_contacts(
                cell_labels_path=cell,
                output_path=out,
                nucleus_labels_path=nucleus,
                overwrite=overwrite,
                progress_cb=self._progress_emitter.progress.emit,
            )

        worker = _worker()
        self._build_worker = worker
        if self._build_completion_pending or self._build_error_pending:
            self._build_worker = None
            self._build_completion_pending = False
            self._build_error_pending = False
            self._update_action_states()

    # ------------------------------------------------------------------- batch
    def _on_browse_batch_root(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select top-level folder")
        if path:
            self._batch_root_edit.setText(path)
            self._rediscover()

    def _on_batch_progress(self, done: int, total: int, label: str) -> None:
        if total > 0:
            self.batch_progress_bar.setRange(0, total)
            self.batch_progress_bar.setValue(done)
        self.batch_status_lbl.setText(f"Batch: {done}/{total} {label}")

    def _on_batch_done(self, results: list) -> None:
        self._batch_completion_pending = True
        self._batch_worker = None
        self.batch_progress_bar.setVisible(False)
        built = sum(1 for r in results if r.status == "built")
        skipped = sum(1 for r in results if r.status == "skipped")
        failed = sum(1 for r in results if r.status == "failed")
        self.batch_status_lbl.setText(
            f"Processed all: built {built} / skipped {skipped} / failed {failed}"
        )
        self._refresh_discovery_status()
        self._refresh_files_tracker()
        self._update_action_states()

    def _on_batch_error(self, exc: Exception) -> None:
        self._batch_completion_pending = True
        self._batch_worker = None
        self.batch_progress_bar.setVisible(False)
        self.batch_status_lbl.setText(f"Batch error: {exc}")
        self._update_action_states()

    def _on_run_batch(self) -> None:
        # Re-scan so the run matches the list the user is looking at.
        self._rediscover()
        jobs = self._discovered_jobs
        if not jobs:
            return

        overwrite = self.batch_overwrite_cb.isChecked()
        self._batch_cancel = False
        self._batch_completion_pending = False
        self.batch_progress_bar.setRange(0, len(jobs))
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setVisible(True)
        self.batch_status_lbl.setText(f"Batch: {len(jobs)} positions...")

        @thread_worker(
            connect={
                "returned": self._on_batch_done,
                "errored": self._on_batch_error,
            }
        )
        def _worker():
            return run_contact_batch(
                jobs,
                overwrite=overwrite,
                progress_cb=self._batch_progress_emitter.progress.emit,
                cancel=lambda: self._batch_cancel,
            )

        worker = _worker()
        self._batch_worker = worker
        if self._batch_completion_pending:
            self._batch_worker = None
            self._batch_completion_pending = False
        self._update_action_states()

    def _on_cancel_batch(self) -> None:
        self._batch_cancel = True
        worker = self._batch_worker
        if worker is not None:
            self._batch_worker = None
            worker.quit()
        self.batch_progress_bar.setVisible(False)
        self.batch_status_lbl.setText("Batch: cancelled.")
        self._update_action_states()

    def _show_from_disk(self) -> None:
        if self.viewer is None:
            self._set_contact_analysis_status("Status: no viewer available.")
            self._update_action_states()
            return
        contact_analysis_path = self.contact_analysis_out_path
        if contact_analysis_path is None or not contact_analysis_path.exists():
            self._set_contact_analysis_status("Status: contact analysis file not found.")
            self._update_action_states()
            return

        # Cache contact analysis to avoid re-reading HDF5 on every Show click.
        # Re-read when there is no cached analysis (first show, context change, or
        # a fresh build dropped it) or when the output path changed. A path change
        # means a different position, so its label/centroid caches are stale too; a
        # same-path rebuild leaves those inputs untouched, so they stay cached.
        if (
            self._cached_contact_analysis is None
            or self._cached_contact_analysis_path != contact_analysis_path
        ):
            if self._cached_contact_analysis_path != contact_analysis_path:
                self._cached_cell_labels = None
                self._cached_nucleus_labels = None
                self._cached_track_centroids = None
            self._cached_contact_analysis = read_position_contacts(contact_analysis_path)
            self._cached_contact_analysis_path = contact_analysis_path

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

        options: dict[str, Any] = {
            "color_edges_by_id": self.color_edges_by_id_cb.isChecked(),
            "color_edges_by_label": self.color_edges_by_label_cb.isChecked(),
            "hide_border_edges": self.hide_border_edges_cb.isChecked(),
        }
        signature = (contact_analysis_path, *options.values())

        # Skip the remove/re-add when the same position is already shown with the
        # same display options and its layers are still present. Re-adding is pure
        # churn here: it flickers and, via napari's QtLayerList repaint race, can
        # leave phantom (empty, unnamed) rows in the layer list.
        if (
            signature == self._displayed_contact_analysis_signature
            and self._contact_analysis_layer_names()
        ):
            self._set_contact_analysis_status(f"Status: showing {contact_analysis_path.name}")
            self._update_action_states()
            return

        show_kwargs: dict[str, Any] = {
            "prefix": self._contact_analysis_layer_prefix,
            **options,
        }
        if self._cached_cell_labels is not None:
            show_kwargs["cell_labels"] = self._cached_cell_labels
        if self._cached_nucleus_labels is not None:
            show_kwargs["nucleus_labels"] = self._cached_nucleus_labels
        if self._cached_track_centroids is not None:
            show_kwargs["nucleus_track_centroids"] = self._cached_track_centroids

        # Clear now, but defer the re-add to the next event-loop tick. Removing the
        # old layers and inserting the new ones in one synchronous slot races
        # napari's QtLayerList repaint and can leave phantom (empty, unnamed) rows
        # in the layer list; letting Qt process the removals first avoids it.
        self._clear_contact_analysis_layers(set_status=False)
        contact_analysis = self._cached_contact_analysis

        def _deferred_add() -> None:
            self._pending_show_timer = None
            if self.viewer is None or contact_analysis is None:
                return
            add_contact_analysis_layers(self.viewer, contact_analysis, **show_kwargs)
            self._displayed_contact_analysis_signature = signature
            self._set_contact_analysis_status(f"Status: loaded {contact_analysis_path.name}")
            self._update_action_states()

        # Supersede any still-pending deferred add (e.g. rapid re-clicks) so a
        # later Show can't stack a second set of overlays on top of an earlier one.
        if self._pending_show_timer is not None:
            self._pending_show_timer.stop()
        # Parent the timer to ``self`` so it is destroyed with the widget and never
        # fires its callback against an already-deleted viewer/widget.
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(_deferred_add)
        timer.start(0)
        self._pending_show_timer = timer

    def _contact_analysis_layer_names(self) -> list[str]:
        if self.viewer is None:
            return []
        names: list[str] = []
        for layer in list(self.viewer.layers):
            layer_name = getattr(layer, "name", layer)
            if isinstance(layer_name, str) and layer_name.startswith(self._contact_analysis_layer_prefix):
                names.append(layer_name)
        return names

    def _clear_contact_analysis_layers(self, *, set_status: bool) -> int:
        # Any clear invalidates the "already shown" fast-path in _show_from_disk.
        self._displayed_contact_analysis_signature = None
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
            cleanup = getattr(layer, "_itasc_frame_shape_cleanup", None)
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
