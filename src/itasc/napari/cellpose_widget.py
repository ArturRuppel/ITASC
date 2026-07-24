"""Local Cellpose-SAM widget — per-channel rows with preview, run, cancel."""
from __future__ import annotations

import logging
from pathlib import Path

import napari
import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from itasc.napari._standalone_paths import StandalonePathsMixin
from itasc.napari._widget_helpers import (
    dslider as _dslider,
    islider as _islider,
    tool_btn as _tool_btn,
)
from itasc.napari.divergence_maps_widget import DivergenceMapsWidget
from itasc.napari.ui_gate import ControlClass, UiGate
from itasc.napari.ui_style import (
    add_section_full_row,
    add_section_pair_row,
    section_grid,
    stage_header_action_button,
    stage_header_label,
    status_label,
)
from itasc.napari.widgets import (
    CollapsibleSection,
    PipelineFilesWidget,
    make_pipeline_files_header,
)
from itasc.cellpose import cellpose_runner

logger = logging.getLogger(__name__)


# Canonical declared rel-paths for the two configurable inputs. The actual
# location is resolved live in ``_input_path`` (a relative override under the
# position dir, an absolute path, or a standalone pick) and need not sit under
# ``0_input`` — see ``_input_overrides``, which feeds the real paths to the
# Pipeline Files panel so its status tracks the file the stage will read.
_INPUT_NUCLEUS_REL = "0_input/nucleus.tif"
_INPUT_CELL_REL = "0_input/cell.tif"

_PIPELINE_FILES = [
    ("Inputs", [
        (_INPUT_NUCLEUS_REL, "Nucleus 3D+t"),
        (_INPUT_CELL_REL, "Cell 3D+t"),
    ]),
    ("Cellpose Outputs", [
        ("1_cellpose/nucleus_prob.tif", "Nucleus prob 3D+t"),
        ("1_cellpose/nucleus_dp.tif", "Nucleus dp 3D+t"),
        ("1_cellpose/cell_prob.tif", "Cell prob 3D+t"),
        ("1_cellpose/cell_dp.tif", "Cell dp 3D+t"),
    ]),
    ("Divergence Maps", [
        ("1_cellpose/nucleus_contours.tif", "Nucleus contours"),
        ("1_cellpose/nucleus_foreground.tif", "Nucleus foreground"),
        ("1_cellpose/cell_contours.tif", "Cell contours"),
        ("1_cellpose/cell_foreground.tif", "Cell foreground"),
    ]),
]


_REFERENCE_LAYER_NAMES = {
    "nucleus": "Reference: Nucleus 3D+t",
    "cell": "Reference: Cell 3D+t",
}

# Input dimensionality options; default to 3D+t (the historical assumption).
_LAYOUT_OPTIONS = ["2D", "2D+t", "3D", "3D+t"]
_DEFAULT_LAYOUT = "3D+t"
_MODEL_CPSAM = "Cellpose-SAM"
_MODEL_CUSTOM = "Custom"


def _layout_combo() -> QComboBox:
    combo = QComboBox()
    combo.addItems(_LAYOUT_OPTIONS)
    combo.setCurrentText(_DEFAULT_LAYOUT)
    return combo


def _make_status() -> QLabel:
    lbl = QLabel("")
    lbl.setWordWrap(True)
    lbl.setVisible(False)
    status_label(lbl)
    return lbl


def _make_progress() -> QProgressBar:
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(True)
    bar.setVisible(False)
    return bar


class CellposeWidget(StandalonePathsMixin, QWidget):
    """Local Cellpose-SAM runner — two rows (Nucleus, Cell)."""

    _progress_signal = Signal(int, int, str)

    #: QSettings application key for the standalone path pickers.
    _SETTINGS_APP = "itasc_cellpose"

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: QWidget | None = None,
        gate: UiGate | None = None,
        standalone: bool = False,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        #: App-wide UI gate; a private one is created for standalone use.
        self.gate = gate if gate is not None else UiGate(self)
        self._pos_dir: Path | None = None
        #: Standalone explicit paths (orchestrated mode leaves these ``None`` and
        #: derives everything from ``_pos_dir``'s staged subdirectories).
        self._standalone = standalone
        self._sa_nucleus: Path | None = None
        self._sa_cell: Path | None = None
        self._sa_output_dir: Path | None = None
        self._running_stage: str | None = None
        #: Whether the in-flight job can be cancelled. Full runs iterate frames
        #: and poll ``_cancel_requested``; previews are a single blocking frame
        #: with no cancellation point, so they show no ✕ and disable the row.
        self._running_cancellable = True
        self._worker = None
        self._cancel_requested = False

        self._setup_ui()
        self._connect_signals()
        self._register_gate_controls()
        self._progress_signal.connect(self._progress)
        if self._standalone:
            self._load_standalone_settings()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(6)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        # ── Input pickers (both modes) + standalone output dir ─────────
        # In the full app every path is fixed by the project structure
        # (inputs at <pos_dir>/0_input/*.tif, maps at <pos_dir>/1_cellpose/),
        # so none of these pickers are shown — see the container visibility gate
        # below. Standalone: the two inputs are explicit and required, and the
        # output-dir picker chooses where the maps are written.
        self._paths_container = QWidget()
        paths_col = QVBoxLayout(self._paths_container)
        paths_col.setContentsMargins(0, 0, 0, 0)
        self._nucleus_edit = self._add_path_row(
            paths_col,
            "Nucleus channel",
            ("raw nucleus stack (.tif)" if self._standalone
             else "default: 0_input/nucleus.tif — or pick any .tif"),
            self._on_browse_nucleus,
            self._on_input_paths_changed,
        )
        self._cell_edit = self._add_path_row(
            paths_col,
            "Cell channel",
            ("raw cell stack (.tif)" if self._standalone
             else "default: 0_input/cell.tif — or pick any .tif"),
            self._on_browse_cell,
            self._on_input_paths_changed,
        )
        self._output_dir_row = QWidget()
        out_col = QVBoxLayout(self._output_dir_row)
        out_col.setContentsMargins(0, 0, 0, 0)
        self._output_dir_edit = self._add_path_row(
            out_col,
            "Output dir",
            "directory for Cellpose maps",
            self._on_browse_output_dir,
            self._apply_standalone_paths,
        )
        paths_col.addWidget(self._output_dir_row)
        root.addWidget(self._paths_container)
        # The full app derives all paths from the project structure, so the
        # input and output pickers are standalone-only.
        self._paths_container.setVisible(self._standalone)

        # ── Pipeline files ─────────────────────────────────────────────
        self._files_widget = PipelineFilesWidget(_PIPELINE_FILES, viewer=self.viewer)
        self.output_files_tracker = self._files_widget
        self.input_files_tracker = self._files_widget
        self._pipeline_files_section = CollapsibleSection(
            "Pipeline Files", self._files_widget, expanded=False,
        )
        (
            self.pipeline_files_header,
            self.pipeline_files_header_lbl,
            self.pipeline_files_toggle_btn,
        ) = make_pipeline_files_header(
            self._pipeline_files_section, stage_key="cellpose", parent=self,
        )
        root.addWidget(self.pipeline_files_header)
        root.addWidget(self._pipeline_files_section)

        # ── Nucleus row + params ───────────────────────────────────────
        self.nucleus_params_btn = _tool_btn(
            "⚙", "Show parameters for nucleus Cellpose.", checkable=True,
        )
        self.nucleus_preview_btn = _tool_btn("▷", "Preview on current frame.")
        self.nucleus_run_btn = _tool_btn("▶", "Run nucleus Cellpose on all frames.")
        for button in (
            self.nucleus_params_btn,
            self.nucleus_preview_btn,
            self.nucleus_run_btn,
        ):
            stage_header_action_button(button, "cellpose")
        self.nucleus_section = self._build_nucleus_params_section()
        self.nucleus_section.set_header_visible(False)
        self.nucleus_section.collapse()
        self.nucleus_params_btn.toggled.connect(
            lambda checked: self.nucleus_section._toggle.setChecked(checked)
        )
        root.addLayout(self._stage_row(
            self._stage_label("Nucleus Cellpose"),
            self.nucleus_params_btn,
            self.nucleus_preview_btn,
            self.nucleus_run_btn,
        ))
        root.addWidget(self.nucleus_section)

        # ── Cell row + params ──────────────────────────────────────────
        self.cell_params_btn = _tool_btn(
            "⚙", "Show parameters for cell Cellpose.", checkable=True,
        )
        self.cell_preview_btn = _tool_btn("▷", "Preview on current frame/z-slice.")
        self.cell_run_btn = _tool_btn("▶", "Run cell Cellpose on all frames.")
        for button in (
            self.cell_params_btn,
            self.cell_preview_btn,
            self.cell_run_btn,
        ):
            stage_header_action_button(button, "cellpose")
        self.cell_section = self._build_cell_params_section()
        self.cell_section.set_header_visible(False)
        self.cell_section.collapse()
        self.cell_params_btn.toggled.connect(
            lambda checked: self.cell_section._toggle.setChecked(checked)
        )
        root.addLayout(self._stage_row(
            self._stage_label("Cell Cellpose"),
            self.cell_params_btn,
            self.cell_preview_btn,
            self.cell_run_btn,
        ))
        root.addWidget(self.cell_section)

        # ── Status + progress (shared) ─────────────────────────────────
        self.status_lbl = _make_status()
        root.addWidget(self.status_lbl)
        self.progress_bar = _make_progress()
        root.addWidget(self.progress_bar)

        # ── Divergence maps from Cellpose prob/dp outputs ─────────────
        self.divergence_maps_widget = DivergenceMapsWidget(
            self.viewer,
            show_pipeline_files=False,
            gate=self.gate,
        )
        # The divergence widget owns no Pipeline Files panel here, yet its output
        # (the foreground/contour maps) is the cellpose stage's done-signal.
        # Refresh ours when it finishes so the section dot + catalog rail repaint.
        self.divergence_maps_widget.maps_built.connect(self._on_divergence_maps_built)
        root.addWidget(self.divergence_maps_widget)

    def _build_nucleus_params_section(self) -> CollapsibleSection:
        body = QWidget(self)
        grid = section_grid()
        grid.setContentsMargins(8, 4, 4, 4)
        body.setLayout(grid)
        self.nuc_model_combo, self.nuc_model_edit, self.nuc_model_browse_btn = (
            self._add_model_controls(grid, 0, "nucleus")
        )
        self.nuc_layout_combo = _layout_combo()
        self.nuc_3d_chk = QCheckBox("3D mode")
        self.nuc_3d_chk.setChecked(False)
        self.nuc_anisotropy_spin = _dslider(0.1, 20.0, 1.5, 0.1, 2)
        self.nuc_diameter_spin = _dslider(0.0, 500.0, 0.0, 1.0, 1)
        self.nuc_min_size_spin = _islider(0, 100000, 15)
        self.nuc_gamma_spin = _dslider(0.1, 5.0, 1.0, 0.1, 2)
        row = 1
        add_section_pair_row(grid, row, "Input layout:", self.nuc_layout_combo); row += 1
        add_section_full_row(grid, row, self.nuc_3d_chk); row += 1
        add_section_pair_row(
            grid, row,
            "Anisotropy:", self.nuc_anisotropy_spin,
            "Diameter:", self.nuc_diameter_spin,
        ); row += 1
        add_section_pair_row(
            grid, row,
            "Min size:", self.nuc_min_size_spin,
            "Gamma:", self.nuc_gamma_spin,
        )
        # True-3D segmentation only makes sense when the input has a Z axis.
        self.nuc_layout_combo.currentTextChanged.connect(self._sync_nucleus_3d_enabled)
        self._sync_nucleus_3d_enabled(self.nuc_layout_combo.currentText())
        return CollapsibleSection("Nucleus parameters", body, expanded=False)

    def _sync_nucleus_3d_enabled(self, layout: str) -> None:
        has_z = cellpose_runner.layout_has_z(layout)
        self.nuc_3d_chk.setEnabled(has_z)
        self.nuc_anisotropy_spin.setEnabled(has_z)

    def _build_cell_params_section(self) -> CollapsibleSection:
        body = QWidget(self)
        grid = section_grid()
        grid.setContentsMargins(8, 4, 4, 4)
        body.setLayout(grid)
        self.cell_model_combo, self.cell_model_edit, self.cell_model_browse_btn = (
            self._add_model_controls(grid, 0, "cell")
        )
        self.cell_layout_combo = _layout_combo()
        self.cell_diameter_spin = _dslider(0.0, 500.0, 0.0, 1.0, 1)
        self.cell_min_size_spin = _islider(0, 100000, 0)
        self.cell_gamma_spin = _dslider(0.1, 5.0, 1.0, 0.1, 2)
        row = 1
        add_section_pair_row(grid, row, "Input layout:", self.cell_layout_combo); row += 1
        add_section_pair_row(
            grid, row,
            "Diameter:", self.cell_diameter_spin,
            "Min size:", self.cell_min_size_spin,
        ); row += 1
        add_section_pair_row(grid, row, "Gamma:", self.cell_gamma_spin)
        return CollapsibleSection("Cell parameters", body, expanded=False)

    def _add_model_controls(
        self, grid, row: int, channel: str
    ) -> tuple[QComboBox, QLineEdit, QPushButton]:
        combo = QComboBox()
        combo.addItems([_MODEL_CPSAM, _MODEL_CUSTOM])
        combo.setCurrentText(_MODEL_CPSAM)
        combo.setToolTip("Choose the Cellpose model used for this channel.")
        edit = QLineEdit()
        edit.setPlaceholderText("custom Cellpose model file")
        edit.setVisible(False)
        browse = QPushButton("Browse...")
        browse.setVisible(False)
        browse.clicked.connect(lambda _checked=False, ch=channel: self._on_browse_model(ch))
        combo.currentTextChanged.connect(lambda _mode, ch=channel: self._on_model_mode_changed(ch))
        add_section_pair_row(grid, row, "Model:", combo, "Custom model:", edit)
        grid.addWidget(browse, row, 4)
        return combo, edit, browse

    def _on_model_mode_changed(self, channel: str) -> None:
        custom = self._model_combo(channel).currentText() == _MODEL_CUSTOM
        self._model_edit(channel).setVisible(custom)
        self._model_browse_btn(channel).setVisible(custom)

    def _on_browse_model(self, channel: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {channel} Cellpose model",
            filter="All files (*)",
        )
        if path:
            self._model_edit(channel).setText(path)

    def _model_combo(self, channel: str) -> QComboBox:
        return self.nuc_model_combo if channel == "nucleus" else self.cell_model_combo

    def _model_edit(self, channel: str) -> QLineEdit:
        return self.nuc_model_edit if channel == "nucleus" else self.cell_model_edit

    def _model_browse_btn(self, channel: str) -> QPushButton:
        return (
            self.nuc_model_browse_btn
            if channel == "nucleus"
            else self.cell_model_browse_btn
        )

    def _selected_model_path(self, channel: str) -> Path | None:
        if self._model_combo(channel).currentText() != _MODEL_CUSTOM:
            return None
        text = self._model_edit(channel).text().strip()
        return Path(text) if text else None

    def _apply_model_selection(self, channel: str) -> bool:
        path = self._selected_model_path(channel)
        if path is not None:
            if not path.is_file():
                return False
            cellpose_runner.set_pretrained_model(path)
            return True
        cellpose_runner.set_pretrained_model(cellpose_runner.DEFAULT_PRETRAINED_MODEL)
        return True

    def _model_not_ready_reason(self, channel: str) -> str:
        path = self._selected_model_path(channel)
        label = "nucleus" if channel == "nucleus" else "cell"
        if path is None:
            return f"Select a custom {label} Cellpose model file first."
        return f"Custom {label} Cellpose model not found: {path}"

    def _model_label(self, channel: str) -> str:
        path = self._selected_model_path(channel)
        return path.name if path is not None else "Cellpose-SAM"

    @staticmethod
    def _stage_label(text: str) -> QLabel:
        return stage_header_label(QLabel(text), "cellpose")

    @staticmethod
    def _stage_row(label: QLabel, *trailing: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        row.addWidget(label)
        for w in trailing:
            row.addWidget(w)
        row.addStretch(1)
        return row

    # ------------------------------------------------------------------
    # Signals (run/cancel handlers are filled in in later tasks)
    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        self.nucleus_run_btn.clicked.connect(self._on_nucleus_run_clicked)
        self.cell_run_btn.clicked.connect(self._on_cell_run_clicked)
        self.nucleus_preview_btn.clicked.connect(self._on_nucleus_preview)
        self.cell_preview_btn.clicked.connect(self._on_cell_preview)

    def _on_nucleus_run_clicked(self) -> None:
        if self._running_stage is not None:
            self._on_cancel()
            return
        self._run_channel("nucleus")

    def _on_cell_run_clicked(self) -> None:
        if self._running_stage is not None:
            self._on_cancel()
            return
        self._run_channel("cell")

    def _on_nucleus_preview(self) -> None:
        self._preview_channel("nucleus")

    def _on_cell_preview(self) -> None:
        self._preview_channel("cell")

    def _on_cancel(self) -> None:
        self._cancel_requested = True
        worker = self._worker
        if worker is not None and hasattr(worker, "quit"):
            worker.quit()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _input_path(self, channel: str) -> Path | None:
        if self._standalone:
            return self._sa_nucleus if channel == "nucleus" else self._sa_cell
        if self._pos_dir is None:
            return None
        # Integrated: the input field overrides the default 0_input name. A blank
        # field falls back to the canonical name; a relative override resolves
        # under the position dir, an absolute one is used verbatim.
        edit = self._nucleus_edit if channel == "nucleus" else self._cell_edit
        default_name = "nucleus.tif" if channel == "nucleus" else "cell.tif"
        text = edit.text().strip()
        if text:
            override = Path(text)
            return override if override.is_absolute() else self._pos_dir / override
        return self._pos_dir / "0_input" / default_name

    def _output_dir(self) -> Path | None:
        if self._standalone:
            return self._sa_output_dir
        return None if self._pos_dir is None else self._pos_dir / "1_cellpose"

    def _input_overrides(self) -> dict[str, Path | None]:
        """Live locations for the two configurable Pipeline-Files input rows.

        The rows are declared with the canonical ``0_input/*.tif`` names, but
        the real input is configurable and need not sit under ``0_input``. Resolve
        it via :meth:`_input_path` so the panel checks the file the stage reads.
        """
        return {
            _INPUT_NUCLEUS_REL: self._input_path("nucleus"),
            _INPUT_CELL_REL: self._input_path("cell"),
        }

    def _refresh_files(self, pos_dir: Path | None) -> None:
        """Refresh the Pipeline Files panel with the live input overrides."""
        self._files_widget.refresh(pos_dir, overrides=self._input_overrides())

    # ── Standalone helpers ─────────────────────────────────────────────
    # Row building / browse plumbing / QSettings come from StandalonePathsMixin;
    # the apply step is Cellpose-specific (two explicit input stacks + a flat
    # output directory that also serves as the divergence widget's maps dir).
    def _apply_standalone_paths(self) -> None:
        nuc = self._nucleus_edit.text().strip()
        cel = self._cell_edit.text().strip()
        out = self._output_dir_edit.text().strip()
        self._sa_nucleus = Path(nuc) if nuc else None
        self._sa_cell = Path(cel) if cel else None
        self._sa_output_dir = Path(out) if out else None
        self._save_standalone_settings()
        # The existing _pos_dir-based guards and the pipeline-files panel expect a
        # real directory; the output dir doubles as the staged-file root here.
        self._pos_dir = self._sa_output_dir
        self._refresh_files(self._sa_output_dir)
        self.divergence_maps_widget.set_maps_dir(self._sa_output_dir)
        self.gate.recompute()

    def _on_browse_nucleus(self) -> None:
        self._browse_file_into(
            self._nucleus_edit, "Select nucleus channel", self._on_nucleus_selected
        )

    def _on_input_paths_changed(self) -> None:
        """Input-field edit handler.

        Standalone reads all three pickers together (inputs + output dir);
        integrated reads the two input fields lazily in :meth:`_input_path`, so a
        light status refresh is enough — and calling the standalone apply here
        would wrongly clobber ``_pos_dir`` with the (empty) output-dir field.
        """
        if self._standalone:
            self._apply_standalone_paths()
        else:
            self._refresh_files(self._pos_dir)
            self.gate.recompute()

    def _on_nucleus_selected(self) -> None:
        self._on_input_paths_changed()
        self._autoselect_layout("nucleus")

    def _on_browse_cell(self) -> None:
        self._browse_file_into(
            self._cell_edit, "Select cell channel", self._on_cell_selected
        )

    def _on_cell_selected(self) -> None:
        self._on_input_paths_changed()
        self._autoselect_layout("cell")

    def _on_browse_output_dir(self) -> None:
        self._browse_dir_into(
            self._output_dir_edit, "Select output directory", self._apply_standalone_paths
        )

    def _standalone_fields(self) -> dict:
        return {
            "nucleus": self._nucleus_edit,
            "cell": self._cell_edit,
            "output_dir": self._output_dir_edit,
        }

    def _load_standalone_settings(self) -> None:
        self._load_path_settings(self._SETTINGS_APP, self._standalone_fields())
        if any(edit.text().strip() for edit in self._standalone_fields().values()):
            self._apply_standalone_paths()

    def _save_standalone_settings(self) -> None:
        self._save_path_settings(self._SETTINGS_APP, self._standalone_fields())

    # ------------------------------------------------------------------
    # Run flow
    # ------------------------------------------------------------------
    def _channel_layout(self, channel: str) -> str:
        combo = self.nuc_layout_combo if channel == "nucleus" else self.cell_layout_combo
        return combo.currentText()

    def _autoselect_layout(self, channel: str) -> None:
        """Best-effort: preselect the layout from the input file's ndim.

        Only acts on the unambiguous 2-D / 4-D cases; a 3-D file keeps the user's
        explicit choice. Never raises — a missing/unreadable file is ignored.
        """
        path = self._input_path(channel)
        if path is None:
            return
        try:
            with tifffile.TiffFile(str(path)) as tf:
                ndim = len(tf.series[0].shape)
        except Exception:
            return
        inferred = cellpose_runner.infer_layout_from_ndim(ndim)
        if inferred is None:
            return
        combo = self.nuc_layout_combo if channel == "nucleus" else self.cell_layout_combo
        combo.setCurrentText(inferred)

    def _build_nucleus_params(self) -> cellpose_runner.NucleusParams:
        # True-3D segmentation requires a Z axis; a Z-less input forces 2D.
        do_3d = self.nuc_3d_chk.isChecked() and cellpose_runner.layout_has_z(
            self._channel_layout("nucleus")
        )
        return cellpose_runner.NucleusParams(
            do_3d=do_3d,
            anisotropy=float(self.nuc_anisotropy_spin.value()),
            diameter=float(self.nuc_diameter_spin.value()),
            min_size=int(self.nuc_min_size_spin.value()),
            gamma=float(self.nuc_gamma_spin.value()),
        )

    def _build_cell_params(self) -> cellpose_runner.CellParams:
        return cellpose_runner.CellParams(
            diameter=float(self.cell_diameter_spin.value()),
            min_size=int(self.cell_min_size_spin.value()),
            gamma=float(self.cell_gamma_spin.value()),
        )

    def _run_channel(self, channel: str) -> None:
        if self._pos_dir is None:
            self._status("No project open.")
            return
        in_path = self._input_path(channel)
        if in_path is None or not in_path.exists():
            self._status(f"Missing: {in_path.name if in_path else '(no path)'}")
            return
        if not self._apply_model_selection(channel):
            self._status(self._model_not_ready_reason(channel))
            return
        out_dir = self._output_dir()
        params = (
            self._build_nucleus_params() if channel == "nucleus"
            else self._build_cell_params()
        )
        layout = self._channel_layout(channel)
        pos_dir = self._pos_dir
        self._cancel_requested = False

        def _done(result):
            self._worker = None
            self._set_running_stage(None)
            self._clear_progress()
            self._refresh_files(pos_dir)
            self._refresh_divergence(pos_dir)
            label = "Nucleus" if channel == "nucleus" else "Cell"
            self._status(f"{label} Cellpose complete — wrote {channel}_*.tif")

        def _error(exc):
            self._worker = None
            self._set_running_stage(None)
            self._clear_progress()
            if isinstance(exc, cellpose_runner.CancelledError):
                self._status("Cancelled.")
            else:
                self._status(f"Error: {exc}")
                logger.exception("Cellpose run error", exc_info=exc)

        progress_signal = self._progress_signal

        @thread_worker(connect={
            "yielded": self._on_progress,
            "returned": _done,
            "errored": _error,
        })
        def _worker():
            yield (0, 1, "Loading input...")
            stack = cellpose_runner.to_tzyx(
                np.asarray(tifffile.imread(str(in_path))), layout
            )

            def _cb_progress(done, total, msg):
                progress_signal.emit(int(done), int(total), str(msg))

            def _cb_cancel():
                return self._cancel_requested

            if channel == "nucleus":
                prob, dp = cellpose_runner.run_nucleus_stack(
                    stack, params,
                    progress_cb=_cb_progress, cancel_cb=_cb_cancel,
                )
            else:
                prob, dp = cellpose_runner.run_cell_stack(
                    stack, params,
                    progress_cb=_cb_progress, cancel_cb=_cb_cancel,
                )
            yield (1, 1, "Writing outputs...")
            cellpose_runner.write_outputs(prob, dp, out_dir, channel)
            return None

        self._set_running_stage(channel)
        self._status(
            f"Loading {self._model_label(channel)} model on {cellpose_runner.device_label()} "
            f"(~10s on first run)..." if not cellpose_runner.is_model_loaded()
            else f"Running {channel} Cellpose..."
        )
        self._worker = _worker()

    # ------------------------------------------------------------------
    # Preview flow
    # ------------------------------------------------------------------
    def _current_tz(self) -> tuple[int, int]:
        step = getattr(getattr(self.viewer, "dims", None), "current_step", (0, 0))
        t = int(step[0]) if len(step) >= 1 else 0
        z = int(step[1]) if len(step) >= 2 else 0
        return t, z

    @staticmethod
    def _flow_magnitude(dp: np.ndarray) -> np.ndarray:
        # dp has shape (C, ...) — sum-of-squares over the channel axis.
        return np.sqrt(np.sum(np.asarray(dp, dtype=np.float32) ** 2, axis=0))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)

    @staticmethod
    def _flow_contrast_limits(flow: np.ndarray) -> tuple[float, float]:
        # Derive limits from the populated frame only — flow_full is mostly
        # zeros, so napari's auto-contrast undersamples and clips the peaks.
        hi = float(np.asarray(flow, dtype=np.float32).max())
        return 0.0, max(hi, 1e-6)

    def _preview_channel(self, channel: str) -> None:
        if self._running_stage is not None:
            self._status("Cellpose task already running.")
            return
        if self._pos_dir is None:
            self._status("No project open.")
            return
        in_path = self._input_path(channel)
        if in_path is None or not in_path.exists():
            self._status(f"Missing: {in_path.name if in_path else '(no path)'}")
            return
        if not self._apply_model_selection(channel):
            self._status(self._model_not_ready_reason(channel))
            return

        params = (
            self._build_nucleus_params() if channel == "nucleus"
            else self._build_cell_params()
        )
        layout = self._channel_layout(channel)
        self._cancel_requested = False
        self._set_running_stage(channel, cancellable=False)
        self._progress(0, 0, f"Loading {channel} reference stack for preview...")
        try:
            stack = cellpose_runner.to_tzyx(
                np.asarray(tifffile.imread(str(in_path))), layout
            )
            self._show_reference_stack(channel, stack)
            t, z = self._current_tz()
        except Exception as exc:
            self._set_running_stage(None)
            self._clear_progress()
            self._status(f"Error: {exc}")
            logger.exception("Cellpose preview load error", exc_info=exc)
            return

        def _done(result):
            self._worker = None
            self._set_running_stage(None)
            self._clear_progress()
            status_msg, layers = result
            for name, data, kwargs in layers:
                self._show_layer(name, data, kwargs, self.viewer.add_image)
            self._status(status_msg)

        def _error(exc):
            self._worker = None
            self._set_running_stage(None)
            self._clear_progress()
            self._status(f"Error: {exc}")
            logger.exception("Cellpose preview error", exc_info=exc)

        @thread_worker(connect={
            "yielded": self._on_progress,
            "returned": _done,
            "errored": _error,
        })
        def _worker():
            T, Z = stack.shape[:2]
            t_clamped = min(max(t, 0), T - 1)
            z_clamped = min(max(z, 0), Z - 1)

            if channel == "nucleus":
                if params.do_3d:
                    yield (
                        0, 0,
                        f"Previewing nucleus 3D t={t_clamped} "
                        f"on {cellpose_runner.device_label()} "
                        f"(Z={Z}, anisotropy={params.anisotropy})...",
                    )
                    prob_logits, dp = cellpose_runner.run_nucleus_frame(
                        stack[t_clamped], z=None, params=params,
                    )
                    prob = self._sigmoid(prob_logits)
                    flow = self._flow_magnitude(dp)  # (Z, Y, X)
                    prob_full = np.zeros((T, Z, *prob.shape[-2:]), dtype=np.float32)
                    flow_full = np.zeros_like(prob_full)
                    prob_full[t_clamped] = prob
                    flow_full[t_clamped] = flow
                    flow_clim = self._flow_contrast_limits(flow)
                    status_msg = (
                        f"Preview: nucleus 3D t={t_clamped} "
                        f"(Z={Z}, anisotropy={params.anisotropy})"
                    )
                else:
                    yield (
                        0, 0,
                        f"Previewing nucleus 2D t={t_clamped} z={z_clamped} "
                        f"on {cellpose_runner.device_label()}...",
                    )
                    prob_logits, dp = cellpose_runner.run_nucleus_frame(
                        stack[t_clamped], z=z_clamped, params=params,
                    )
                    prob = self._sigmoid(prob_logits)
                    flow = self._flow_magnitude(dp)  # (Y, X)
                    prob_full = np.zeros((T, Z, *prob.shape), dtype=np.float32)
                    flow_full = np.zeros_like(prob_full)
                    prob_full[t_clamped, z_clamped] = prob
                    flow_full[t_clamped, z_clamped] = flow
                    flow_clim = self._flow_contrast_limits(flow)
                    status_msg = (
                        f"Preview: nucleus 2D t={t_clamped} z={z_clamped} "
                        f"(diameter={params.diameter})"
                    )
                return status_msg, [
                    (
                        "Preview: Nucleus prob",
                        prob_full,
                        {
                            "colormap": "viridis",
                            "blending": "additive",
                            "contrast_limits": (0.0, 1.0),
                        },
                    ),
                    (
                        "Preview: Nucleus flow",
                        flow_full,
                        {
                            "colormap": "inferno",
                            "blending": "additive",
                            "contrast_limits": flow_clim,
                        },
                    ),
                ]

            yield (
                0, 0,
                f"Previewing cell 2D t={t_clamped} z={z_clamped} "
                f"on {cellpose_runner.device_label()}...",
            )
            prob_logits, dp = cellpose_runner.run_cell_frame(
                stack[t_clamped], z=z_clamped, params=params,
            )
            prob = self._sigmoid(prob_logits)
            flow = self._flow_magnitude(dp)
            prob_full = np.zeros((T, Z, *prob.shape), dtype=np.float32)
            flow_full = np.zeros_like(prob_full)
            prob_full[t_clamped, z_clamped] = prob
            flow_full[t_clamped, z_clamped] = flow
            flow_clim = self._flow_contrast_limits(flow)
            return (
                f"Preview: cell t={t_clamped} z={z_clamped} "
                f"(diameter={params.diameter})",
                [
                    (
                        "Preview: Cell prob",
                        prob_full,
                        {
                            "colormap": "viridis",
                            "blending": "additive",
                            "contrast_limits": (0.0, 1.0),
                        },
                    ),
                    (
                        "Preview: Cell flow",
                        flow_full,
                        {
                            "colormap": "inferno",
                            "blending": "additive",
                            "contrast_limits": flow_clim,
                        },
                    ),
                ],
            )

        self._worker = _worker()

    def _show_reference_stack(self, channel: str, stack: np.ndarray) -> None:
        name = _REFERENCE_LAYER_NAMES[channel]
        self._show_layer(
            name, stack,
            {"colormap": "gray", "blending": "additive"},
            self.viewer.add_image,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_input_names(self, names: dict[str, str]) -> None:
        """Adopt the host's configured raw-input names (integrated mode only).

        The full app configures the input names once, in the Data-folders panel's
        discovery fields, and they need not be the canonical ``0_input/*.tif``.
        Mirror them into the (hidden) per-channel input fields so ``_input_path``
        — and thus run, preview, and Pipeline Files status — track the file the
        stage will actually read. Standalone owns its own explicit pickers.
        """
        if self._standalone:
            return
        self._nucleus_edit.setText(names.get("nucleus", ""))
        self._cell_edit.setText(names.get("cell", ""))
        # editingFinished does not fire on programmatic setText — refresh by hand.
        self._on_input_paths_changed()

    def refresh(self, pos_dir: Path | None) -> None:
        self._pos_dir = pos_dir
        self._refresh_files(pos_dir)
        self._refresh_divergence(pos_dir)
        if pos_dir is not None:
            self._autoselect_layout("nucleus")
            self._autoselect_layout("cell")

    def _on_divergence_maps_built(self) -> None:
        """Repaint Pipeline Files after the embedded divergence widget writes maps.

        The maps are this stage's tracked output but the divergence widget has no
        Pipeline Files panel of its own, so refresh ours against the active root
        (standalone output dir or the staged position dir).
        """
        root = self._sa_output_dir if self._standalone else self._pos_dir
        self._refresh_files(root)

    def _refresh_divergence(self, pos_dir: Path | None) -> None:
        """Point the embedded divergence widget at the active maps location.

        Orchestrated: maps live under ``<pos_dir>/1_cellpose``. Standalone: maps
        live directly in the chosen output directory.
        """
        if self._standalone:
            self.divergence_maps_widget.set_maps_dir(self._sa_output_dir)
        else:
            self.divergence_maps_widget.refresh(pos_dir)

    def get_state(self) -> dict:
        return {
            "nucleus": {
                "model": self.nuc_model_combo.currentText(),
                "custom_model": self.nuc_model_edit.text().strip(),
                "layout": self.nuc_layout_combo.currentText(),
                "do_3d": self.nuc_3d_chk.isChecked(),
                "anisotropy": self.nuc_anisotropy_spin.value(),
                "diameter": self.nuc_diameter_spin.value(),
                "min_size": self.nuc_min_size_spin.value(),
                "gamma": self.nuc_gamma_spin.value(),
            },
            "cell": {
                "model": self.cell_model_combo.currentText(),
                "custom_model": self.cell_model_edit.text().strip(),
                "layout": self.cell_layout_combo.currentText(),
                "diameter": self.cell_diameter_spin.value(),
                "min_size": self.cell_min_size_spin.value(),
                "gamma": self.cell_gamma_spin.value(),
            },
            "divergence_maps": self.divergence_maps_widget.get_state(),
        }

    def set_state(self, state: dict) -> None:
        if not isinstance(state, dict):
            return
        nuc = state.get("nucleus", {})
        if isinstance(nuc, dict):
            if nuc.get("model") in (_MODEL_CPSAM, _MODEL_CUSTOM):
                self.nuc_model_combo.setCurrentText(nuc["model"])
            if "custom_model" in nuc:
                self.nuc_model_edit.setText(str(nuc["custom_model"]))
            if nuc.get("layout") in _LAYOUT_OPTIONS:
                self.nuc_layout_combo.setCurrentText(nuc["layout"])
            if "do_3d" in nuc:
                self.nuc_3d_chk.setChecked(bool(nuc["do_3d"]))
            if "anisotropy" in nuc:
                self.nuc_anisotropy_spin.setValue(float(nuc["anisotropy"]))
            if "diameter" in nuc:
                self.nuc_diameter_spin.setValue(float(nuc["diameter"]))
            if "min_size" in nuc:
                self.nuc_min_size_spin.setValue(int(nuc["min_size"]))
            if "gamma" in nuc:
                self.nuc_gamma_spin.setValue(float(nuc["gamma"]))
        cel = state.get("cell", {})
        if isinstance(cel, dict):
            if cel.get("model") in (_MODEL_CPSAM, _MODEL_CUSTOM):
                self.cell_model_combo.setCurrentText(cel["model"])
            if "custom_model" in cel:
                self.cell_model_edit.setText(str(cel["custom_model"]))
            if cel.get("layout") in _LAYOUT_OPTIONS:
                self.cell_layout_combo.setCurrentText(cel["layout"])
            if "diameter" in cel:
                self.cell_diameter_spin.setValue(float(cel["diameter"]))
            if "min_size" in cel:
                self.cell_min_size_spin.setValue(int(cel["min_size"]))
            if "gamma" in cel:
                self.cell_gamma_spin.setValue(float(cel["gamma"]))
        if "divergence_maps" in state:
            self.divergence_maps_widget.set_state(state["divergence_maps"])

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _status(self, msg: str) -> None:
        self.status_lbl.setText(msg)
        self.status_lbl.setVisible(bool(msg))
        if msg:
            logger.info(msg)

    def _progress(self, done: int, total: int, msg: str) -> None:
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)
        self._status(msg)

    def _on_progress(self, data) -> None:
        if isinstance(data, tuple):
            self._progress(*data)
        else:
            self._status(str(data))

    def _clear_progress(self) -> None:
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

    def _register_gate_controls(self) -> None:
        """Register the two channel rows with the app-wide UI gate.

        Cellpose writes to the viewer, so its run/preview/params are blocked
        while any viewer owner (correction / live preview) is active. Within
        the widget, the active channel's row stays usable while the other is
        disabled — expressed via ``when`` predicates over ``_running_stage``.
        """
        g = self.gate
        idle = lambda: self._running_stage is None
        for channel, params_btn, preview_btn, run_btn in (
            ("nucleus", self.nucleus_params_btn, self.nucleus_preview_btn, self.nucleus_run_btn),
            ("cell", self.cell_params_btn, self.cell_preview_btn, self.cell_run_btn),
        ):
            # The run/✕ button stays live on its own row only while a
            # cancellable job (a full run) is in flight; a preview disables it.
            own = lambda c=channel: self._running_stage is None or (
                self._running_stage == c and self._running_cancellable
            )
            # ⚙ params just toggle a parameter panel — always available.
            g.register(params_btn, ControlClass.HARMLESS)
            g.register(preview_btn, ControlClass.RUN_VIEWER, when=idle)
            g.register(run_btn, ControlClass.RUN_VIEWER, when=own)
        g.recompute()

    def _set_running_stage(self, stage_key: str | None, *, cancellable: bool = True) -> None:
        """Swap the active row's ▶/✕ glyph; enablement is owned by the gate.

        ``None`` means idle; ``'nucleus'`` or ``'cell'`` claims that row. The
        gate's ``when`` predicates read ``self._running_stage`` to disable the
        other row and the active row's preview while a job is in flight.

        ``cancellable`` is ``True`` for full runs (the ✕ cancels them); previews
        pass ``False`` — they can't be interrupted, so the glyph stays ▶ and the
        gate disables the whole row until the frame returns.
        """
        self._running_stage = stage_key
        self._running_cancellable = cancellable
        if stage_key is None:
            self.nucleus_run_btn.setText("▶")
            self.nucleus_run_btn.setToolTip("Run nucleus Cellpose on all frames.")
            self.cell_run_btn.setText("▶")
            self.cell_run_btn.setToolTip("Run cell Cellpose on all frames.")
            self._cancel_requested = False
        elif cancellable:
            run_btn = self.nucleus_run_btn if stage_key == "nucleus" else self.cell_run_btn
            run_btn.setText("✕")
            run_btn.setToolTip("Cancel.")
        self.gate.recompute()

    # ------------------------------------------------------------------
    # Layer helper (mirrors CellWorkflowWidget._show_layer)
    # ------------------------------------------------------------------
    def _show_layer(self, name, data, kwargs, adder):
        if name in self.viewer.layers:
            try:
                layer = self.viewer.layers[name]
                layer.data = data
                clim = kwargs.get("contrast_limits")
                if clim is not None:
                    layer.contrast_limits = clim
                return
            except Exception:
                self.viewer.layers.remove(self.viewer.layers[name])
                adder(data, name=name, **kwargs)
        else:
            adder(data, name=name, **kwargs)


def make_cellpose_widget(napari_viewer=None):
    """napari plugin factory for the standalone Cellpose piece.

    Patches the napari layer-controls delegate (best-effort) and returns the
    Cellpose widget in standalone mode, with its own nucleus/cell input stacks
    and output-directory pickers.
    """
    try:
        from itasc.napari._napari_compat import patch_napari_layer_delegate

        patch_napari_layer_delegate()
    except Exception:
        pass
    # napari does not inject the viewer into function-based widget factories
    # (only into class-based callables / magicgui types), so ``napari_viewer``
    # arrives as ``None``. The widget needs a live viewer, so fall back to the
    # active one.
    if napari_viewer is None:
        napari_viewer = napari.current_viewer()
    return CellposeWidget(napari_viewer, standalone=True)
