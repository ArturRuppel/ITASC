"""New Project wizard dialog.

Lets the user pick separate data-input and project-output directories, choose
which pipeline stages to include.  On acceptance it writes
``pipeline_schema.json``, creates the directory skeleton,
writes a ``PIPELINE_LAYOUT.txt`` description, and updates the viewer state.

Pixel size and time interval are intentionally *not* collected here — they
are pulled from the raw acquisition metadata during the Data Prep stage.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from cellflow.core.paths import STAGE_DIRS, schema_path
from cellflow.core.schema import PipelineSchema
from ._plugin import STAGE_DISPLAY_NAMES, STAGE_ORDER, STAGES


class NewProjectDialog(QDialog):
    """Dialog to create a new CellFlow pipeline project.

    Parameters
    ----------
    viewer:
        Active napari viewer instance.
    state:
        Shared :class:`~cellflow.napari.registry.ViewerState`.
    parent:
        Qt parent widget.
    """

    def __init__(self, viewer, state, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._state = state
        self.setWindowTitle("New Pipeline Project")
        self.setMinimumWidth(520)

        self._input_path: Optional[Path] = None
        self._output_path: Optional[Path] = None
        self._checkboxes: dict[str, QCheckBox] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ── Data input directory ──────────────────────────────────────
        input_group = QGroupBox("Data Input Directory")
        input_layout = QVBoxLayout()
        input_group.setLayout(input_layout)

        input_note = QLabel(
            "Where the raw acquisition data lives (e.g. the NDTiff dataset root).\n"
            "Pixel size and time interval will be read from here automatically."
        )
        input_note.setStyleSheet("color: gray; font-size: 8pt;")
        input_note.setWordWrap(True)
        input_layout.addWidget(input_note)

        input_row = QHBoxLayout()
        self._input_edit = QLineEdit()
        self._input_edit.setPlaceholderText("/path/to/raw_data")
        self._input_edit.setReadOnly(True)
        input_row.addWidget(self._input_edit)

        input_btn = QPushButton("Browse…")
        input_btn.clicked.connect(self._browse_input)
        input_row.addWidget(input_btn)
        input_layout.addLayout(input_row)
        layout.addWidget(input_group)

        # ── Project output directory ──────────────────────────────────
        output_group = QGroupBox("Project Output Directory")
        output_layout = QVBoxLayout()
        output_group.setLayout(output_layout)

        output_note = QLabel(
            "Where processed data will be written. "
            "The pipeline_schema.json and pos## folders are created here."
        )
        output_note.setStyleSheet("color: gray; font-size: 8pt;")
        output_note.setWordWrap(True)
        output_layout.addWidget(output_note)

        output_row = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setPlaceholderText("/path/to/project_output")
        self._output_edit.setReadOnly(True)
        output_row.addWidget(self._output_edit)

        output_btn = QPushButton("Browse…")
        output_btn.clicked.connect(self._browse_output)
        output_row.addWidget(output_btn)
        output_layout.addLayout(output_row)
        layout.addWidget(output_group)

        # ── Stage selection ───────────────────────────────────────────
        stage_group = QGroupBox("Pipeline Stages (select stages to enable)")
        stage_outer = QVBoxLayout()
        stage_group.setLayout(stage_outer)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMaximumHeight(200)

        container = QWidget()
        cb_layout = QVBoxLayout(container)
        cb_layout.setSpacing(3)

        installed = set(STAGES.keys())
        ordered = [s for s in STAGE_ORDER if s in installed]
        ordered += sorted(installed - set(STAGE_ORDER))

        for name in ordered:
            display = STAGE_DISPLAY_NAMES.get(name, name)
            cb = QCheckBox(display)
            cb.setChecked(True)
            cb.setObjectName(name)
            self._checkboxes[name] = cb
            cb_layout.addWidget(cb)

        if not ordered:
            cb_layout.addWidget(QLabel("No stages installed."))

        cb_layout.addStretch()
        scroll.setWidget(container)
        stage_outer.addWidget(scroll)
        layout.addWidget(stage_group)

        # ── Buttons ───────────────────────────────────────────────────
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: red; font-size: 9pt;")
        layout.addWidget(self._status_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _browse_input(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Data Input Directory")
        if d:
            self._input_path = Path(d)
            self._input_edit.setText(d)

    def _browse_output(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Project Output Directory")
        if d:
            self._output_path = Path(d)
            self._output_edit.setText(d)

    def _on_accept(self) -> None:
        if self._output_path is None:
            self._status_label.setText("Please choose a project output directory.")
            return

        selected_stages = [
            name for name, cb in self._checkboxes.items() if cb.isChecked()
        ]
        if not selected_stages:
            self._status_label.setText("Please select at least one stage.")
            return

        try:
            self._create_project(self._output_path, self._input_path, selected_stages)
        except Exception as exc:
            self._status_label.setText(f"Error: {exc}")
            return

        self.accept()

    # ------------------------------------------------------------------
    # Project creation
    # ------------------------------------------------------------------

    def _create_project(
        self,
        output_root: Path,
        input_dir: Optional[Path],
        stages: List[str],
    ) -> None:
        """Write schema, create directory skeleton, write layout doc, update state."""
        schema = PipelineSchema(
            stages=stages,
            input_dir=str(input_dir) if input_dir is not None else None,
        )

        output_root.mkdir(parents=True, exist_ok=True)
        schema.save(schema_path(output_root))

        # Create a single pos00 skeleton so users can see the structure
        _create_pos_skeleton(output_root, pos=0, stages=stages)

        # Write human-readable layout description
        _write_pipeline_layout(output_root, input_dir, stages)

        # Update shared state
        self._state.set_project_dir(output_root)


def _create_pos_skeleton(root: Path, pos: int, stages: List[str]) -> None:
    """Create the directory skeleton for one position."""
    from cellflow.core.paths import pos_dir, stage_dir
    pos_dir(root, pos).mkdir(parents=True, exist_ok=True)
    for stage_name in stages:
        if stage_name in STAGE_DIRS:
            stage_dir(root, pos, stage_name).mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# PIPELINE_LAYOUT.txt generation
# ------------------------------------------------------------------

_STAGE_DESCRIPTIONS: dict[str, tuple[str, str, str]] = {
    # stage_key: (folder_name, description, expected_files_and_shapes)
    "raw_import": (
        "0_raw/",
        "Raw TIFF exports from NDTiff acquisition",
        "t###_nucleus.tif per timepoint — shape (Z, Y, X), dtype uint16",
    ),
    "cellpose_nucleus": (
        "1a_cellpose_nucleus/",
        "Cellpose 3-D nucleus segmentation outputs",
        "t###_prob.tif, t###_dp.tif per timepoint\n"
        "  prob shape: (Z, Y, X), dtype float32\n"
        "  dp shape: (2, Z, Y, X) or (Z, Y, X, 2), dtype float32",
    ),
    "cellpose_cell": (
        "1b_cellpose_cell/",
        "Cellpose 2-D cell segmentation outputs",
        "cell_prob.tif, cell_dp.tif — shape (T, Y, X), dtype float32",
    ),
    "flow_watershed": (
        "2_flow_watershed/",
        "Flow-guided watershed cell labels",
        "cell_labels.tif, cell_labels_raw.tif — shape (T, Y, X), dtype int32",
    ),
    "contours": (
        "2b_contours/",
        "Cellpose-derived foreground and contour maps",
        "foreground.tif, contours.tif — shape (T, Y, X), dtype float32",
    ),
    "tracking": (
        "3_tracking/",
        "Ultrack cell tracking outputs",
        "tracked_labels.tif — shape (T, Y, X), dtype uint32\n"
        "  tracks.csv, data.db",
    ),
    "graph_extraction": (
        "4_analysis/",
        "Graph extraction and topology analysis",
        "tracked_labels.tif (copied from 3_tracking/)",
    ),
    "topology_analysis": (
        "4_analysis/",
        "Graph extraction and topology analysis",
        "tracked_labels.tif (copied from 3_tracking/)",
    ),
}


def _write_pipeline_layout(
    output_root: Path,
    input_dir: Optional[Path],
    stages: List[str],
) -> None:
    """Write PIPELINE_LAYOUT.txt at the project output root."""
    lines: List[str] = [
        "CellFlow Pipeline Directory Layout",
        "=" * 60,
        "",
        f"Created: {date.today().isoformat()}",
        f"Project output root: {output_root}",
    ]
    if input_dir is not None:
        lines.append(f"Raw data input: {input_dir}")
    lines += [
        "",
        "Top-level files:",
        "  pipeline_schema.json  — Pipeline configuration and enabled stages",
        "  PIPELINE_LAYOUT.txt   — This file",
        "",
        "For each position (pos00, pos01, ...):",
        "  pos##/",
        "  ├── pipeline_manifest.json  — Stage run status and timestamps",
        "  ├── pipeline.log            — Stage execution log (JSON-lines)",
    ]

    seen_dirs: set[str] = set()
    for stage_name in stages:
        if stage_name not in _STAGE_DESCRIPTIONS:
            continue
        folder, desc, files = _STAGE_DESCRIPTIONS[stage_name]
        if folder in seen_dirs:
            continue
        seen_dirs.add(folder)
        lines.append(f"  ├── {folder}")
        lines.append(f"  │     {desc}")
        for file_line in files.splitlines():
            lines.append(f"  │     {file_line}")

    lines += [
        "",
        "Stage pipeline order:",
    ]
    for stage_name in stages:
        display = STAGE_DISPLAY_NAMES.get(stage_name, stage_name)
        folder = _STAGE_DESCRIPTIONS.get(stage_name, (stage_name + "/", "", ""))[0]
        lines.append(f"  {display:30s} → {folder}")

    lines += [
        "",
        "Notes:",
        "  - Pixel size (µm) and time interval are read from the raw acquisition",
        "    metadata during the Data Prep stage; they are not stored here.",
        "  - Re-running a stage with 'Overwrite' will replace existing outputs.",
        "  - The manifest tracks the status of each stage run.",
    ]

    txt = "\n".join(lines) + "\n"
    (output_root / "PIPELINE_LAYOUT.txt").write_text(txt)
