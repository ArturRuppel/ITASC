"""New Project wizard dialog.

Lets the user pick separate data-input and project-output directories.
On acceptance it writes ``pipeline_schema.json``, writes a
``PIPELINE_LAYOUT.txt`` description, and updates the viewer state.

Pixel size and time interval are intentionally *not* collected here — they
are pulled from the raw acquisition metadata during the Data Prep stage.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cellflow.core.paths import schema_path
from cellflow.core.schema import PipelineSchema
from ._plugin import STAGE_DISPLAY_NAMES, STAGE_ORDER


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
        input_note.setStyleSheet("color: white; font-size: 8pt;")
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
            "The pipeline_schema.json is created here; pos## folders appear when stages run."
        )
        output_note.setStyleSheet("color: white; font-size: 8pt;")
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

        try:
            self._create_project(self._output_path, self._input_path)
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
    ) -> None:
        """Write schema, create full directory skeleton, write layout doc, update state."""
        schema = PipelineSchema(
            stages=list(STAGE_ORDER),
            input_dir=str(input_dir) if input_dir is not None else None,
        )

        output_root.mkdir(parents=True, exist_ok=True)
        schema.save(schema_path(output_root))

        # Write human-readable layout description
        _write_pipeline_layout(output_root, input_dir)

        # Update shared state
        self._state.set_project_dir(output_root)

# ------------------------------------------------------------------
# PIPELINE_LAYOUT.txt generation
# ------------------------------------------------------------------

# Canonical per-directory descriptions in pipeline order.
# Each entry: (folder, detail_lines).  Directories shared by multiple stages
# (1_cellpose/, 5_analysis/) list all their outputs together.
_DIR_DESCRIPTIONS: List[tuple[str, List[str]]] = [
    (
        "0_input/",
        [
            "Raw TIFF exports from the NDTiff acquisition with z-shift correction  [step 0]",
            "  nucleus_4d.tif           (T, Z, H, W) uint16",
            "  cell_4d.tif              (T, Z, H, W) uint16",
            "  nucleus_zavg.tif         (T, H, W)       uint16",
            "  cell_zavg.tif            (T, H, W)       uint16",
            "  z_shift.csv              time,z_shift_slices,fit_mse,...",
        ],
    ),
    (
        "1_cellpose/",
        [
            "Cellpose Cluster outputs  [step 1]",
            "  nucleus_dp_4d.tif       (T, C, Z, H, W) float32  [C=3 in full-3D mode, C=2 slice-wise]",
            "  nucleus_prob_4d.tif     (T, Z, H, W)    float32",
            "  cell_dp_4d.tif          (T, Z, 2, H, W) float32",
            "  cell_prob_4d.tif        (T, Z, H, W)    float32",
            "  cell_dp_zavg.tif        (T, 2, H, W)    float32",
            "  cell_prob_zavg.tif      (T, H, W)       float32",
        ],
    ),
    (
        "2_nucleus_ultrack/",
        [
            "Nucleus Hypotheses outputs  [step 2]",
            "  hypotheses.h5",
            "  hypotheses_manifest.json",
            "  labelmaps/labelmap_*.tif",
        ],
    ),
    (
        "3_correction/",
        [
            "Manual nucleus correction  [step 3]",
            "  nuclear_labels_corrected.tif  (T, H, W) int32",
        ],
    ),
    (
        "4_cell_ultrack/",
        [
            "Cell Ultrack outputs  [step 4]",
            "  data.db                  Ultrack SQLite database",
            "  tracks.csv",
            "  tracked_labels.tif       (T, Z, H, W) uint32",
            "  cell_labels_2d.tif      (T, H, W)    uint32  — max-proj of tracked_labels",
            "  hypotheses_manifest.json",
            "  labelmaps/labelmap_*.tif",
        ],
    ),
    (
        "5_analysis/",
        [
            "Analysis outputs  [step 5]",
            "  graph.h5",
            "  topology.npz",
        ],
    ),
]

# Stage key → output folder (for the mapping table at the bottom of the TXT)
_STAGE_FOLDER: dict[str, str] = {
    "raw_import": "0_input/",
    "cellpose_cluster": "1_cellpose/",
    "nucleus_ultrack": "2_nucleus_ultrack/",
    "correction": "3_correction/",
    "cell_ultrack": "4_cell_ultrack/",
    "analysis": "5_analysis/",
}


def _write_pipeline_layout(
    output_root: Path,
    input_dir: Optional[Path],
) -> None:
    """Write PIPELINE_LAYOUT.txt at the project output root."""
    lines: List[str] = [
        "CellFlow Pipeline Directory Layout",
        "=" * 60,
        "",
        f"Created:        {date.today().isoformat()}",
        f"Project root:   {output_root}",
    ]
    if input_dir is not None:
        lines.append(f"Raw data input: {input_dir}")
    lines += [
        "",
        "<project_root>/",
        "├── project.json",
        "├── pipeline_schema.json",
        "├── PIPELINE_LAYOUT.txt            ← this file",
        "│",
        "└── pos##/                         (pos00, pos01, ...)",
        "    ├── pipeline_manifest.json     — stage run status and timestamps",
        "    ├── pipeline.log               — execution log (JSON-lines)",
        "    │",
    ]

    for i, (folder, detail_lines) in enumerate(_DIR_DESCRIPTIONS):
        is_last = i == len(_DIR_DESCRIPTIONS) - 1
        connector = "└──" if is_last else "├──"
        bar      = "   " if is_last else "│  "
        lines.append(f"    {connector} {folder}")
        for dl in detail_lines:
            lines.append(f"    {bar}   {dl}")
        if not is_last:
            lines.append(f"    {bar}")

    lines += [
        "",
        "Stage → directory mapping:",
        "",
    ]
    seen: set[str] = set()
    for stage_name in STAGE_ORDER:
        folder = _STAGE_FOLDER.get(stage_name)
        if folder is None:
            continue
        display = STAGE_DISPLAY_NAMES.get(stage_name, stage_name)
        key = f"{stage_name}:{folder}"
        if key not in seen:
            seen.add(key)
            lines.append(f"  {display:30s} → pos##/{folder}")

    lines += [
        "",
        "Notes:",
        "  - nuclear_labels_2d.tif is produced by the nucleus hypothesis/tracking",
        "    stage as a max-projection of tracked_labels.tif along Z; it is the",
        "    input to the correction step.",
        "  - nuclear_labels_corrected.tif is the final output of the correction",
        "    step; it is the input to cell Ultrack.",
        "  - Pixel size (µm) and time interval are read from acquisition",
        "    metadata during the Data Prep stage.",
        "  - Re-running a stage with 'Overwrite' replaces existing outputs.",
        "  - The manifest tracks the status and timestamps of each stage run.",
    ]

    txt = "\n".join(lines) + "\n"
    (output_root / "PIPELINE_LAYOUT.txt").write_text(txt)
