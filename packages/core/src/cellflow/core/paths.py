"""Unified path resolution for the CellFlow pipeline directory layout.

All stage packages must resolve paths through this module so that directory
names stay consistent across the entire pipeline.
"""
from __future__ import annotations

from pathlib import Path

# Authoritative stage-name → output-directory mapping.
# Stage packages must NOT hard-code these strings.
STAGE_DIRS: dict[str, str] = {
    # Step 0 — project config lives at the root; no stage directory.
    "raw_import":        "0_input",
    # Step 1 — Cellpose: two sub-folders inside a shared 1_cellpose/ parent.
    "cellpose_nucleus":  "1_cellpose/nucleus",
    "cellpose_cell":     "1_cellpose/cell",
    # Steps 2+3 — Contours (intermediate) and Ultrack share one directory.
    "contours":          "2_ultrack",
    "tracking":          "2_ultrack",
    # Step 4 — Correction produces a single file in its own folder.
    "correction":        "3_correction",
    # Step 5 — Nucleus-anchored cell segmentation (gravity flow).
    "cell_segmentation": "4_cell_segmentation",
    # Step 5b — Seeded watershed hypothesis sweep.
    "seeded_watershed":  "4_seeded_watershed",
    # Step 6 — Edge analysis.
    "graph_extraction":  "5_analysis",
    "topology_analysis": "5_analysis",
}


def pos_dir(root: Path | str, pos: int) -> Path:
    """Return ``<root>/pos<pos:02d>``."""
    return Path(root) / f"pos{pos:02d}"


def stage_dir(root: Path | str, pos: int, stage: str) -> Path:
    """Return the output directory for *stage* at position *pos*.

    Falls back to the stage name verbatim when it is not in ``STAGE_DIRS``.
    """
    dirname = STAGE_DIRS.get(stage, stage)
    return pos_dir(root, pos) / dirname


def manifest_path(root: Path | str, pos: int) -> Path:
    return pos_dir(root, pos) / "pipeline_manifest.json"


def schema_path(root: Path | str) -> Path:
    return Path(root) / "pipeline_schema.json"


def log_path(root: Path | str, pos: int) -> Path:
    return pos_dir(root, pos) / "pipeline.log"


def project_config_path(root: Path | str) -> Path:
    return Path(root) / "project.json"


def resolve_interface_path(
    root: Path | str,
    pos: int,
    template: str,
    stem: str = "",
) -> Path:
    """Expand an :class:`~cellflow.core.schema.InterfaceSpec` ``path_template``.

    The template may contain ``{pos:02d}`` and ``{stem}`` placeholders.
    """
    return Path(root) / template.format(pos=pos, stem=stem)
