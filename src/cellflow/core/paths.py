"""Unified path resolution for the CellFlow pipeline directory layout."""
from __future__ import annotations

from pathlib import Path

# Authoritative stage-name → output-directory mapping.
STAGE_DIRS: dict[str, str] = {
    "raw_import": "0_input",
    "cellpose": "1_cellpose",
    "nucleus": "2_nucleus",
    "cell": "3_cell",
    "contact_analysis": "4_contact_analysis",
}


def pos_dir(root: Path | str, pos: int) -> Path:
    """Return <root>/pos<pos:02d>."""
    return Path(root) / f"pos{pos:02d}"


def stage_dir(root: Path | str, pos: int, stage: str) -> Path:
    """Return the output directory for *stage* at position *pos*."""
    dirname = STAGE_DIRS.get(stage, stage)
    return pos_dir(root, pos) / dirname


def log_path(root: Path | str, pos: int) -> Path:
    """Return the path to the pipeline log for a position."""
    return pos_dir(root, pos) / "pipeline.log"
