"""Unified path resolution for the CellFlow pipeline directory layout.

All stage packages must resolve paths through this module so that directory
names stay consistent across the entire pipeline.
"""
from __future__ import annotations

from pathlib import Path

# Authoritative stage-name → output-directory mapping.
# Stage packages must NOT hard-code these strings.
STAGE_DIRS: dict[str, str] = {
    "raw_import": "0_raw",
    "cellpose_nucleus": "1a_cellpose_nucleus",
    "cellpose_cell": "1b_cellpose_cell",
    "flow_watershed": "2_flow_watershed",
    "contours": "2b_contours",
    "tracking": "3_tracking",
    "graph_extraction": "4_analysis",
    "topology_analysis": "4_analysis",
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
