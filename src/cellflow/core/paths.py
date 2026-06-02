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


