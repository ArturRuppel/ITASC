"""Qt-free per-position pipeline status — the source the catalog rail reads.

A position moves through four human-in-the-loop stages (Cellpose → Ultrack
nucleus tracking → cell workflow → contact analysis). :func:`position_stage_status`
reports how far one position has progressed by header-only file existence + mtime
checks (no pixel decode), mirroring the on-disk done-signals:

* **cellpose** (2-state) — the nucleus divergence maps
  (``1_cellpose/nucleus_foreground.tif`` + ``_contours.tif``) exist.
* **nucleus** / **cell** (3-state) — the working-vs-committed split from the P2
  commit contract (:func:`cellflow.core.commit.commit_state`): a working file in
  the numbered stage dir, optionally promoted to the base-folder ``*_labels.tif``.
* **contacts** (2-state) — the contact-analysis ``.h5`` exists.

The widget layer calls this per row on refresh and maps the returned states onto
dot rendering. A row with no canonical position root (a hand-authored catalog CSV
pointing at scattered paths) passes ``None`` and gets ``unknown`` for every stage.
"""
from __future__ import annotations

from pathlib import Path

from cellflow.contact_analysis.catalog import CONTACT_ANALYSIS_RELPATH
from cellflow.core.commit import commit_state
from cellflow.napari._paths import NucleusArtifactPaths

# Stage keys, ordered left→right as the rail renders them.
STAGE_CELLPOSE = "cellpose"
STAGE_NUCLEUS = "nucleus"
STAGE_CELL = "cell"
STAGE_CONTACTS = "contacts"
STAGES: tuple[str, ...] = (STAGE_CELLPOSE, STAGE_NUCLEUS, STAGE_CELL, STAGE_CONTACTS)

# State vocabulary. The widget maps each onto a dot glyph:
MISSING = "missing"  # empty ring — nothing on disk yet
WORKING = "working"  # hollow — working file present, not committed (3-state only)
DONE = "done"        # filled — final artifact present
STALE = "stale"      # committed, but the working file is newer (3-state only)
UNKNOWN = "unknown"  # grey — no canonical position root to derive from

# commit_state's vocabulary → the rail's.
_COMMIT_TO_STATE: dict[str, str] = {
    "missing": MISSING,
    "uncommitted": WORKING,
    "committed": DONE,
    "stale": STALE,
}


def position_stage_status(pos_dir: Path | str | None) -> dict[str, str]:
    """Return each pipeline stage's status for one canonical position directory.

    ``pos_dir`` of ``None`` yields ``unknown`` for every stage (a hand-authored
    catalog row with no canonical root). See the module docstring for the
    per-stage done-signals and the state vocabulary.
    """
    if pos_dir is None:
        return {stage: UNKNOWN for stage in STAGES}

    paths = NucleusArtifactPaths(Path(pos_dir))
    cellpose = (
        DONE
        if paths.nucleus_foreground.is_file() and paths.nucleus_contours.is_file()
        else MISSING
    )
    nucleus = _COMMIT_TO_STATE[commit_state(paths.tracked, paths.nucleus_labels)]
    cell = _COMMIT_TO_STATE[commit_state(paths.cell_tracked, paths.cell_labels)]
    contacts = DONE if (paths.pos_dir / CONTACT_ANALYSIS_RELPATH).is_file() else MISSING

    return {
        STAGE_CELLPOSE: cellpose,
        STAGE_NUCLEUS: nucleus,
        STAGE_CELL: cell,
        STAGE_CONTACTS: contacts,
    }
