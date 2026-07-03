"""The final-output commit contract.

The numbered stage dirs (``2_nucleus/``, ``3_cell/``) hold re-runnable *working*
label files; :func:`promote_labels` copies one up to a stable *committed* name in
the position base folder (``nucleus_labels.tif`` / ``cell_labels.tif``), the
downstream-stable output that discovery defaults to. :func:`commit_state` reports
where a position sits in that working-vs-committed split — the signal the
workflow widgets surface and the catalog status rail reads.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import tifffile

CommitState = str  # one of: "missing" | "uncommitted" | "committed" | "stale"


def promote_labels(src: Path | str, dst: Path | str) -> Path:
    """Promote a working label file *src* to the committed output *dst*.

    Copies bytes verbatim (the working file is already a dtype-correct label
    image), creating ``dst``'s parent and overwriting any existing ``dst``.
    Raises ``FileNotFoundError`` if *src* is missing and ``ValueError`` if it is
    not an integer-typed label image.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.is_file():
        raise FileNotFoundError(f"No working label file to commit: {src}")
    with tifffile.TiffFile(str(src)) as tf:
        dtype = tf.series[0].dtype
    if not np.issubdtype(dtype, np.integer):
        raise ValueError(
            f"Not a label image (dtype {dtype}); refusing to commit {src}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def commit_state(working: Path | str, committed: Path | str) -> CommitState:
    """Return the working-vs-committed state for one stage output.

    * ``"missing"`` — no working file yet.
    * ``"uncommitted"`` — working file exists but has not been committed.
    * ``"committed"`` — committed copy is at least as new as the working file.
    * ``"stale"`` — the working file was re-run after the last commit (strictly
      newer), so the committed copy no longer reflects it.
    """
    working = Path(working)
    committed = Path(committed)
    if not working.is_file():
        return "missing"
    if not committed.is_file():
        return "uncommitted"
    if working.stat().st_mtime > committed.stat().st_mtime:
        return "stale"
    return "committed"
