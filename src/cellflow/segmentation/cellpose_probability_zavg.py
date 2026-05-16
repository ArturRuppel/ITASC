"""Precompute z-averaged Cellpose probability maps from logits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


CELLPOSE_DIR = "1_cellpose"
CELL_PROB_3DT = "cell_prob_3dt.tif"
NUCLEUS_PROB_3DT = "nucleus_prob_3dt.tif"
CELL_PROB_ZAVG = "cell_prob_zavg.tif"
NUCLEUS_PROB_ZAVG = "nucleus_prob_zavg.tif"


@dataclass(frozen=True)
class CellposeProbabilityZavgResult:
    position_dir: Path
    wrote_cell: bool
    wrote_nucleus: bool
    skipped: bool
    message: str


def sigmoid_z_average(stack: np.ndarray) -> np.ndarray:
    """Apply sigmoid to Cellpose logits, then average across z."""
    arr = np.asarray(stack, dtype=np.float32)
    arr = np.clip(arr, -88.0, 88.0)
    probs = 1.0 / (1.0 + np.exp(-arr))
    if probs.ndim == 4:
        return probs.mean(axis=1, dtype=np.float32).astype(np.float32, copy=False)
    if probs.ndim == 3:
        return probs.mean(axis=0, dtype=np.float32).astype(np.float32, copy=False)
    if probs.ndim == 2:
        return probs.astype(np.float32, copy=False)
    raise ValueError(
        "Expected probability logits shaped TZYX, ZYX, or YX; "
        f"got {tuple(probs.shape)}."
    )


def write_cellpose_probability_zavgs_for_position(
    position_dir: str | Path,
    *,
    overwrite: bool = True,
) -> CellposeProbabilityZavgResult:
    pos_dir = Path(position_dir)
    cellpose_dir = pos_dir / CELLPOSE_DIR
    if not cellpose_dir.is_dir():
        return CellposeProbabilityZavgResult(
            position_dir=pos_dir,
            wrote_cell=False,
            wrote_nucleus=False,
            skipped=True,
            message="missing 1_cellpose directory",
        )

    wrote: list[str] = []
    missing: list[str] = []
    skipped_existing: list[str] = []
    wrote_cell = _write_one(
        cellpose_dir / CELL_PROB_3DT,
        cellpose_dir / CELL_PROB_ZAVG,
        overwrite=overwrite,
        wrote=wrote,
        missing=missing,
        skipped_existing=skipped_existing,
    )
    wrote_nucleus = _write_one(
        cellpose_dir / NUCLEUS_PROB_3DT,
        cellpose_dir / NUCLEUS_PROB_ZAVG,
        overwrite=overwrite,
        wrote=wrote,
        missing=missing,
        skipped_existing=skipped_existing,
    )

    parts: list[str] = []
    if wrote:
        parts.append("wrote " + ", ".join(wrote))
    if skipped_existing:
        parts.append("kept existing " + ", ".join(skipped_existing))
    if missing:
        parts.append("missing " + ", ".join(missing))
    message = "; ".join(parts) if parts else "no inputs found"
    return CellposeProbabilityZavgResult(
        position_dir=pos_dir,
        wrote_cell=wrote_cell,
        wrote_nucleus=wrote_nucleus,
        skipped=False,
        message=message,
    )


def write_cellpose_probability_zavgs_for_root(
    root: str | Path,
    *,
    overwrite: bool = True,
) -> list[CellposeProbabilityZavgResult]:
    root_path = Path(root)
    position_dirs = _position_dirs(root_path)
    return [
        write_cellpose_probability_zavgs_for_position(
            position_dir,
            overwrite=overwrite,
        )
        for position_dir in position_dirs
    ]


def _position_dirs(root: Path) -> list[Path]:
    if (root / CELLPOSE_DIR).is_dir():
        return [root]
    return sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and (child / CELLPOSE_DIR).is_dir()
    )


def _write_one(
    source: Path,
    target: Path,
    *,
    overwrite: bool,
    wrote: list[str],
    missing: list[str],
    skipped_existing: list[str],
) -> bool:
    if not source.exists():
        missing.append(source.name)
        return False
    if target.exists() and not overwrite:
        skipped_existing.append(target.name)
        return False
    data = sigmoid_z_average(tifffile.imread(str(source)))
    tifffile.imwrite(target, data.astype(np.float32, copy=False))
    wrote.append(target.name)
    return True
