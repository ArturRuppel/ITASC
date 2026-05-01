"""Thin wrapper around ultrack.core.solve.processing.solve."""
from __future__ import annotations

from pathlib import Path
from typing import Generator

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.ingest import _build_ultrack_config


def run_solve(
    working_dir: str | Path,
    cfg: TrackingConfig,
    *,
    overwrite: bool = True,
    use_annotations: bool = False,
) -> Generator[tuple[int, int, str], None, None]:
    """Run the ILP solver, yielding (step, total, label) progress tuples."""
    from ultrack.core.solve.processing import solve

    total = 2
    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)

    yield (0, total, "Running ILP solver…")
    solve(ultrack_cfg, overwrite=overwrite, use_annotations=use_annotations)
    yield (total, total, "Solve done.")
