"""Thin wrapper around ultrack.core.solve.processing.solve."""
from __future__ import annotations

from pathlib import Path
from collections.abc import Generator

from itasc.tracking_ultrack.config import TrackingConfig
from itasc.tracking_ultrack.ingest import _build_ultrack_config


def database_has_annotations(working_dir: str | Path) -> bool:
    """Return whether ``data.db`` contains REAL or FAKE node annotations."""
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, VarAnnotation

    db_path = Path(working_dir) / "data.db"
    if not db_path.exists():
        return False

    engine = sqla.create_engine(f"sqlite:///{db_path}")
    try:
        with Session(engine) as session:
            return bool(
                session.query(NodeDB.id)
                .where(NodeDB.node_annot.in_([VarAnnotation.REAL, VarAnnotation.FAKE]))
                .limit(1)
                .first()
            )
    finally:
        engine.dispose()


def run_solve(
    working_dir: str | Path,
    cfg: TrackingConfig,
    *,
    overwrite: bool = True,
    use_annotations: bool | None = True,
) -> Generator[tuple[int, int, str], None, None]:
    """Run the ILP solver, yielding (step, total, label) progress tuples."""
    from ultrack.core.solve.processing import solve

    total = 2
    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)
    if use_annotations is None:
        use_annotations = True

    yield (0, total, "Running ILP solver…")
    solve(ultrack_cfg, overwrite=overwrite, use_annotations=use_annotations)
    yield (total, total, "Solve done.")
