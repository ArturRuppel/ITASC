"""Post-segmentation database pruning helpers for Ultrack."""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import numpy as np

from cellflow.ultrack.config import TrackingConfig


def _node_from_pickle_value(value):
    """Return a node object from a DB value.

    Ultrack's ``MaybePickleType`` usually returns a decoded node object, but
    older/raw test fixtures may still yield pickled bytes.  Support both.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return pickle.loads(value)
    return value


def _circularity(area: int, mask: np.ndarray) -> float:
    """4π·area/perimeter² for a binary mask crop."""
    from skimage.measure import perimeter as skimage_perimeter

    p = skimage_perimeter(mask)
    if p == 0:
        return 0.0
    return 4 * np.pi * area / (p * p)


def _pruned_ids_for_timepoint(
    database_path: str,
    t: int,
    min_circularity: float,
) -> list[int]:
    """Return node IDs at timepoint *t* whose circularity is below threshold."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from ultrack.core.database import NodeDB

    engine = create_engine(database_path, hide_parameters=True)
    with Session(engine) as session:
        rows = session.query(NodeDB.id, NodeDB.area, NodeDB.pickle).where(
            NodeDB.t == t
        )
        pruned = []
        for node_id, area, blob in rows:
            node = _node_from_pickle_value(blob)
            if _circularity(area, node.mask) < min_circularity:
                pruned.append(node_id)
    return pruned


def prune_circularity_filtered_candidates(
    working_dir: str | Path,
    cfg: TrackingConfig,
    *,
    n_workers: int = 1,
) -> int:
    """Remove candidate nodes whose circularity is below ``cfg.min_circularity``.

    Runs per-timepoint work in parallel when *n_workers* > 1.  Returns the
    total number of removed nodes.
    """
    if cfg.min_circularity <= 0.0:
        return 0

    from sqlalchemy import create_engine, func, or_
    from sqlalchemy.orm import Session

    from cellflow.ultrack.stages.tracking import _build_ultrack_config
    from ultrack.core.database import NodeDB, OverlapDB
    from ultrack.core.linking.utils import clear_linking_data

    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)
    database_path = ultrack_cfg.data_config.database_path

    engine = create_engine(database_path, hide_parameters=True)
    with Session(engine) as session:
        timepoints: Sequence[int] = [
            t for t, in session.query(func.distinct(NodeDB.t))
        ]

    pruned_ids: list[int] = []

    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(
                    _pruned_ids_for_timepoint,
                    database_path,
                    t,
                    cfg.min_circularity,
                ): t
                for t in timepoints
            }
            for future in as_completed(futures):
                pruned_ids.extend(future.result())
    else:
        for t in timepoints:
            pruned_ids.extend(
                _pruned_ids_for_timepoint(database_path, t, cfg.min_circularity)
            )

    if pruned_ids:
        with Session(engine) as session:
            session.query(OverlapDB).where(
                or_(
                    OverlapDB.node_id.in_(pruned_ids),
                    OverlapDB.ancestor_id.in_(pruned_ids),
                )
            ).delete(synchronize_session=False)
            session.query(NodeDB).where(NodeDB.id.in_(pruned_ids)).delete(
                synchronize_session=False
            )
            session.commit()

        clear_linking_data(database_path)

    return len(pruned_ids)
