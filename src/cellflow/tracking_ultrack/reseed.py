"""Re-solve with validated tracks: prune → solve → merge.

The validate-and-resolve loop lets the user lock in confirmed cell tracks
(validated_tracks), then re-solve the rest of the stack — possibly with
tweaked parameters — without losing the validated work.

Mechanism:
1. `prune_validated_overlaps` — remove any hypothesis node that overlaps any
   validated mask from NodeDB (and cascade-delete referencing OverlapDB rows),
   so the solver never competes with validated cells.
2. Normal Ultrack link → solve → export against the pruned DB.
3. `merge_validated_into_export` — paste validated masks back onto the exported
   labelmap with fresh unique track IDs, overwriting any stray pixels.

The contract for `validated_tracks` is `dict[int, set[int]]` where keys are
cell IDs and values are the set of frames at which that cell is validated.
This is independent of the on-disk JSON shape (which is managed separately by
`cellflow.database.validation`).
"""
from __future__ import annotations

import logging
import pickle
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import find_objects

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.ingest import _build_ultrack_config

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: build per-frame validated-mask index
# ---------------------------------------------------------------------------

def _build_frame_masks(
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
) -> dict[int, list[tuple[int, np.ndarray, tuple[int, int, int, int]]]]:
    """Build a per-frame index of validated masks.

    Returns
    -------
    dict mapping frame index ``t`` to a list of
    ``(cell_id, bool_mask_crop, (y0, x0, y1, x1))`` tuples, one per validated
    cell at that frame.  Both ``bool_mask_crop`` and the bbox refer to the
    cropped region (exclusive-max convention).

    Only frames where at least one cell is validated are included.
    The 2D spatial slice is taken from ``tracked_labels[t]`` (shape Y×X).
    For a 4-D input ``(T, Z, Y, X)`` the z-axis is max-projected to give a
    2D footprint for intersection tests (conservative: any z-slice counts).
    """
    frame_index: dict[int, list[tuple[int, np.ndarray, tuple[int, int, int, int]]]] = {}

    for cell_id, frames in validated_tracks.items():
        for t in frames:
            t = int(t)
            frame_vol = tracked_labels[t]  # (Y, X) or (Z, Y, X)

            # Project to 2D for intersection (max over Z if 3D)
            if frame_vol.ndim == 3:
                frame_2d = (frame_vol == cell_id).any(axis=0)
            else:
                frame_2d = (frame_vol == cell_id)

            if not frame_2d.any():
                # Cell absent at this frame — skip silently
                continue

            # Tight bbox around the mask
            rows = np.where(frame_2d.any(axis=1))[0]
            cols = np.where(frame_2d.any(axis=0))[0]
            y0, y1 = int(rows[0]), int(rows[-1]) + 1
            x0, x1 = int(cols[0]), int(cols[-1]) + 1
            mask_crop = frame_2d[y0:y1, x0:x1]

            if t not in frame_index:
                frame_index[t] = []
            frame_index[t].append((cell_id, mask_crop, (y0, x0, y1, x1)))

    return frame_index


# ---------------------------------------------------------------------------
# 1. prune_validated_overlaps
# ---------------------------------------------------------------------------

def prune_validated_overlaps(
    working_dir: str | Path,
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
) -> int:
    """Delete NodeDB rows that overlap any validated mask and cascade to OverlapDB.

    Parameters
    ----------
    working_dir:
        Ultrack working directory containing ``data.db``.
    validated_tracks:
        ``{cell_id: {frames}}`` — which frames each validated cell spans.
    tracked_labels:
        Current corrected labelmap, shape ``(T, Y, X)`` or ``(T, Z, Y, X)``.

    Returns
    -------
    int
        Number of NodeDB rows deleted.

    Notes
    -----
    Conflict criterion: **any pixel intersection** between a hypothesis node's
    2D footprint and any validated cell mask at the same frame counts as a
    conflict.  The node (and all OverlapDB rows referencing it) is deleted.

    The 3D node masks stored in NodeDB are stored as ``(1, h, w)`` crops
    (the ingest convention in ``ingest.py``).  We squeeze the Z=1 dimension
    for the 2D intersection test.  True 3D (Z > 1) is currently not supported;
    the bbox test gracefully falls back to a bounding-box-only check in that
    case.

    Both NodeDB and referencing OverlapDB rows are deleted in a single
    SQLite transaction to preserve DB consistency.
    """
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB, OverlapDB

    if not validated_tracks:
        return 0

    working_dir = Path(working_dir)
    db_url = f"sqlite:///{working_dir / 'data.db'}"
    engine = sqla.create_engine(db_url)

    frame_masks = _build_frame_masks(validated_tracks, tracked_labels)
    if not frame_masks:
        return 0

    total_deleted = 0

    with Session(engine) as session:
        for t, val_cells in frame_masks.items():
            # Query all nodes at this frame (id + pickle blob)
            rows = (
                session.query(NodeDB.id, NodeDB.pickle)
                .where(NodeDB.t == t)
                .all()
            )
            if not rows:
                continue

            conflict_ids: list[int] = []

            for node_id_val, node_blob in rows:
                # Deserialize; NodeDB.pickle is a MaybePickleType that
                # returns a Node object on read.
                if isinstance(node_blob, (bytes, memoryview)):
                    try:
                        node = pickle.loads(bytes(node_blob))
                    except Exception:
                        LOG.warning(
                            "Could not unpickle node id=%s at t=%s — skipping",
                            node_id_val, t,
                        )
                        continue
                else:
                    # SQLAlchemy MaybePickleType already deserialized it
                    node = node_blob

                bbox = np.asarray(node.bbox)
                ndim = len(bbox) // 2
                if ndim == 3:
                    # bbox = [z0, y0, x0, z1, y1, x1]
                    ny0, nx0, ny1, nx1 = int(bbox[1]), int(bbox[2]), int(bbox[4]), int(bbox[5])
                    node_mask = getattr(node, "mask", None)
                    if node_mask is not None:
                        node_mask = np.asarray(node_mask, dtype=bool)
                        if node_mask.shape[0] == 1:
                            node_mask_2d = node_mask[0]  # (h, w)
                        else:
                            # Multi-Z: project
                            node_mask_2d = node_mask.any(axis=0)
                    else:
                        node_mask_2d = None
                elif ndim == 2:
                    # bbox = [y0, x0, y1, x1]
                    ny0, nx0, ny1, nx1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    node_mask = getattr(node, "mask", None)
                    if node_mask is not None:
                        node_mask_2d = np.asarray(node_mask, dtype=bool)
                        if node_mask_2d.ndim == 3 and node_mask_2d.shape[0] == 1:
                            node_mask_2d = node_mask_2d[0]
                    else:
                        node_mask_2d = None
                else:
                    LOG.warning(
                        "Unexpected bbox ndim=%s for node id=%s — skipping",
                        ndim, node_id_val,
                    )
                    continue

                # Check against every validated cell at this frame
                is_conflict = False
                for _cell_id, val_crop, (vy0, vx0, vy1, vx1) in val_cells:
                    # Fast bbox intersection test first
                    if ny1 <= vy0 or vy1 <= ny0 or nx1 <= vx0 or vx1 <= nx0:
                        continue  # bboxes don't overlap → no conflict

                    # Compute the overlap region in image coordinates
                    oy0, oy1 = max(ny0, vy0), min(ny1, vy1)
                    ox0, ox1 = max(nx0, vx0), min(nx1, vx1)

                    if node_mask_2d is None:
                        # No mask data — treat bbox overlap as conflict
                        is_conflict = True
                        break

                    # Crop both masks to the overlap region
                    nm_crop = node_mask_2d[
                        oy0 - ny0: oy1 - ny0,
                        ox0 - nx0: ox1 - nx0,
                    ]
                    vm_crop = val_crop[
                        oy0 - vy0: oy1 - vy0,
                        ox0 - vx0: ox1 - vx0,
                    ]
                    if nm_crop.shape != vm_crop.shape:
                        # Shape mismatch due to off-by-one — treat as conflict
                        is_conflict = True
                        break
                    if np.any(nm_crop & vm_crop):
                        is_conflict = True
                        break

                if is_conflict:
                    conflict_ids.append(int(node_id_val))

            if not conflict_ids:
                continue

            # Delete OverlapDB rows referencing any conflict node, then nodes —
            # all in one transaction (the session is autocommit=False by default).
            (
                session.query(OverlapDB)
                .where(
                    sqla.or_(
                        OverlapDB.node_id.in_(conflict_ids),
                        OverlapDB.ancestor_id.in_(conflict_ids),
                    )
                )
                .delete(synchronize_session=False)
            )
            (
                session.query(NodeDB)
                .where(NodeDB.id.in_(conflict_ids))
                .delete(synchronize_session=False)
            )
            total_deleted += len(conflict_ids)

        session.commit()

    LOG.info("prune_validated_overlaps: deleted %d NodeDB rows", total_deleted)
    return total_deleted


# ---------------------------------------------------------------------------
# 2. merge_validated_into_export
# ---------------------------------------------------------------------------

def merge_validated_into_export(
    exported_labels: np.ndarray,
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    """Paste validated masks back onto the solver's exported labelmap.

    Each validated ``cell_id`` gets a **single fresh track ID** (starting from
    ``exported_labels.max() + 1``) so that all frames of one validated cell
    form a continuous track in the output.  Validated pixels overwrite any
    conflicting content in the exported labelmap (the pruning was aggressive
    but the solver may still produce stray segments near validated cells;
    validated cells always win).

    Parameters
    ----------
    exported_labels:
        Labelmap produced by Ultrack export, shape ``(T, Y, X)`` or
        ``(T, Z, Y, X)``.  Modified **in place** and returned.
    validated_tracks:
        ``{cell_id: {frames}}`` from the validation store.
    tracked_labels:
        Current corrected labelmap (same spatial shape as ``exported_labels``).
        The source of ground-truth masks.

    Returns
    -------
    tuple[np.ndarray, dict[int, int]]
        ``(merged_labelmap, id_map)`` where ``id_map`` maps each original
        ``old_cell_id`` to the ``new_cell_id`` assigned in the output.
        The labelmap is the same array as ``exported_labels``, modified in place.
    """
    if not validated_tracks:
        return exported_labels, {}

    next_id = int(exported_labels.max()) + 1
    id_map: dict[int, int] = {}

    for cell_id, frames in validated_tracks.items():
        if not frames:
            continue

        new_track_id = next_id
        id_map[cell_id] = new_track_id
        next_id += 1

        for t in sorted(frames):
            t = int(t)
            frame_src = tracked_labels[t]

            if frame_src.ndim == 3:
                mask_3d = frame_src == cell_id
                if not mask_3d.any():
                    continue
                exported_labels[t][mask_3d] = 0      # clear first
                exported_labels[t][mask_3d] = new_track_id
            else:
                mask_2d = frame_src == cell_id
                if not mask_2d.any():
                    continue
                exported_labels[t][mask_2d] = 0      # clear first
                exported_labels[t][mask_2d] = new_track_id

    return exported_labels, id_map


# ---------------------------------------------------------------------------
# 3. resolve_with_validation
# ---------------------------------------------------------------------------

def resolve_with_validation(
    working_dir: str | Path,
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
    cfg: TrackingConfig,
    progress_cb: "Callable[[str], None] | None" = None,
) -> tuple[np.ndarray, dict[int, int]]:
    """Prune validated overlaps, re-solve, and merge validated cells back in.

    Orchestration:

    1. :func:`prune_validated_overlaps` — remove hypothesis nodes that overlap
       any validated mask so the solver never competes with them.
    2. ``ultrack.core.linking.processing.link`` — build candidate links in the
       pruned NodeDB.
    3. ``ultrack.core.solve.processing.solve`` — run the ILP solver.
    4. :func:`~cellflow.tracking_ultrack.export.export_tracked_labels` — export
       the solver's result to a temporary file, load it back as a numpy array.
    5. :func:`merge_validated_into_export` — paste validated cells back with
       fresh unique track IDs.

    Parameters
    ----------
    working_dir:
        Ultrack working directory that already contains a populated ``data.db``
        (NodeDB + OverlapDB) from a prior ingestion run.
    validated_tracks:
        ``{cell_id: {frames}}`` — the locked-in validated tracks.
    tracked_labels:
        Current corrected labelmap, ``(T, Y, X)`` or ``(T, Z, Y, X)``.
    cfg:
        Tracking configuration (ILP parameters, linking options, …).
    progress_cb:
        Optional callable that receives a human-readable status string at each
        stage.  Called synchronously from this function (safe from a worker
        thread).

    Returns
    -------
    tuple[np.ndarray, dict[int, int]]
        ``(merged_labelmap, id_map)`` where ``id_map`` maps each original
        validated ``cell_id`` to its newly assigned track ID in the output.
    """
    from collections.abc import Callable  # noqa: F401 — used in annotation above

    from ultrack.core.linking.processing import link
    from ultrack.core.linking.utils import clear_linking_data
    from ultrack.core.solve.processing import solve

    from cellflow.tracking_ultrack.export import export_tracked_labels

    def _notify(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    working_dir = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, working_dir)

    # Step 1: prune
    _notify("Pruning validated overlaps…")
    n_pruned = prune_validated_overlaps(working_dir, validated_tracks, tracked_labels)
    LOG.info("resolve_with_validation: pruned %d nodes", n_pruned)

    # Step 2: link
    _notify("Linking hypotheses…")
    clear_linking_data(ultrack_cfg.data_config.database_path)
    try:
        link(ultrack_cfg, overwrite=False)
    except (ValueError, IndexError) as exc:
        # Ultrack's linker raises ValueError / IndexError when a frame is empty
        # (e.g. all nodes were pruned).  Fall back to an all-zero labelmap so
        # that step 5 can still paste the validated cells back.
        LOG.warning(
            "resolve_with_validation: linking failed (%s: %s) — "
            "falling back to empty labelmap for solver output",
            type(exc).__name__, exc,
        )
        exported = np.zeros_like(tracked_labels, dtype=np.uint32)
        _notify("Merging validated cells…")
        exported, id_map = merge_validated_into_export(exported, validated_tracks, tracked_labels)
        return exported, id_map

    # Step 3: solve
    _notify("Solving ILP…")
    solve(ultrack_cfg, overwrite=True)

    # Step 4: export to a temporary file, load as numpy
    _notify("Exporting tracks…")
    with tempfile.TemporaryDirectory(prefix="cellflow_resolve_") as tmpdir:
        out_path = Path(tmpdir) / "tracked_labels.tif"
        exported = export_tracked_labels(working_dir, cfg, out_path)

    # Ensure uint32 and (T, Y, X) or (T, Z, Y, X) consistent with tracked_labels
    exported = np.asarray(exported, dtype=np.uint32)

    # Step 5: merge validated cells back
    _notify("Merging validated cells…")
    exported, id_map = merge_validated_into_export(exported, validated_tracks, tracked_labels)

    return exported, id_map
