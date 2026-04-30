"""Re-solve with validated tracks using annotated Ultrack seed nodes.

The validate-and-resolve loop lets the user lock in confirmed cell tracks
(validated_tracks), then re-solve the rest of the stack — possibly with
tweaked parameters — without losing the validated work.

Mechanism:
1. Rebuild Ultrack's NodeDB from ``hypotheses.h5`` in a fresh temporary
   working directory.
2. Insert validated masks as annotated ``REAL`` nodes and mark overlapping
   candidates ``FAKE``.
3. Score candidate node probabilities from image quality plus proximity to
   validated seed nodes.
4. Link → solve with annotations → export directly from Ultrack.

The contract for `validated_tracks` is `dict[int, set[int]]` where keys are
cell IDs and values are the set of frames at which that cell is validated.
This is independent of the on-disk JSON shape (which is managed separately by
`cellflow.database.validation`).

`prune_validated_overlaps`, `_build_frame_masks`, and
`merge_validated_into_export` are kept for existing unit tests and ad hoc use;
the live pipeline no longer calls them.
"""
from __future__ import annotations

from collections.abc import Callable
import logging
import pickle
import tempfile
from pathlib import Path

import numpy as np

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.export import export_tracked_labels
from cellflow.tracking_ultrack.ingest import ingest_hypotheses_to_db
from cellflow.tracking_ultrack.linking import run_linking
from cellflow.tracking_ultrack.seed_prior import write_seed_prior_node_probs
from cellflow.tracking_ultrack.solve import run_solve
from cellflow.tracking_ultrack.validation_nodes import inject_validated_nodes

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


def _build_frame_forbidden_masks(
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Return ``{t: bool_array (Y, X)}`` — union of all validated cell footprints per frame.

    Frames with no validated cells are omitted.  For 4-D ``(T, Z, Y, X)`` input
    the z-axis is max-projected to give a conservative 2D footprint.
    """
    if not validated_tracks:
        return {}

    n_frames = tracked_labels.shape[0]
    if tracked_labels.ndim == 4:
        Y, X = tracked_labels.shape[2], tracked_labels.shape[3]
    else:
        Y, X = tracked_labels.shape[1], tracked_labels.shape[2]

    # Group cell IDs by frame so we touch each frame slice once.
    cells_by_frame: dict[int, list[int]] = {}
    for cell_id, frames in validated_tracks.items():
        for t in frames:
            cells_by_frame.setdefault(int(t), []).append(int(cell_id))

    out: dict[int, np.ndarray] = {}
    for t, cell_ids in cells_by_frame.items():
        if t < 0 or t >= n_frames:
            continue
        frame_vol = tracked_labels[t]
        if frame_vol.ndim == 3:
            frame_2d = np.isin(frame_vol, cell_ids).any(axis=0)
        else:
            frame_2d = np.isin(frame_vol, cell_ids)
        if frame_2d.any():
            out[t] = np.ascontiguousarray(frame_2d, dtype=bool)
    return out


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

def _validated_export_id_map(
    exported_labels: np.ndarray,
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
) -> dict[int, int]:
    id_map: dict[int, int] = {}
    for cell_id, frames in validated_tracks.items():
        seen: list[int] = []
        for t in sorted(frames):
            t = int(t)
            frame_src = tracked_labels[t]
            frame_out = exported_labels[t]
            if frame_src.ndim == 3:
                mask = frame_src == cell_id
            else:
                mask = frame_src == cell_id
            if not mask.any():
                continue
            if mask.ndim == 2 and frame_out.ndim == 3:
                out_values = frame_out[:, mask]
            elif mask.ndim == 3 and frame_out.ndim == 2:
                out_values = frame_out[mask.any(axis=0)]
            else:
                out_values = frame_out[mask]
            out_values = out_values[out_values != 0]
            if out_values.size:
                values, counts = np.unique(out_values, return_counts=True)
                seen.append(int(values[int(np.argmax(counts))]))
        if seen:
            values, counts = np.unique(np.asarray(seen, dtype=np.int64), return_counts=True)
            id_map[int(cell_id)] = int(values[int(np.argmax(counts))])
    return id_map


def resolve_with_validation(
    hypotheses_path: str | Path,
    validated_tracks: dict[int, set[int]],
    tracked_labels: np.ndarray,
    cfg: TrackingConfig,
    progress_cb: Callable[[str], None] | None = None,
    *,
    intensity_image_path: str | Path,
) -> tuple[np.ndarray, dict[int, int]]:
    """Re-ingest hypotheses, inject validated nodes, solve with annotations.

    Each call creates a **fresh temporary working directory** for Ultrack and
    discards it on exit.  This keeps the resolve idempotent (no carry-over of
    DB state, freelist bloat, or stale WAL files between runs).

    Orchestration:

    0. :func:`~cellflow.tracking_ultrack.ingest.ingest_hypotheses_to_db` —
       build the NodeDB/OverlapDB from ``hypotheses.h5`` in a fresh temp dir.
    1. :func:`~cellflow.tracking_ultrack.validation_nodes.inject_validated_nodes`
       — add validated masks as ``REAL`` nodes and suppress overlaps.
    2. :func:`~cellflow.tracking_ultrack.seed_prior.write_seed_prior_node_probs`
       — score candidate nodes from the nucleus zavg image and seed affinity.
    3. :func:`~cellflow.tracking_ultrack.linking.run_linking` — build candidate
       links.
    4. :func:`~cellflow.tracking_ultrack.solve.run_solve` — run the ILP solver
       with annotations enabled.
    5. :func:`~cellflow.tracking_ultrack.export.export_tracked_labels` — export
       the solver's result to a temporary file, load it back as a numpy array.

    Parameters
    ----------
    hypotheses_path:
        Path to ``hypotheses.h5`` — the source of truth for the rebuild.
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
    intensity_image_path:
        Path to the nucleus z-average image used to score candidate quality.

    Returns
    -------
    tuple[np.ndarray, dict[int, int]]
        ``(exported_labelmap, id_map)`` where ``id_map`` maps each original
        validated ``cell_id`` to the exported ID that covers its validated
        pixels most often.
    """
    def _notify(msg: str) -> None:
        if progress_cb is not None:
            progress_cb(msg)

    hypotheses_path = Path(hypotheses_path)
    if not validated_tracks:
        return np.asarray(tracked_labels, dtype=np.uint32).copy(), {}

    with tempfile.TemporaryDirectory(prefix="cellflow_resolve_workdir_") as tmp_workdir:
        working_dir = Path(tmp_workdir)

        _notify("Ingesting hypotheses…")
        ingest_hypotheses_to_db(hypotheses_path, working_dir, cfg, overwrite=True)

        _notify("Injecting validated masks…")
        injection = inject_validated_nodes(working_dir, validated_tracks, tracked_labels, cfg)
        if injection.skipped_missing:
            _notify(
                f"Skipped {injection.skipped_missing} validated cell-frame(s) "
                "missing from tracked labels."
            )
        if injection.inserted == 0:
            raise ValueError("No validated masks could be injected; resolve aborted before solve.")
        _notify(
            f"Injected {injection.inserted} validated node(s); "
            f"marked {injection.faked} overlapping candidate(s) false."
        )

        _notify("Scoring node probabilities…")
        score_report = write_seed_prior_node_probs(working_dir, intensity_image_path, cfg)
        _notify(
            f"Scored {score_report.scored} candidate node(s) from "
            f"{score_report.seeds} validated seed node(s)."
        )

        _notify("Linking hypotheses…")
        for _step, _total, label in run_linking(working_dir, cfg):
            _notify(label)

        _notify("Solving ILP with annotations…")
        for _step, _total, label in run_solve(
            working_dir,
            cfg,
            overwrite=True,
            use_annotations=True,
        ):
            _notify(label)

        _notify("Exporting tracks…")
        with tempfile.TemporaryDirectory(prefix="cellflow_resolve_export_") as tmpdir:
            out_path = Path(tmpdir) / "tracked_labels.tif"
            exported = export_tracked_labels(working_dir, cfg, out_path)

    exported = np.asarray(exported, dtype=np.uint32)
    id_map = _validated_export_id_map(exported, validated_tracks, tracked_labels)
    return exported, id_map
