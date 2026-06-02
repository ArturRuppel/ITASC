"""Re-solve with validated tracks using annotated Ultrack seed nodes.

The validate-and-resolve loop lets the user lock in confirmed cell tracks
(validated_tracks), then re-solve the rest of the stack — possibly with
tweaked parameters — without losing the validated work.

The contract for `validated_tracks` is `dict[int, set[int]]` where keys are
cell IDs and values are the set of frames at which that cell is validated.
This is independent of the on-disk JSON shape (which is managed separately by
`cellflow.database.validation`).

`prune_validated_overlaps` and `_build_frame_masks` are kept for existing
unit tests and ad hoc use.  `merge_validated_into_export` is called at the
end of `resolve_with_canonical_segment` to paste validated IDs back.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np


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
    Conflict criterion: **any pixel intersection** between a candidate node's
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

    Validated ``cell_id`` values are reserved and pasted back with their
    original IDs so the validation store remains stable across resolve runs.
    If the solver independently used a reserved validated ID elsewhere, those
    solver pixels are moved to a fresh ID before the validated mask is pasted.
    Validated pixels overwrite any conflicting content in the exported labelmap.

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
        ``(merged_labelmap, id_map)`` where ``id_map`` maps original validated
        IDs to changed output IDs. The normal path preserves validated IDs, so
        ``id_map`` is usually empty. The labelmap is the same array as
        ``exported_labels``, modified in place.
    """
    if not validated_tracks:
        return exported_labels, {}

    id_map: dict[int, int] = {}
    reserved_ids = {int(cell_id) for cell_id, frames in validated_tracks.items() if frames}
    used_fresh_ids = set(int(v) for v in np.unique(exported_labels))

    def _next_fresh_id() -> int:
        next_id = int(exported_labels.max()) + 1
        while next_id in reserved_ids or next_id in used_fresh_ids:
            next_id += 1
        used_fresh_ids.add(next_id)
        return next_id

    # Build validated_masks: per cell_id, list of (t, mask) pairs present in tracked_labels
    validated_masks: dict[int, list[tuple[int, np.ndarray]]] = {}
    for cell_id, frames in validated_tracks.items():
        cell_id = int(cell_id)
        if not frames:
            continue
        present: list[tuple[int, np.ndarray]] = []
        for t in sorted(frames):
            t = int(t)
            if t < 0 or t >= tracked_labels.shape[0] or t >= exported_labels.shape[0]:
                continue
            frame_src = tracked_labels[t]
            mask = np.asarray(frame_src == cell_id)
            if mask.any():
                present.append((t, mask))
        if present:
            validated_masks[cell_id] = present

    # Build solver_track_remap: solver_id -> validated cell_id
    # Inspect each validated frame to find the dominant solver track covering that mask region,
    # then propagate that solver track's ID to the validated cell_id everywhere in the export.
    solver_track_remap: dict[int, int] = {}
    for cell_id in sorted(validated_masks.keys()):
        present_masks = validated_masks[cell_id]
        dominant_solver_id: int | None = None
        best_count = 0
        for t, mask in present_masks:
            if exported_labels.ndim == 4:
                # (T, Z, Y, X): project mask to 3D for indexing
                frame_exp = exported_labels[t]
                if mask.ndim == 2:
                    solver_pixels = frame_exp[:, mask]
                elif mask.ndim == 3:
                    solver_pixels = frame_exp[mask]
                else:
                    continue
            else:
                frame_exp = exported_labels[t]
                if mask.ndim == frame_exp.ndim:
                    solver_pixels = frame_exp[mask]
                elif mask.ndim == 2 and frame_exp.ndim == 3:
                    solver_pixels = frame_exp[:, mask]
                elif mask.ndim == 3 and frame_exp.ndim == 2:
                    solver_pixels = frame_exp[mask.any(axis=0)]
                else:
                    continue
            nonzero = solver_pixels[solver_pixels != 0]
            if nonzero.size == 0:
                continue
            unique_ids, counts = np.unique(nonzero, return_counts=True)
            t_best_idx = int(np.argmax(counts))
            t_best_id = int(unique_ids[t_best_idx])
            t_best_count = int(counts[t_best_idx])
            if t_best_count > best_count:
                best_count = t_best_count
                dominant_solver_id = t_best_id

        if dominant_solver_id is None or dominant_solver_id == 0:
            continue
        if dominant_solver_id == cell_id:
            # Solver already used the correct ID — no remap needed, but still handle
            # collision below where the solver used cell_id for unrelated pixels
            pass
        elif dominant_solver_id in solver_track_remap:
            LOG.warning(
                "Solver track %d claimed by multiple validated cells; "
                "keeping first claim (cell %d), skipping cell %d",
                dominant_solver_id,
                solver_track_remap[dominant_solver_id],
                cell_id,
            )
            continue
        else:
            solver_track_remap[dominant_solver_id] = cell_id

    # Resolve collisions: if any existing pixels have value == target cell_id
    # but DON'T belong to the source solver track, move them to a fresh ID.
    # This is the same as the original collision handling, now generalized to
    # cover both the remap targets and direct-match cases.
    for solver_id, target_cell_id in list(solver_track_remap.items()):
        if solver_id == target_cell_id:
            continue
        # Pixels currently carrying target_cell_id that are NOT from this solver track
        collision_mask = np.asarray(exported_labels == target_cell_id)
        # Remove pixels belonging to the source solver track from collision set
        src_mask = np.asarray(exported_labels == solver_id)
        # Exclude src_mask locations — those will become target_cell_id after remap anyway
        collision_mask = collision_mask & ~src_mask
        if collision_mask.any():
            exported_labels[collision_mask] = _next_fresh_id()

    # Handle cells that are not remapped (dominant_solver_id == cell_id or no dominant found)
    # — original collision logic for reserved IDs used by solver for unrelated pixels.
    remapped_sources = set(solver_track_remap.keys())
    remapped_targets = set(solver_track_remap.values())
    for cell_id, present_masks in validated_masks.items():
        if cell_id in remapped_targets and cell_id not in remapped_sources:
            # Already handled in collision resolution above
            continue
        if cell_id in remapped_sources:
            # This cell IS the source solver track being remapped — no separate collision needed
            continue
        # No remap found: solver either didn't have this cell or already used correct ID.
        # Run original collision resolution for the reserved cell_id.
        solver_collision = np.asarray(exported_labels == cell_id)
        for t, mask in present_masks:
            frame_collision = solver_collision[t]
            if mask.ndim == frame_collision.ndim:
                frame_collision[mask] = False
            elif mask.ndim == 2 and frame_collision.ndim == 3:
                frame_collision[:, mask] = False
            elif mask.ndim == 3 and frame_collision.ndim == 2:
                frame_collision[mask.any(axis=0)] = False
            else:
                raise ValueError(
                    "Validated mask shape is incompatible with exported labels: "
                    f"mask={mask.shape}, exported={frame_collision.shape}"
                )
        if solver_collision.any():
            exported_labels[solver_collision] = _next_fresh_id()

    # Apply solver track remap: rename solver track IDs to validated cell IDs
    for solver_id, target_cell_id in solver_track_remap.items():
        if solver_id == target_cell_id:
            continue
        exported_labels[exported_labels == solver_id] = target_cell_id

    # Paste validated masks (overrides geometry at validated frames)
    for cell_id, present_masks in validated_masks.items():
        for t, mask in present_masks:
            frame_out = exported_labels[t]
            if mask.ndim == frame_out.ndim:
                frame_out[mask] = cell_id
            elif mask.ndim == 2 and frame_out.ndim == 3:
                frame_out[:, mask] = cell_id
            elif mask.ndim == 3 and frame_out.ndim == 2:
                frame_out[mask.any(axis=0)] = cell_id
            else:
                raise ValueError(
                    "Validated mask shape is incompatible with exported labels: "
                    f"mask={mask.shape}, exported={frame_out.shape}"
                )

    return exported_labels, id_map
