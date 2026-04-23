"""Simple seeded tracker driven by persisted hypothesis labelmaps."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace
from pathlib import Path
from typing import Generator, Sequence

import numpy as np
import tifffile

from cellflow.ultrack.ingestion import load_hypothesis_labelmaps
from cellflow.ultrack.linking import _node_mask, _node_origin
from cellflow.ultrack.pruning import _node_from_pickle_value


def _as_uint32(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim not in (2, 3, 4):
        raise ValueError(f"Expected a 2D, 3D, or 4D label array, got shape {labels.shape}")
    return labels.astype(np.uint32, copy=False)


def _consensus_stack(labelmaps: Sequence[np.ndarray]) -> np.ndarray:
    """Return the per-pixel median consensus across the hypothesis stack."""
    if not labelmaps:
        raise ValueError("labelmaps must contain at least one array")

    arrays = [_as_uint32(labels) for labels in labelmaps]
    shapes = {tuple(arr.shape) for arr in arrays}
    if len(shapes) != 1:
        raise ValueError(f"All labelmaps must share the same shape, got {sorted(shapes)}")

    return np.rint(np.median(np.stack(arrays, axis=0), axis=0)).astype(np.uint32)


def _frame_slices(labels: np.ndarray) -> list[np.ndarray]:
    """Return the 2D slices for the first frame of a label stack."""
    labels = np.asarray(labels, dtype=np.uint32)
    if labels.ndim == 2:
        return [labels]
    if labels.ndim == 3:
        return [np.asarray(slice_, dtype=np.uint32) for slice_ in labels]
    if labels.ndim == 4:
        return [np.asarray(slice_, dtype=np.uint32) for slice_ in labels[0]]
    raise ValueError(f"Expected a 2D, 3D, or 4D label array, got shape {labels.shape}")


def _frame_to_2d(labels: np.ndarray) -> np.ndarray:
    """Project a single frame to 2D when it still carries a Z axis."""
    labels = np.asarray(labels, dtype=np.uint32)
    if labels.ndim == 2:
        return labels
    if labels.ndim == 3:
        return np.rint(np.median(labels, axis=0)).astype(np.uint32)
    raise ValueError(f"Expected a 2D or 3D frame, got shape {labels.shape}")


def _relabel_sequential(labels: np.ndarray, *, start_id: int = 1) -> tuple[np.ndarray, dict[int, int]]:
    """Relabel non-zero IDs to a compact consecutive range."""
    labels = np.asarray(labels, dtype=np.uint32)
    out = np.zeros_like(labels, dtype=np.uint32)
    mapping: dict[int, int] = {}
    next_id = int(start_id)
    for label_id in sorted(int(v) for v in np.unique(labels) if int(v) != 0):
        mapping[label_id] = next_id
        out[labels == label_id] = next_id
        next_id += 1
    return out, mapping


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(mask_a, mask_b).sum()
    return float(inter / union)


def _label_masks(labels: np.ndarray) -> dict[int, np.ndarray]:
    labels = np.asarray(labels, dtype=np.uint32)
    masks: dict[int, np.ndarray] = {}
    for label_id in sorted(int(v) for v in np.unique(labels) if int(v) != 0):
        masks[label_id] = labels == label_id
    return masks


def _foreground_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU on the foreground support of two full label images."""
    return _iou(np.asarray(a) > 0, np.asarray(b) > 0)


def _select_medoid_frame(frames: Sequence[np.ndarray]) -> tuple[np.ndarray, int, float]:
    """Pick the image whose foreground overlaps the others best."""
    if not frames:
        raise ValueError("frames must contain at least one image")

    best_idx = 0
    best_score = float("-inf")
    for idx, frame in enumerate(frames):
        score = 0.0
        for other in frames:
            score += _foreground_iou(frame, other)
        if score > best_score:
            best_idx = idx
            best_score = score
    return np.asarray(frames[best_idx], dtype=np.uint32), best_idx, float(best_score)


def _match_frame(
    current: np.ndarray,
    candidate: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Assign each current label to the best remaining candidate label.

    This keeps existing track IDs only. Unmatched candidate labels are left
    unassigned for manual review.
    """
    current = np.asarray(current, dtype=np.uint32)
    candidate = np.asarray(candidate, dtype=np.uint32)

    current_masks = _label_masks(current)
    candidate_masks = _label_masks(candidate)

    assigned_candidate: set[int] = set()
    rows: list[dict[str, object]] = []
    next_frame = np.zeros_like(candidate, dtype=np.uint32)

    for current_id in sorted(current_masks):
        current_mask = current_masks[current_id]
        best_candidate_id: int | None = None
        best_score = 0.0
        for candidate_id in sorted(candidate_masks):
            if candidate_id in assigned_candidate:
                continue
            score = _iou(current_mask, candidate_masks[candidate_id])
            if score > best_score:
                best_score = score
                best_candidate_id = candidate_id

        if best_candidate_id is None or best_score <= 0.0:
            continue

        assigned_candidate.add(best_candidate_id)
        next_frame[candidate == best_candidate_id] = current_id
        rows.append(
            {
                "track_id": current_id,
                "source_track_id": current_id,
                "candidate_label_id": best_candidate_id,
                "iou": float(best_score),
            }
        )

    return next_frame, rows


def _node_canvas(node, frame_shape: tuple[int, ...]) -> np.ndarray | None:
    """Rasterize a node mask into a full-frame boolean canvas."""
    mask = _node_mask(node)
    if mask is None or mask.ndim != len(frame_shape):
        return None

    origin = _node_origin(node, int(mask.ndim))
    if origin is None:
        origin = np.zeros(mask.ndim, dtype=np.float32)

    start = np.rint(np.asarray(origin, dtype=np.float32).reshape(-1)).astype(int)
    if start.size < mask.ndim:
        return None
    start = start[-mask.ndim :]

    canvas = np.zeros(frame_shape, dtype=bool)
    src_slices: list[slice] = []
    dst_slices: list[slice] = []
    for axis, (axis_start, axis_size, mask_size) in enumerate(
        zip(start, frame_shape, mask.shape)
    ):
        dst_start = max(int(axis_start), 0)
        dst_end = min(int(axis_start) + int(mask_size), int(axis_size))
        if dst_start >= dst_end:
            return None
        src_start = dst_start - int(axis_start)
        src_end = src_start + (dst_end - dst_start)
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    canvas[tuple(dst_slices)] = mask[tuple(src_slices)]
    return canvas


def _candidate_nodes_for_timepoint(
    working_dir: str | Path,
    cfg,
    t: int,
) -> list[SimpleNamespace]:
    """Load candidate nodes from the Ultrack database for one timepoint."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from cellflow.ultrack.stages.tracking import _build_ultrack_config
    from ultrack.core.database import NodeDB

    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)
    engine = create_engine(ultrack_cfg.data_config.database_path, hide_parameters=True)

    columns = [NodeDB.id, NodeDB.pickle]
    shift_columns: list[str] = []
    for attr in ("z_shift", "y_shift", "x_shift"):
        if hasattr(NodeDB, attr):
            columns.append(getattr(NodeDB, attr))
            shift_columns.append(attr)

    with Session(engine) as session:
        rows = list(session.query(*columns).where(NodeDB.t == t))

    candidates: list[SimpleNamespace] = []
    for row in rows:
        node_id = int(row[0])
        node = _node_from_pickle_value(row[1])
        node_mask = _node_mask(node)
        if node_mask is None:
            continue

        origin = _node_origin(node, int(node_mask.ndim))
        if origin is None and len(row) > 2:
            shifts = np.asarray(row[2:], dtype=np.float32).reshape(-1)
            if shifts.size >= node_mask.ndim:
                origin = shifts[-node_mask.ndim :]
        if origin is None:
            origin = np.zeros(node_mask.ndim, dtype=np.float32)

        candidates.append(
            SimpleNamespace(
                id=node_id,
                mask=np.asarray(node_mask, dtype=bool),
                origin=np.asarray(origin, dtype=np.float32),
                centroid=getattr(node, "centroid", None),
                shift_columns=tuple(shift_columns),
            )
        )

    return candidates


def _match_frame_against_nodes(
    current: np.ndarray,
    candidate_nodes: Sequence[object],
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Match each current label to the best DB node candidate."""
    current = np.asarray(current, dtype=np.uint32)
    current_masks = _label_masks(current)
    candidate_entries: list[tuple[object, np.ndarray]] = []
    for node in candidate_nodes:
        canvas = _node_canvas(node, tuple(current.shape))
        if canvas is None or not canvas.any():
            continue
        candidate_entries.append((node, canvas))

    assigned_candidate: set[int] = set()
    rows: list[dict[str, object]] = []
    next_frame = np.zeros_like(current, dtype=np.uint32)

    for current_id in sorted(current_masks):
        current_mask = current_masks[current_id]
        best_node = None
        best_canvas = None
        best_score = 0.0
        for node, canvas in candidate_entries:
            node_id = int(getattr(node, "id"))
            if node_id in assigned_candidate:
                continue
            score = _iou(current_mask, canvas)
            if score > best_score:
                best_score = score
                best_node = node
                best_canvas = canvas

        if best_node is None or best_canvas is None or best_score <= 0.0:
            continue

        node_id = int(getattr(best_node, "id"))
        assigned_candidate.add(node_id)
        next_frame[best_canvas] = current_id
        rows.append(
            {
                "track_id": current_id,
                "source_track_id": current_id,
                "candidate_label_id": node_id,
                "iou": float(best_score),
            }
        )

    return next_frame, rows


def build_seeded_tracker_inputs(
    working_dir: str | Path,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, str]:
    """Load hypotheses and return the hypothesis stack plus medoid seed frame."""
    wd = Path(working_dir)
    labelmaps, manifest = load_hypothesis_labelmaps(wd)
    if not labelmaps:
        raise FileNotFoundError(f"No hypothesis labelmaps found in {wd}")

    consensus_stack = _consensus_stack(labelmaps)
    candidate_frames: list[np.ndarray] = []
    candidate_sources: list[tuple[int, int]] = []
    for hyp_idx, labels in enumerate(labelmaps):
        for slice_idx, slice_ in enumerate(_frame_slices(labels)):
            candidate_frames.append(slice_)
            candidate_sources.append((hyp_idx, slice_idx))

    seed, seed_idx, seed_score = _select_medoid_frame(candidate_frames)
    hyp_idx, slice_idx = candidate_sources[seed_idx]
    seed_source = f"medoid:h{hyp_idx}:z{slice_idx}"

    # Preserve the manifest information for callers that want to surface it.
    _ = manifest
    _ = seed_score
    return labelmaps, consensus_stack, seed.astype(np.uint32, copy=False), seed_source


def run_seeded_tracker(
    working_dir: str | Path,
    cfg,
    *,
    overwrite: bool = True,
) -> Generator[tuple[int, int, str], None, None]:
    """Propagate a corrected first frame forward using raw IoU matching."""
    wd = Path(working_dir)
    labelmaps, consensus_stack, seed, seed_source = build_seeded_tracker_inputs(wd)

    out_labels = wd / "tracked_labels.tif"
    out_tracks = wd / "tracks.csv"
    state_path = wd / "seeded_tracker_state.json"

    if not overwrite and out_labels.exists() and out_tracks.exists():
        yield (0, 1, "tracked_labels.tif and tracks.csv already exist, skipping")
        return

    if consensus_stack.shape[0] == 0:
        raise ValueError("Consensus stack is empty")

    n_frames = 1 if consensus_stack.ndim == 2 else int(consensus_stack.shape[0])
    total = n_frames + 2
    yield (0, total, "Bootstrapping seed frame…")

    seed = np.asarray(seed, dtype=np.uint32)
    first_consensus_frame = consensus_stack if consensus_stack.ndim == 2 else consensus_stack[0]
    first_consensus_frame = _frame_to_2d(first_consensus_frame)
    if seed.shape != first_consensus_frame.shape:
        raise ValueError(
            "Seed frame shape does not match the first consensus frame: "
            f"{seed.shape} vs {first_consensus_frame.shape}"
        )

    tracked_frames: list[np.ndarray] = []
    track_rows: list[dict[str, object]] = []

    tracked_seed, seed_mapping = _relabel_sequential(seed)
    tracked_frames.append(tracked_seed)
    for src_label, track_id in seed_mapping.items():
        track_rows.append(
            {
                "track_id": track_id,
                "time": 0,
                "source_label_id": src_label,
                "candidate_label_id": src_label,
                "iou": 1.0,
            }
        )

    current = tracked_seed

    out_labels.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(out_labels),
        tracked_seed[np.newaxis, ...],
        compression="zlib",
        photometric="minisblack",
    )

    for t in range(1, n_frames):
        yield (t, total, f"Matching frame {t}/{n_frames - 1}…")
        candidate_nodes = _candidate_nodes_for_timepoint(wd, cfg, t)
        next_frame, rows = _match_frame_against_nodes(current, candidate_nodes)
        tracked_frames.append(next_frame)
        for row in rows:
            row["time"] = t
            track_rows.append(row)
        current = next_frame

        tracked_stack = np.stack(tracked_frames, axis=0).astype(np.uint32)
        tifffile.imwrite(
            str(out_labels),
            tracked_stack,
            compression="zlib",
            photometric="minisblack",
        )

    tracked_stack = np.stack(tracked_frames, axis=0).astype(np.uint32)

    yield (total - 1, total, "Writing tracked outputs…")
    tifffile.imwrite(
        str(out_labels),
        tracked_stack,
        compression="zlib",
        photometric="minisblack",
    )

    with out_tracks.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "track_id",
                "time",
                "source_track_id",
                "source_label_id",
                "candidate_label_id",
                "iou",
            ],
        )
        writer.writeheader()
        for row in track_rows:
            writer.writerow(row)

    state = {
        "version": 1,
        "seed_source": seed_source,
        "frame_count": int(tracked_stack.shape[0]),
        "track_count": int(tracked_stack.max()) if tracked_stack.size else 0,
        "labelmap_count": len(labelmaps),
        "shape": list(tracked_stack.shape),
        "tracked_labels": out_labels.name,
        "tracks": out_tracks.name,
    }
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    yield (total, total, "Seeded tracker done.")
