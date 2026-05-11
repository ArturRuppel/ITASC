"""Track-conditioned cell boundary selection utilities.

This module implements the phase-1 pure-Python backend from the design: given
per-frame candidate masks and existing nucleus track labels, choose one
candidate per known track-frame with a small dynamic-programming solver.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import tifffile


@dataclass(frozen=True, slots=True)
class BoundaryCandidate:
    """One cell-boundary candidate mask for a single frame."""

    node_id: int
    t: int
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class BoundarySelectionParams:
    """Weights and thresholds for per-track boundary dynamic programming."""

    min_nucleus_fraction: float = 0.0
    anchor_weight: float = 2.0
    conflict_weight: float = 0.25
    centroid_jump_weight: float = 0.25
    area_change_weight: float = 0.5
    iou_loss_weight: float = 1.0
    missing_penalty: float = 10.0


@dataclass(frozen=True, slots=True)
class AnchorScore:
    track_id: int
    nucleus_pixels: int
    covered_pixels: int
    fraction: float
    other_nucleus_pixels: int


@dataclass(frozen=True, slots=True)
class BoundarySelectionResult:
    track_id: int
    selected_node_ids: dict[int, int | None]
    total_score: float

    @property
    def missing_frames(self) -> set[int]:
        return {t for t, node_id in self.selected_node_ids.items() if node_id is None}


@dataclass(frozen=True, slots=True)
class OverlapConflict:
    t: int
    track_ids: tuple[int, int]
    node_ids: tuple[int, int]
    overlap_pixels: int


def _normalize_tyx_array(array: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr[np.newaxis]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        if arr.shape[1] != 1:
            raise ValueError(
                f"{name} has unsupported non-singleton Z/channel axis: {arr.shape}"
            )
        return arr[:, 0]
    raise ValueError(f"{name} has unsupported ndim {arr.ndim}: {arr.shape}")


def _read_required_tiff(path: str | Path, *, name: str) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return np.asarray(tifffile.imread(str(path)))


def validate_cell_boundary_inputs(
    contour_maps_path: str | Path,
    foreground_masks_path: str | Path,
    nucleus_tracked_labels_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate boundary-selection TIFF inputs as ``(T, Y, X)`` arrays."""
    contours = _normalize_tyx_array(
        _read_required_tiff(contour_maps_path, name="cell contour maps"),
        name="cell contour maps",
    ).astype(np.float32, copy=False)
    foreground = _normalize_tyx_array(
        _read_required_tiff(foreground_masks_path, name="cell foreground masks"),
        name="cell foreground masks",
    ).astype(bool, copy=False)
    nuclei = _normalize_tyx_array(
        _read_required_tiff(nucleus_tracked_labels_path, name="nucleus tracked labels"),
        name="nucleus tracked labels",
    ).astype(np.uint32, copy=False)

    if contours.shape != foreground.shape:
        raise ValueError(
            "cell contour maps and foreground masks shape mismatch: "
            f"{contours.shape} != {foreground.shape}"
        )
    if contours.shape != nuclei.shape:
        raise ValueError(
            "cell boundary inputs and nucleus tracked labels shape mismatch: "
            f"{contours.shape} != {nuclei.shape}"
        )
    return contours, foreground, nuclei


def _data_db_path(working_dir: str | Path) -> Path:
    path = Path(working_dir)
    if path.name == "data.db":
        return path
    return path / "data.db"


def load_candidates_from_db(working_dir: str | Path) -> list[BoundaryCandidate]:
    """Read Ultrack ``NodeDB`` rows from ``working_dir/data.db`` as candidates."""
    db_path = _data_db_path(working_dir)
    if not db_path.exists():
        raise FileNotFoundError(f"Ultrack data.db not found: {db_path}")

    try:
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB
    except ImportError as exc:
        raise ImportError(
            "sqlalchemy and ultrack must be installed to load candidates from data.db"
        ) from exc

    from cellflow.tracking_ultrack.validation_nodes import _node_bbox_and_mask

    engine = sqla.create_engine(f"sqlite:///{db_path}")
    try:
        with Session(engine) as session:
            rows = (
                session.query(NodeDB.id, NodeDB.t, NodeDB.pickle, NodeDB.node_prob)
                .order_by(NodeDB.t, NodeDB.id)
                .all()
            )
    finally:
        engine.dispose()

    candidates: list[BoundaryCandidate] = []
    for node_id, t, node_pickle, node_prob in rows:
        bbox, mask = _node_bbox_and_mask(int(node_id), node_pickle)
        raw_score = None if node_prob is None else float(node_prob)
        score = 0.0 if raw_score is None or raw_score < 0.0 else raw_score
        candidates.append(
            BoundaryCandidate(
                node_id=int(node_id),
                t=int(t),
                mask=np.ascontiguousarray(mask, dtype=bool),
                bbox=tuple(int(v) for v in bbox),
                score=score,
            )
        )
    return candidates


def _bbox_slices(bbox: tuple[int, int, int, int]) -> tuple[slice, slice]:
    y0, x0, y1, x1 = bbox
    return slice(int(y0), int(y1)), slice(int(x0), int(x1))


def _candidate_full_mask(candidate: BoundaryCandidate, shape: tuple[int, int]) -> np.ndarray:
    full = np.zeros(shape, dtype=bool)
    y_slice, x_slice = _bbox_slices(candidate.bbox)
    crop = np.asarray(candidate.mask, dtype=bool)
    expected_shape = _validate_candidate_geometry(candidate, shape)
    if crop.shape != expected_shape:
        raise ValueError(
            f"candidate {candidate.node_id} mask shape {crop.shape} does not match "
            f"bbox shape {expected_shape}"
        )
    full[y_slice, x_slice] = crop
    return full


def _validate_candidate_geometry(
    candidate: BoundaryCandidate,
    frame_shape: tuple[int, int],
) -> tuple[int, int]:
    y0, x0, y1, x1 = candidate.bbox
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if y0 < 0 or x0 < 0 or y1 > height or x1 > width or y0 >= y1 or x0 >= x1:
        raise ValueError(
            f"candidate {candidate.node_id} bbox {candidate.bbox} is outside "
            f"output shape {frame_shape}"
        )
    expected_shape = (int(y1 - y0), int(x1 - x0))
    crop = np.asarray(candidate.mask, dtype=bool)
    if crop.shape != expected_shape:
        raise ValueError(
            f"candidate {candidate.node_id} mask shape {crop.shape} does not match "
            f"bbox shape {expected_shape}"
        )
    return expected_shape


def _candidate_centroid(candidate: BoundaryCandidate) -> tuple[float, float]:
    mask = np.asarray(candidate.mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        y0, x0, y1, x1 = candidate.bbox
        return ((y0 + y1) / 2.0, (x0 + x1) / 2.0)
    return (
        float(candidate.bbox[0] + ys.mean()),
        float(candidate.bbox[1] + xs.mean()),
    )


def _candidate_area(candidate: BoundaryCandidate) -> int:
    return int(np.asarray(candidate.mask, dtype=bool).sum())


def score_candidate_anchor(
    candidate: BoundaryCandidate,
    nucleus_frame: np.ndarray,
    track_id: int,
) -> AnchorScore:
    """Return nucleus anchor coverage for ``candidate`` and ``track_id``."""
    frame = np.asarray(nucleus_frame)
    if frame.ndim != 2:
        raise ValueError(f"Expected 2D nucleus frame, got shape {frame.shape}")

    nucleus = frame == int(track_id)
    nucleus_pixels = int(nucleus.sum())
    if nucleus_pixels == 0:
        return AnchorScore(int(track_id), 0, 0, 0.0, 0)

    full_mask = _candidate_full_mask(candidate, frame.shape)
    covered = int(np.logical_and(full_mask, nucleus).sum())
    other_nucleus_pixels = int(np.logical_and(full_mask, (frame > 0) & ~nucleus).sum())
    return AnchorScore(
        track_id=int(track_id),
        nucleus_pixels=nucleus_pixels,
        covered_pixels=covered,
        fraction=float(covered) / float(nucleus_pixels),
        other_nucleus_pixels=other_nucleus_pixels,
    )


def is_candidate_eligible_for_track(
    candidate: BoundaryCandidate,
    nucleus_frame: np.ndarray,
    track_id: int,
    *,
    min_nucleus_fraction: float = 0.0,
) -> bool:
    """Return true when the candidate satisfies the hard-anchor rule."""
    anchor = score_candidate_anchor(candidate, nucleus_frame, track_id)
    if anchor.covered_pixels <= 0:
        return False
    return anchor.fraction >= float(min_nucleus_fraction)


def _candidate_iou(lhs: BoundaryCandidate, rhs: BoundaryCandidate) -> float:
    ly0, lx0, ly1, lx1 = lhs.bbox
    ry0, rx0, ry1, rx1 = rhs.bbox
    oy0, ox0 = max(ly0, ry0), max(lx0, rx0)
    oy1, ox1 = min(ly1, ry1), min(lx1, rx1)
    intersection = 0
    if oy0 < oy1 and ox0 < ox1:
        lhs_crop = np.asarray(lhs.mask, dtype=bool)[oy0 - ly0: oy1 - ly0, ox0 - lx0: ox1 - lx0]
        rhs_crop = np.asarray(rhs.mask, dtype=bool)[oy0 - ry0: oy1 - ry0, ox0 - rx0: ox1 - rx0]
        intersection = int(np.logical_and(lhs_crop, rhs_crop).sum())
    union = _candidate_area(lhs) + _candidate_area(rhs) - intersection
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


def _transition_score(
    previous: BoundaryCandidate | None,
    current: BoundaryCandidate | None,
    params: BoundarySelectionParams,
) -> float:
    if previous is None or current is None:
        return 0.0
    py, px = _candidate_centroid(previous)
    cy, cx = _candidate_centroid(current)
    centroid_penalty = float(np.hypot(cy - py, cx - px)) * params.centroid_jump_weight

    prev_area = max(_candidate_area(previous), 1)
    curr_area = max(_candidate_area(current), 1)
    area_ratio = abs(np.log(float(curr_area) / float(prev_area)))
    area_penalty = area_ratio * params.area_change_weight

    iou_penalty = (1.0 - _candidate_iou(previous, current)) * params.iou_loss_weight
    return -(centroid_penalty + area_penalty + iou_penalty)


def _unary_score(
    candidate: BoundaryCandidate | None,
    nucleus_frame: np.ndarray,
    track_id: int,
    params: BoundarySelectionParams,
) -> float:
    if candidate is None:
        return -float(params.missing_penalty)
    anchor = score_candidate_anchor(candidate, nucleus_frame, track_id)
    conflict_penalty = anchor.other_nucleus_pixels * params.conflict_weight
    return float(candidate.score) + anchor.fraction * params.anchor_weight - conflict_penalty


def select_track_boundaries_dp(
    candidates: list[BoundaryCandidate],
    nucleus_labels: np.ndarray,
    track_id: int,
    params: BoundarySelectionParams | None = None,
) -> BoundarySelectionResult:
    """Select one candidate or missing state for each frame of a known track."""
    params = params or BoundarySelectionParams()
    labels = np.asarray(nucleus_labels)
    if labels.ndim != 3:
        raise ValueError(f"Expected nucleus labels shaped (T, Y, X), got {labels.shape}")

    frames = [
        int(t)
        for t in range(labels.shape[0])
        if np.any(labels[t] == int(track_id))
    ]
    by_frame: dict[int, list[BoundaryCandidate | None]] = {}
    for t in frames:
        eligible = [
            candidate
            for candidate in candidates
            if int(candidate.t) == t
            and is_candidate_eligible_for_track(
                candidate,
                labels[t],
                track_id,
                min_nucleus_fraction=params.min_nucleus_fraction,
            )
        ]
        by_frame[t] = [None] + sorted(eligible, key=lambda candidate: int(candidate.node_id))

    if not frames:
        return BoundarySelectionResult(int(track_id), {}, 0.0)

    scores: dict[tuple[int, int], float] = {}
    back: dict[tuple[int, int], int | None] = {}
    first_t = frames[0]
    for idx, state in enumerate(by_frame[first_t]):
        scores[(first_t, idx)] = _unary_score(state, labels[first_t], track_id, params)
        back[(first_t, idx)] = None

    for t_prev, t_cur in zip(frames, frames[1:]):
        for cur_idx, cur_state in enumerate(by_frame[t_cur]):
            best_score = -np.inf
            best_prev_idx: int | None = None
            for prev_idx, prev_state in enumerate(by_frame[t_prev]):
                candidate_score = (
                    scores[(t_prev, prev_idx)]
                    + _transition_score(prev_state, cur_state, params)
                    + _unary_score(cur_state, labels[t_cur], track_id, params)
                )
                if candidate_score > best_score:
                    best_score = float(candidate_score)
                    best_prev_idx = prev_idx
            scores[(t_cur, cur_idx)] = best_score
            back[(t_cur, cur_idx)] = best_prev_idx

    last_t = frames[-1]
    last_idx = max(range(len(by_frame[last_t])), key=lambda idx: scores[(last_t, idx)])
    total_score = scores[(last_t, last_idx)]

    selected: dict[int, int | None] = {}
    idx: int | None = last_idx
    for frame_index in range(len(frames) - 1, -1, -1):
        t = frames[frame_index]
        state = by_frame[t][int(idx)]
        selected[t] = None if state is None else int(state.node_id)
        idx = back[(t, int(idx))]
    selected = dict(sorted(selected.items()))
    return BoundarySelectionResult(int(track_id), selected, total_score)


def select_all_track_boundaries(
    candidates: list[BoundaryCandidate],
    nucleus_labels: np.ndarray,
    *,
    track_ids: list[int] | None = None,
    params: BoundarySelectionParams | None = None,
) -> dict[int, BoundarySelectionResult]:
    """Run per-track boundary selection for each known nucleus track ID."""
    labels = np.asarray(nucleus_labels)
    if labels.ndim != 3:
        raise ValueError(f"Expected nucleus labels shaped (T, Y, X), got {labels.shape}")
    if track_ids is None:
        track_ids = [
            int(track_id)
            for track_id in sorted(np.unique(labels))
            if int(track_id) != 0
        ]
    return {
        int(track_id): select_track_boundaries_dp(
            candidates,
            labels,
            int(track_id),
            params=params,
        )
        for track_id in track_ids
    }


def _selected_items(
    selections: dict[int, dict[int, int | None]],
    candidates: dict[int, BoundaryCandidate],
    t: int,
) -> list[tuple[int, BoundaryCandidate]]:
    items: list[tuple[int, BoundaryCandidate]] = []
    for track_id, by_frame in selections.items():
        node_id = by_frame.get(t)
        if node_id is None:
            continue
        items.append((int(track_id), candidates[int(node_id)]))
    return items


def _overlap_pixels(lhs: BoundaryCandidate, rhs: BoundaryCandidate) -> int:
    ly0, lx0, ly1, lx1 = lhs.bbox
    ry0, rx0, ry1, rx1 = rhs.bbox
    oy0, ox0 = max(ly0, ry0), max(lx0, rx0)
    oy1, ox1 = min(ly1, ry1), min(lx1, rx1)
    if oy0 >= oy1 or ox0 >= ox1:
        return 0
    lhs_crop = np.asarray(lhs.mask, dtype=bool)[oy0 - ly0: oy1 - ly0, ox0 - lx0: ox1 - lx0]
    rhs_crop = np.asarray(rhs.mask, dtype=bool)[oy0 - ry0: oy1 - ry0, ox0 - rx0: ox1 - rx0]
    return int(np.logical_and(lhs_crop, rhs_crop).sum())


def detect_overlap_conflicts(
    selections: dict[int, dict[int, int | None]],
    candidates: dict[int, BoundaryCandidate],
    *,
    min_overlap_pixels: int = 1,
) -> list[OverlapConflict]:
    """Return selected candidate overlaps between different tracks."""
    frames = sorted({t for by_frame in selections.values() for t in by_frame})
    conflicts: list[OverlapConflict] = []
    for t in frames:
        for (lhs_track, lhs), (rhs_track, rhs) in combinations(
            _selected_items(selections, candidates, t),
            2,
        ):
            overlap = _overlap_pixels(lhs, rhs)
            if overlap >= int(min_overlap_pixels):
                conflicts.append(
                    OverlapConflict(
                        t=int(t),
                        track_ids=(lhs_track, rhs_track),
                        node_ids=(int(lhs.node_id), int(rhs.node_id)),
                        overlap_pixels=overlap,
                    )
                )
    return conflicts


def export_selected_boundaries(
    selections: dict[int, dict[int, int | None]],
    candidates: dict[int, BoundaryCandidate],
    *,
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Rasterize selected candidates with the original nucleus track IDs."""
    if len(shape) != 3:
        raise ValueError(f"Expected output shape (T, Y, X), got {shape}")
    labels = np.zeros(shape, dtype=np.uint32)
    for track_id in sorted(selections):
        for t, node_id in sorted(selections[track_id].items()):
            if node_id is None:
                continue
            candidate = candidates[int(node_id)]
            if int(candidate.t) != int(t):
                raise ValueError(
                    f"candidate {candidate.node_id} belongs to t={candidate.t}, "
                    f"not selected frame t={t}"
                )
            if int(t) < 0 or int(t) >= shape[0]:
                raise ValueError(
                    f"selected frame t={t} is outside output shape {shape}"
                )
            _validate_candidate_geometry(candidate, shape[1:])
            y_slice, x_slice = _bbox_slices(candidate.bbox)
            labels[int(t), y_slice, x_slice][np.asarray(candidate.mask, dtype=bool)] = int(track_id)
    return labels


@dataclass(frozen=True, slots=True)
class BoundarySelectionRunResult:
    """Result of a ``run_track_conditioned_boundary_selection`` invocation."""

    output_path: Path
    diagnostics_path: Path
    candidate_count: int
    missing_frames_by_track: dict[int, list[int]]
    overlap_conflicts: list[OverlapConflict]


def run_track_conditioned_boundary_selection(
    pos_dir: str | Path,
    *,
    candidate_loader: Callable[[str | Path], list[BoundaryCandidate]] | None = None,
    params: BoundarySelectionParams | None = None,
) -> BoundarySelectionRunResult:
    """Run the full track-conditioned boundary selection pipeline for a position.

    Resolves default input and output paths relative to *pos_dir*:

    * ``3_cell/contour_maps.tif``
    * ``3_cell/foreground_masks.tif``
    * ``2_nucleus/tracked_labels.tif``
    * ``3_cell/ultrack_workdir`` (Ultrack working directory)
    * ``3_cell/tracked_labels.tif`` (output)
    * ``3_cell/boundary_selection_diagnostics.json`` (diagnostics)

    Parameters
    ----------
    pos_dir:
        Root position directory containing ``3_cell`` and ``2_nucleus`` subdirs.
    candidate_loader:
        Callable that receives the working directory path and returns a list of
        ``BoundaryCandidate`` objects.  When *None* (default), the standard
        ``load_candidates_from_db`` loader is used.
    params:
        Optional :class:`BoundarySelectionParams` controlling the per-track DP.

    Returns
    -------
    BoundarySelectionRunResult
    """
    pos_dir = Path(pos_dir)
    cell_dir = pos_dir / "3_cell"
    nucleus_dir = pos_dir / "2_nucleus"

    contour_maps_path = cell_dir / "contour_maps.tif"
    foreground_masks_path = cell_dir / "foreground_masks.tif"
    nucleus_tracked_labels_path = nucleus_dir / "tracked_labels.tif"
    working_dir = cell_dir / "ultrack_workdir"
    output_path = cell_dir / "tracked_labels.tif"
    diagnostics_path = cell_dir / "boundary_selection_diagnostics.json"

    # --- validate inputs ---
    _, _, nuclei = validate_cell_boundary_inputs(
        contour_maps_path,
        foreground_masks_path,
        nucleus_tracked_labels_path,
    )

    # --- load candidates ---
    loader = candidate_loader if candidate_loader is not None else load_candidates_from_db
    candidates = loader(working_dir)

    # --- per-track DP ---
    results = select_all_track_boundaries(candidates, nuclei, params=params)

    selections: dict[int, dict[int, int | None]] = {
        track_id: result.selected_node_ids
        for track_id, result in results.items()
    }
    candidates_by_id: dict[int, BoundaryCandidate] = {
        candidate.node_id: candidate for candidate in candidates
    }

    # --- rasterize & write output ---
    labels = export_selected_boundaries(selections, candidates_by_id, shape=nuclei.shape)
    tifffile.imwrite(str(output_path), labels, compression="zlib")

    # --- overlap diagnostics ---
    conflicts = detect_overlap_conflicts(selections, candidates_by_id)

    missing_frames_by_track: dict[int, list[int]] = {
        track_id: sorted(result.missing_frames)
        for track_id, result in results.items()
    }

    diagnostics: dict = {
        "candidate_count": len(candidates),
        "track_count": len(results),
        "conflicts": [
            {
                "t": c.t,
                "track_ids": list(c.track_ids),
                "node_ids": list(c.node_ids),
                "overlap_pixels": c.overlap_pixels,
            }
            for c in conflicts
        ],
        "missing_frames_by_track": {
            str(k): v for k, v in missing_frames_by_track.items()
        },
    }
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2))

    return BoundarySelectionRunResult(
        output_path=output_path,
        diagnostics_path=diagnostics_path,
        candidate_count=len(candidates),
        missing_frames_by_track=missing_frames_by_track,
        overlap_conflicts=conflicts,
    )
