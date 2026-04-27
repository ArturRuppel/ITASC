"""Anchor-LAP propagator: pick globally best hypothesis frame and propagate IDs via linear assignment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from cellflow.database.hypotheses import list_hypotheses, read_hypothesis_labels


@dataclass(slots=True)
class PropagationContext:
    """Per-frame propagation context. prev_labels and validated_history are reserved for future motion-prediction features."""

    current_labels: np.ndarray
    prev_labels: np.ndarray | None = None
    validated_history: dict[int, np.ndarray] | None = None


def _iou_matrix(current: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return (n_cur_cells, n_cand_cells) IoU matrix, excluding background (label 0)."""
    n_cur = int(current.max()) + 1
    n_cand = int(candidate.max()) + 1
    idx = current.astype(np.int64).ravel() * n_cand + candidate.astype(np.int64).ravel()
    conf = np.bincount(idx, minlength=n_cur * n_cand).reshape(n_cur, n_cand)
    area_cur = conf.sum(axis=1)
    area_cand = conf.sum(axis=0)
    inter = conf[1:, 1:]
    union = area_cur[1:, None] + area_cand[None, 1:] - inter
    iou = np.where(union > 0, inter / union, 0.0)
    return iou


def _score_hypothesis(current: np.ndarray, candidate: np.ndarray, *, min_match_iou: float, alpha: float) -> tuple:
    """Return (score, n_matched, total_iou, row_ind, col_ind) for one hypothesis candidate."""
    if candidate.max() == 0:
        n_cur_cells = int(current.max())
        return -alpha * n_cur_cells, 0, 0.0, np.array([], dtype=int), np.array([], dtype=int)

    iou = _iou_matrix(current, candidate)
    row_ind, col_ind = linear_sum_assignment(iou, maximize=True)

    matched_iou = iou[row_ind, col_ind]
    mask = matched_iou >= min_match_iou
    row_ind, col_ind, matched_iou = row_ind[mask], col_ind[mask], matched_iou[mask]

    n_cur_cells = int(current.max())
    n_matched = int(mask.sum())
    total_iou = float(matched_iou.sum())
    score = total_iou - alpha * (n_cur_cells - n_matched)
    return score, n_matched, total_iou, row_ind, col_ind


def find_best_hypothesis_v2(
    context: PropagationContext,
    candidates: list[np.ndarray],
    *,
    min_match_iou: float = 0.1,
    alpha: float = 0.3,
) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Return (relabeled_next_frame, winning_p_index) or (None, None).

    Scores each hypothesis in candidates via LAP matching and IoU thresholding,
    picks the globally highest-scoring one, and relabels its matched cells.
    """
    if not candidates:
        return None, None

    current = context.current_labels
    if current.max() == 0:
        return None, None

    best_score = -np.inf
    best_p = -1
    best_n_matched = 0
    best_row_ind = None
    best_col_ind = None
    best_cand = None

    for p, candidate in enumerate(candidates):
        score, n_matched, _, row_ind, col_ind = _score_hypothesis(current, candidate, min_match_iou=min_match_iou, alpha=alpha)
        if score > best_score:
            best_score = score
            best_p = p
            best_n_matched = n_matched
            best_row_ind = row_ind
            best_col_ind = col_ind
            best_cand = candidate

    if best_score == -np.inf or best_p < 0 or best_n_matched == 0:
        return None, None

    Y, X = best_cand.shape
    propagated = np.zeros((Y, X), dtype=np.uint32)
    for cur_idx, cand_idx in zip(best_row_ind, best_col_ind):
        cur_id = cur_idx + 1
        cand_id = cand_idx + 1
        propagated[best_cand == cand_id] = cur_id

    return propagated, best_p


def _build_composite_frame(
    current: np.ndarray,
    candidates: list[np.ndarray],
    *,
    min_match_iou: float = 0.1,
) -> tuple[np.ndarray, int]:
    """Assemble a composite next frame by selecting the best-matching candidate cell
    from *any* hypothesis for each current cell.

    For every current cell, scan all hypothesis candidates and record the
    highest-IoU match (candidate label + source hypothesis index p).
    Then paint cells into a composite label image in descending IoU order
    so confident matches claim their pixels first; lower-confidence cells
    receive whatever pixels remain unoccupied.

    Returns (composite_labels, winning_p_index) where winning_p_index is the
    hypothesis that contributed the most cells.
    """
    n_cur = int(current.max())
    H, W = current.shape

    # best_for_cell[c_idx] = (best_iou, best_p, best_cand_idx)  -- all 0-based
    best_for_cell: list[tuple[float, int, int]] = [(-1.0, -1, -1)] * n_cur

    for p, cand in enumerate(candidates):
        if cand.max() == 0:
            continue
        iou = _iou_matrix(current, cand)  # (n_cur, n_cand)
        if iou.shape[1] == 0:
            continue
        best_cand_idxs = np.argmax(iou, axis=1)            # (n_cur,)
        best_ious = iou[np.arange(n_cur), best_cand_idxs]  # (n_cur,)
        for c_idx in range(n_cur):
            if best_ious[c_idx] > best_for_cell[c_idx][0]:
                best_for_cell[c_idx] = (float(best_ious[c_idx]), p, int(best_cand_idxs[c_idx]))

    # Paint highest-IoU cells first to resolve pixel conflicts
    order = sorted(range(n_cur), key=lambda i: best_for_cell[i][0], reverse=True)

    composite = np.zeros((H, W), dtype=np.uint32)
    occupied = np.zeros((H, W), dtype=bool)
    source_counts: dict[int, int] = {}

    for c_idx in order:
        best_iou, best_p, best_cand_idx = best_for_cell[c_idx]
        if best_iou < min_match_iou or best_p < 0:
            continue
        c_id = c_idx + 1
        cand = candidates[best_p]
        cand_id = best_cand_idx + 1
        pixels = (cand == cand_id) & ~occupied
        composite[pixels] = c_id
        occupied |= pixels
        source_counts[best_p] = source_counts.get(best_p, 0) + 1

    if not source_counts:
        return composite, -1

    winning_p = max(source_counts, key=source_counts.__getitem__)
    return composite, winning_p


def find_composite_v2(
    context: PropagationContext,
    candidates: list[np.ndarray],
    *,
    min_match_iou: float = 0.1,
    alpha: float = 0.3,
) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Return (relabeled_next_frame, winning_p_index) or (None, None).

    Assembles a composite next frame by selecting the best-matching candidate
    cell from *any* hypothesis for each current cell, rather than picking a
    single globally best hypothesis.
    """
    if not candidates:
        return None, None

    current = context.current_labels
    if current.max() == 0:
        return None, None

    composite, winning_p = _build_composite_frame(current, candidates, min_match_iou=min_match_iou)

    if winning_p < 0 or composite.max() == 0:
        return None, None

    return composite, winning_p


def propagate_one_frame_composite(
    hypotheses_h5: str | Path,
    current_labels: np.ndarray,
    t_next: int,
    prev_labels: np.ndarray | None = None,
    validated_history: dict[int, np.ndarray] | None = None,
    *,
    min_match_iou: float = 0.1,
    alpha: float = 0.3,
) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Propagate tracking to t_next using the greedy composite frame algorithm.

    For each current nucleus, picks the best-matching candidate from *any*
    hypothesis rather than choosing a single globally best hypothesis.

    Returns (relabeled_next_frame, winning_p_index) or (None, None) if no valid match.
    """
    hypotheses_h5 = Path(hypotheses_h5)

    n_p, _ = list_hypotheses(hypotheses_h5)
    if n_p == 0:
        return None, None

    candidates: list[np.ndarray] = []
    blank = np.zeros(current_labels.shape, dtype=np.uint32)
    for p in range(n_p):
        try:
            raw = read_hypothesis_labels(hypotheses_h5, t_next, p)
        except (KeyError, ValueError):
            candidates.append(blank)
            continue
        candidate = raw.squeeze(axis=0) if raw.ndim == 3 and raw.shape[0] == 1 else raw[0]
        candidates.append(candidate)

    context = PropagationContext(
        current_labels=current_labels,
        prev_labels=prev_labels,
        validated_history=validated_history,
    )
    return find_composite_v2(context, candidates, min_match_iou=min_match_iou, alpha=alpha)


def propagate_one_frame_v2(
    hypotheses_h5: str | Path,
    current_labels: np.ndarray,
    t_next: int,
    prev_labels: np.ndarray | None = None,
    validated_history: dict[int, np.ndarray] | None = None,
    *,
    min_match_iou: float = 0.1,
    alpha: float = 0.3,
) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Propagate tracking to t_next using current_labels as source.

    Searches all hypotheses in the database for t_next, matches current nuclei
    to candidate nuclei via LAP global assignment, returns the best hypothesis.

    Returns (relabeled_next_frame, winning_p_index) or (None, None) if no valid match.
    """
    hypotheses_h5 = Path(hypotheses_h5)

    n_p, _ = list_hypotheses(hypotheses_h5)
    if n_p == 0:
        return None, None

    candidates: list[np.ndarray] = []
    blank = np.zeros(current_labels.shape, dtype=np.uint32)
    for p in range(n_p):
        try:
            raw = read_hypothesis_labels(hypotheses_h5, t_next, p)
        except (KeyError, ValueError):
            candidates.append(blank)
            continue
        candidate = raw.squeeze(axis=0) if raw.ndim == 3 and raw.shape[0] == 1 else raw[0]
        candidates.append(candidate)

    context = PropagationContext(
        current_labels=current_labels,
        prev_labels=prev_labels,
        validated_history=validated_history,
    )
    return find_best_hypothesis_v2(context, candidates, min_match_iou=min_match_iou, alpha=alpha)
