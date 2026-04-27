"""Anchor-LAP propagator: pick the globally best hypothesis frame and propagate IDs via LAP.

Also includes a cost-function benchmark: for the first N_BENCHMARK frames, computes the
oracle score (tracked[t] → tracked[t+1]) and compares it against every single hypothesis
and against a greedy composite frame assembled from the best per-cell match across all
hypotheses.
"""

import numpy as np
import tifffile
import napari
from scipy.optimize import linear_sum_assignment

from cellflow.database.hypotheses import list_hypotheses, read_hypothesis_labels
from cellflow.database.tracked import read_full_tracked_stack

# --- Paths ---
HYPO_H5 = (
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/"
    "v2/pos00/2_nucleus/hypotheses.h5"
)
TRACKED_TIFF = (
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/"
    "v2/pos00/2_nucleus/tracked_labels.tif"
)
CELL_ZAVG_TIFF = (
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/"
    "v2/pos00/0_input/cell_zavg.tif"
)
NUC_ZAVG_TIFF = (
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/"
    "v2/pos00/0_input/nucleus_zavg.tif"
)

# --- Algorithm constants ---
MIN_MATCH_IOU = 0.1
ALPHA = 0.3

# --- Benchmark constants ---
N_BENCHMARK = 10  # number of tracked frames to use as oracle baseline


def compute_iou_matrix(current: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return (n_cur_cells, n_cand_cells) IoU matrix, excluding background (label 0)."""
    n_cur = int(current.max()) + 1
    n_cand = int(candidate.max()) + 1
    idx = current.astype(np.int64).ravel() * n_cand + candidate.astype(np.int64).ravel()
    conf = np.bincount(idx, minlength=n_cur * n_cand).reshape(n_cur, n_cand)
    area_cur = conf.sum(axis=1)
    area_cand = conf.sum(axis=0)
    inter = conf[1:, 1:]
    union = area_cur[1:, None] + area_cand[None, 1:] - inter
    with np.errstate(invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def score_hypothesis(current: np.ndarray, candidate: np.ndarray):
    """Return (score, n_matched, total_iou, row_ind, col_ind)."""
    if candidate.max() == 0:
        n_cur_cells = int(current.max())
        return -ALPHA * n_cur_cells, 0, 0.0, np.array([], dtype=int), np.array([], dtype=int)

    iou = compute_iou_matrix(current, candidate)
    row_ind, col_ind = linear_sum_assignment(iou, maximize=True)

    matched_iou = iou[row_ind, col_ind]
    mask = matched_iou >= MIN_MATCH_IOU
    row_ind, col_ind, matched_iou = row_ind[mask], col_ind[mask], matched_iou[mask]

    n_cur_cells = int(current.max())
    n_matched = int(mask.sum())
    total_iou = float(matched_iou.sum())
    score = total_iou - ALPHA * (n_cur_cells - n_matched)
    return score, n_matched, total_iou, row_ind, col_ind


def build_composite_frame(
    current: np.ndarray,
    candidates: list[np.ndarray],
) -> tuple[np.ndarray, dict[int, tuple[float, int, int]]]:
    """Assemble a composite frame by selecting the best-matching candidate for each cell.

    For every current cell, scans all hypotheses and records the highest-IoU match
    (candidate label + source hypothesis index).  Cells are then painted into the
    composite in descending IoU order so confident matches claim their pixels first;
    lower-confidence cells receive whatever pixels remain unoccupied.

    Returns
    -------
    composite : (Y, X) uint32 label image
    per_cell_info : dict mapping c_id -> (best_iou, winning_p, cand_id)
    """
    n_cur = int(current.max())
    if n_cur == 0:
        return np.zeros_like(current), {}

    H, W = current.shape

    # best_for_cell[c_idx] = (best_iou, best_p, best_cand_idx)  -- all 0-based
    best_for_cell: list[tuple[float, int, int]] = [(-1.0, -1, -1)] * n_cur

    for p, cand in enumerate(candidates):
        if cand.max() == 0:
            continue
        iou = compute_iou_matrix(current, cand)
        if iou.shape[1] == 0:
            continue
        best_cand_idxs = np.argmax(iou, axis=1)          # (n_cur,)
        best_ious = iou[np.arange(n_cur), best_cand_idxs] # (n_cur,)
        for c_idx in range(n_cur):
            if best_ious[c_idx] > best_for_cell[c_idx][0]:
                best_for_cell[c_idx] = (
                    float(best_ious[c_idx]), p, int(best_cand_idxs[c_idx])
                )

    # Paint cells highest-IoU-first to resolve pixel conflicts
    order = sorted(range(n_cur), key=lambda i: best_for_cell[i][0], reverse=True)

    composite = np.zeros((H, W), dtype=np.uint32)
    occupied = np.zeros((H, W), dtype=bool)
    per_cell_info: dict[int, tuple[float, int, int]] = {}

    for c_idx in order:
        best_iou, best_p, best_cand_idx = best_for_cell[c_idx]
        c_id = c_idx + 1

        if best_iou < MIN_MATCH_IOU or best_p < 0:
            per_cell_info[c_id] = (0.0, -1, -1)
            continue

        cand = candidates[best_p]
        cand_id = best_cand_idx + 1
        pixels = (cand == cand_id) & ~occupied
        composite[pixels] = c_id
        occupied |= pixels
        per_cell_info[c_id] = (best_iou, best_p, cand_id)

    return composite, per_cell_info


def _load_candidates(t_tgt: int, n_p: int, shape) -> list[np.ndarray]:
    blank = np.zeros(shape, dtype=np.uint32)
    candidates = []
    for p in range(n_p):
        try:
            raw = read_hypothesis_labels(HYPO_H5, t_tgt, p)
        except (KeyError, ValueError):
            candidates.append(blank)
            continue
        cand = raw.squeeze(axis=0) if raw.ndim == 3 and raw.shape[0] == 1 else raw[0]
        candidates.append(cand)
    return candidates


def run_benchmark() -> None:
    """Compare oracle / best single hypothesis / greedy composite for first N_BENCHMARK pairs."""
    tracked = read_full_tracked_stack(TRACKED_TIFF)
    non_empty = [t for t in range(tracked.shape[0]) if tracked[t].any()]
    if len(non_empty) < 2:
        print("[benchmark] Need at least 2 non-empty tracked frames.")
        return

    pairs = [(non_empty[i], non_empty[i + 1]) for i in range(min(N_BENCHMARK, len(non_empty) - 1))]
    n_p, _ = list_hypotheses(HYPO_H5)

    print(f"\n{'=' * 88}")
    print(f"BENCHMARK  ({len(pairs)} frame pairs, {n_p} hypotheses)")
    print(f"  MIN_MATCH_IOU={MIN_MATCH_IOU}  ALPHA={ALPHA}")
    print(f"{'=' * 88}")
    print(f"\n{'t':>4}  {'t+1':>4}  {'oracle':>10}  {'best_hyp_p':>10}  "
          f"{'best_hyp':>10}  {'composite':>10}  {'gap_hyp':>8}  {'gap_comp':>9}")
    print("-" * 88)

    for t_src, t_tgt in pairs:
        current = tracked[t_src]
        gt_next = tracked[t_tgt]

        oracle_score, *_ = score_hypothesis(current, gt_next)

        candidates = _load_candidates(t_tgt, n_p, current.shape)

        best_p, best_score = -1, -np.inf
        for p, cand in enumerate(candidates):
            s, *_ = score_hypothesis(current, cand)
            if s > best_score:
                best_score = s
                best_p = p

        composite, _ = build_composite_frame(current, candidates)
        comp_score, *_ = score_hypothesis(current, composite)

        gap_hyp = oracle_score - best_score
        gap_comp = oracle_score - comp_score

        print(f"{t_src:>4}  {t_tgt:>4}  {oracle_score:>10.4f}  {best_p:>10d}  "
              f"{best_score:>10.4f}  {comp_score:>10.4f}  {gap_hyp:>8.4f}  {gap_comp:>9.4f}")

    print("-" * 88)
    print()


def load_zavg(path: str, t_src: int, t_tgt: int) -> np.ndarray:
    """Return (2, Y, X) array for the two display timepoints."""
    raw = np.asarray(tifffile.imread(path))
    if raw.ndim == 2:
        frames = np.stack([raw, raw], axis=0)
    elif raw.ndim == 3:
        frames = np.stack([raw[t_src], raw[t_tgt]], axis=0)
    else:
        raise ValueError(f"Unexpected zavg shape {raw.shape} in {path}")
    return frames.copy()


def main():
    # 1. Identify timepoints
    tracked = read_full_tracked_stack(TRACKED_TIFF)
    non_empty = [t for t in range(tracked.shape[0]) if tracked[t].any()]
    if len(non_empty) < 2:
        raise RuntimeError(f"Need at least 2 non-empty tracked frames; found {non_empty}")
    t_src = non_empty[-2]
    t_tgt = non_empty[-1]
    print(f"t_src={t_src}  t_tgt={t_tgt}")

    current_labels = tracked[t_src]
    gt_labels = tracked[t_tgt]

    n_p, _ = list_hypotheses(HYPO_H5)
    print(f"Hypothesis pool: {n_p} parameter sets")

    candidates = _load_candidates(t_tgt, n_p, current_labels.shape)

    # Score every single hypothesis
    records = []
    for p, cand in enumerate(candidates):
        score, n_matched, total_iou, row_ind, col_ind = score_hypothesis(current_labels, cand)
        records.append((p, score, n_matched, total_iou, row_ind, col_ind, cand))

    records_sorted = sorted(records, key=lambda r: r[1], reverse=True)
    print(f"\n{'p':>4}  {'score':>8}  {'n_matched':>9}  {'total_iou':>9}")
    print("-" * 40)
    for p, score, n_matched, total_iou, *_ in records_sorted:
        print(f"{p:>4}  {score:>8.3f}  {n_matched:>9d}  {total_iou:>9.3f}")

    best = records_sorted[0]
    p_star, score_star, n_matched_star, total_iou_star, row_ind_star, col_ind_star, cand_star = best
    print(f"\nBest single hypothesis: p*={p_star}  score={score_star:.3f}  "
          f"n_matched={n_matched_star}  total_iou={total_iou_star:.3f}")

    # Build composite and report improvement
    composite, per_cell_info = build_composite_frame(current_labels, candidates)
    comp_score, comp_n_matched, comp_iou, *_ = score_hypothesis(current_labels, composite)
    print(f"Composite frame:        score={comp_score:.3f}  "
          f"n_matched={comp_n_matched}  total_iou={comp_iou:.3f}  "
          f"(+{comp_score - score_star:.3f} vs best single)")

    # Count how many cells were sourced from each hypothesis
    source_counts: dict[int, int] = {}
    for c_id, (_, winning_p, _) in per_cell_info.items():
        if winning_p >= 0:
            source_counts[winning_p] = source_counts.get(winning_p, 0) + 1
    top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"Top hypothesis sources: {top_sources}")

    # Build propagated frame from best single hypothesis (for comparison in napari)
    Y, X = cand_star.shape
    propagated = np.zeros((Y, X), dtype=np.uint32)
    for cur_idx, cand_idx in zip(row_ind_star, col_ind_star):
        propagated[cand_star == cand_idx + 1] = cur_idx + 1

    cell_zavg = load_zavg(CELL_ZAVG_TIFF, t_src, t_tgt)
    nuc_zavg = load_zavg(NUC_ZAVG_TIFF, t_src, t_tgt)

    tracked_stack    = np.stack([tracked[t_src], gt_labels],   axis=0)
    propagated_stack = np.stack([tracked[t_src], propagated],  axis=0)
    composite_stack  = np.stack([tracked[t_src], composite],   axis=0)

    viewer = napari.Viewer()
    viewer.add_image(cell_zavg, name="cell_zavg", colormap="gray", blending="additive")
    viewer.add_image(nuc_zavg, name="nucleus_zavg", colormap="bop orange", blending="additive")
    viewer.add_labels(tracked_stack,    name="tracked (oracle)")
    viewer.add_labels(propagated_stack, name=f"best single (p={p_star})")
    viewer.add_labels(composite_stack,  name="composite")

    napari.run()


if __name__ == "__main__":
    run_benchmark()
    main()
