"""Anchor-LAP propagator: pick the globally best hypothesis frame and propagate IDs via LAP."""

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
    iou = np.where(union > 0, inter / union, 0.0)
    return iou


def score_hypothesis(current: np.ndarray, candidate: np.ndarray):
    """Return (score, n_matched, total_iou, row_ind, col_ind) for one hypothesis candidate."""
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

    # 2. Score every hypothesis
    n_p, _ = list_hypotheses(HYPO_H5)
    print(f"Hypothesis pool: {n_p} parameter sets")

    records = []
    for p in range(n_p):
        raw = read_hypothesis_labels(HYPO_H5, t_tgt, p)
        # raw is (Z, Y, X); squeeze Z (typically Z=1) to get (Y, X)
        candidate = raw.squeeze(axis=0) if raw.ndim == 3 and raw.shape[0] == 1 else raw[0]
        score, n_matched, total_iou, row_ind, col_ind = score_hypothesis(current_labels, candidate)
        records.append((p, score, n_matched, total_iou, row_ind, col_ind, candidate))

    # 3. Pick winner and print table
    records_sorted = sorted(records, key=lambda r: r[1], reverse=True)
    print(f"\n{'p':>4}  {'score':>8}  {'n_matched':>9}  {'total_iou':>9}")
    print("-" * 40)
    for p, score, n_matched, total_iou, *_ in records_sorted:
        print(f"{p:>4}  {score:>8.3f}  {n_matched:>9d}  {total_iou:>9.3f}")

    best = records_sorted[0]
    p_star, score_star, n_matched_star, total_iou_star, row_ind_star, col_ind_star, cand_star = best
    print(f"\nWinner: p*={p_star}  score={score_star:.3f}  n_matched={n_matched_star}  total_iou={total_iou_star:.3f}")

    # 4. Build propagated frame
    Y, X = cand_star.shape
    propagated = np.zeros((Y, X), dtype=np.uint32)
    # row_ind_star indexes into current labels 1..n_cur (0-based offset by 1)
    # col_ind_star indexes into candidate labels 1..n_cand (0-based offset by 1)
    for cur_idx, cand_idx in zip(row_ind_star, col_ind_star):
        cur_id = cur_idx + 1
        cand_id = cand_idx + 1
        propagated[cand_star == cand_id] = cur_id

    # 5. napari viewer
    cell_zavg = load_zavg(CELL_ZAVG_TIFF, t_src, t_tgt)
    nuc_zavg = load_zavg(NUC_ZAVG_TIFF, t_src, t_tgt)

    tracked_stack = np.stack([tracked[t_src], gt_labels], axis=0)
    propagated_stack = np.stack([tracked[t_src], propagated], axis=0)

    viewer = napari.Viewer()
    viewer.add_image(cell_zavg, name="cell_zavg", colormap="gray", blending="additive")
    viewer.add_image(nuc_zavg, name="nucleus_zavg", colormap="bop orange", blending="additive")
    viewer.add_labels(tracked_stack, name="tracked")
    viewer.add_labels(propagated_stack, name="propagated")

    napari.run()


if __name__ == "__main__":
    main()
