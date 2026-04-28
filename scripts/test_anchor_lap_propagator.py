"""
Propagation-strategy benchmark: compare hypothesis-selection strategies over
N_PREDICT frames starting from N_SEED ground-truth seed frames.

Strategies
----------
Closed-loop (each frame uses the previous prediction as input):
1. best_single     — whole-frame best hypothesis by IoU score
2. composite_iou   — per-cell best match by IoU
3. composite_cs    — per-cell best match by centroid proximity + size similarity
4. composite_pcs   — per-cell best match by predicted centroid proximity + size similarity
5. comp_glap       — global LAP over all (hypothesis, candidate) pairs flattened

Open-loop (plans full trajectory from seed in one pass — no error compounding):
6. viterbi         — per-cell Viterbi DP: IoU match to seed at t=0, centroid+size
                     transition cost between consecutive candidates thereafter
"""

import numpy as np
import tifffile
import napari
from collections import defaultdict, deque
from scipy.ndimage import center_of_mass as _center_of_mass
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
PROX_SIGMA = 20.0
GATE_RADIUS = 50  # pixels — used by predicted-centroid strategy

# --- Viterbi-specific constants ---
VITERBI_GATE_RADIUS   = 80     # max inter-frame displacement captured by transition gate
VITERBI_TRANS_SIGMA   = 40.0   # sigma for transition centroid-proximity (wider → less movement penalty)
VITERBI_VEL_SIGMA     = 60.0   # sigma for velocity-prediction unary
VITERBI_TRANS_W       = 1.0    # weight: transition smoothness
VITERBI_VEL_W         = 1.5    # weight: velocity-prediction unary (strongest signal)
VITERBI_AREA_W        = 0.5    # weight: area consistency unary
VITERBI_FALLBACK_COST = 1.5    # score penalty when no gated match found (guarantees assignment)

# --- Benchmark constants ---
N_SEED = 5
N_PREDICT = 5


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_cell_stats(labels: np.ndarray) -> dict[int, tuple[float, float, int]]:
    """Return {cell_id: (cy, cx, area)} for all non-zero labels."""
    ids = [int(v) for v in np.unique(labels) if v != 0]
    if not ids:
        return {}
    coms = _center_of_mass(np.ones_like(labels), labels, ids)
    if len(ids) == 1:
        coms = [coms]
    areas = np.bincount(labels.ravel(), minlength=max(ids) + 1)
    return {cid: (float(com[0]), float(com[1]), int(areas[cid]))
            for cid, com in zip(ids, coms)}


class TrajectoryState:
    """Accumulates per-cell centroids and predicts next positions."""

    def __init__(self, max_history: int = 5) -> None:
        self._history: dict[int, deque[tuple[int, float, float]]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

    def update(self, t: int, labels: np.ndarray) -> None:
        for cell_id, (cy, cx, _) in get_cell_stats(labels).items():
            self._history[cell_id].append((t, cy, cx))

    def predict_centroid(self, cell_id: int, t_next: int) -> tuple[float, float] | None:
        hist = self._history.get(cell_id)
        if not hist:
            return None
        if len(hist) == 1:
            return hist[-1][1], hist[-1][2]
        t1, y1, x1 = hist[-2]
        t2, y2, x2 = hist[-1]
        dt = t2 - t1
        dt_next = t_next - t2
        return y2 + (y2 - y1) / dt * dt_next, x2 + (x2 - x1) / dt * dt_next


def compute_iou_matrix(current: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return (n_cur_cells, n_cand_cells) IoU matrix, excluding background."""
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
    """Return (score, n_matched, total_iou, row_ind, col_ind) via LAP on IoU."""
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


def propagate_ids(current: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Assign IDs from current to candidate pixels via LAP on IoU."""
    if candidate.max() == 0:
        return np.zeros(candidate.shape, dtype=np.uint32)
    iou = compute_iou_matrix(current, candidate)
    row_ind, col_ind = linear_sum_assignment(iou, maximize=True)
    matched_iou = iou[row_ind, col_ind]
    mask = matched_iou >= MIN_MATCH_IOU
    row_ind, col_ind = row_ind[mask], col_ind[mask]
    out = np.zeros(candidate.shape, dtype=np.uint32)
    for cur_idx, cand_idx in zip(row_ind, col_ind):
        out[candidate == cand_idx + 1] = cur_idx + 1
    return out


def evaluate_iou_against_gt(
    predicted: np.ndarray, ground_truth: np.ndarray
) -> tuple[float, int, int]:
    """Sum per-cell IoU matched by shared cell ID. Returns (total_iou, n_shared, n_gt)."""
    gt_ids = {int(v) for v in np.unique(ground_truth) if v != 0}
    pred_ids = {int(v) for v in np.unique(predicted) if v != 0}
    shared = gt_ids & pred_ids
    total_iou = 0.0
    for cid in shared:
        gt_mask = ground_truth == cid
        pred_mask = predicted == cid
        inter = int(np.logical_and(gt_mask, pred_mask).sum())
        union = int(np.logical_or(gt_mask, pred_mask).sum())
        if union > 0:
            total_iou += inter / union
    return total_iou, len(shared), len(gt_ids)


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


# ---------------------------------------------------------------------------
# Strategy 2: composite by IoU
# ---------------------------------------------------------------------------

def build_composite_iou(
    current: np.ndarray,
    candidates: list[np.ndarray],
) -> np.ndarray:
    """Per-cell best match by IoU; cells painted highest-IoU-first."""
    n_cur = int(current.max())
    if n_cur == 0:
        return np.zeros_like(current)
    H, W = current.shape
    best_for_cell: list[tuple[float, int, int]] = [(-1.0, -1, -1)] * n_cur
    for p, cand in enumerate(candidates):
        if cand.max() == 0:
            continue
        iou = compute_iou_matrix(current, cand)
        if iou.shape[1] == 0:
            continue
        best_cand_idxs = np.argmax(iou, axis=1)
        best_ious = iou[np.arange(n_cur), best_cand_idxs]
        for c_idx in range(n_cur):
            if best_ious[c_idx] > best_for_cell[c_idx][0]:
                best_for_cell[c_idx] = (float(best_ious[c_idx]), p, int(best_cand_idxs[c_idx]))
    return _paint_composite(current, candidates, best_for_cell, H, W, n_cur)


# ---------------------------------------------------------------------------
# Strategy 3: composite by centroid proximity + size similarity
# ---------------------------------------------------------------------------

def build_composite_centroid_size(
    current: np.ndarray,
    candidates: list[np.ndarray],
    cur_stats: dict | None = None,
) -> np.ndarray:
    """Per-cell best match by current centroid proximity + size similarity."""
    n_cur = int(current.max())
    if n_cur == 0:
        return np.zeros_like(current)
    H, W = current.shape
    if cur_stats is None:
        cur_stats = get_cell_stats(current)
    two_s2 = 2.0 * PROX_SIGMA ** 2
    best_for_cell: list[tuple[float, int, int]] = [(-1.0, -1, -1)] * n_cur
    for p, cand in enumerate(candidates):
        if cand.max() == 0:
            continue
        cand_stats = get_cell_stats(cand)
        n_cand = int(cand.max())
        mat = np.zeros((n_cur, n_cand), dtype=np.float32)
        for i in range(n_cur):
            cy_a, cx_a, area_a = cur_stats.get(i + 1, (0.0, 0.0, 1))
            for j in range(n_cand):
                cy_b, cx_b, area_b = cand_stats.get(j + 1, (0.0, 0.0, 1))
                size_sim = min(area_a, area_b) / max(area_a, area_b)
                d2 = (cy_b - cy_a) ** 2 + (cx_b - cx_a) ** 2
                prox = float(np.exp(-d2 / two_s2))
                mat[i, j] = 0.5 * size_sim + 0.5 * prox
        best_cand_idxs = np.argmax(mat, axis=1)
        best_scores = mat[np.arange(n_cur), best_cand_idxs]
        for c_idx in range(n_cur):
            if best_scores[c_idx] > best_for_cell[c_idx][0]:
                best_for_cell[c_idx] = (float(best_scores[c_idx]), p, int(best_cand_idxs[c_idx]))
    return _paint_composite(current, candidates, best_for_cell, H, W, n_cur)


# ---------------------------------------------------------------------------
# Strategy 4: composite by predicted centroid proximity + size similarity
# ---------------------------------------------------------------------------

def build_composite_pred_centroid_size(
    current: np.ndarray,
    candidates: list[np.ndarray],
    traj: TrajectoryState,
    t_next: int,
    cur_stats: dict | None = None,
) -> np.ndarray:
    """Per-cell best match by predicted centroid proximity + size similarity."""
    n_cur = int(current.max())
    if n_cur == 0:
        return np.zeros_like(current)
    H, W = current.shape
    if cur_stats is None:
        cur_stats = get_cell_stats(current)
    two_s2 = 2.0 * PROX_SIGMA ** 2
    r2 = GATE_RADIUS ** 2
    best_for_cell: list[tuple[float, int, int]] = [(-1.0, -1, -1)] * n_cur
    for p, cand in enumerate(candidates):
        if cand.max() == 0:
            continue
        cand_stats = get_cell_stats(cand)
        n_cand = int(cand.max())
        mat = np.zeros((n_cur, n_cand), dtype=np.float32)
        for i in range(n_cur):
            cy_a, cx_a, area_a = cur_stats.get(i + 1, (0.0, 0.0, 1))
            pred = traj.predict_centroid(i + 1, t_next)
            if pred is None:
                pred = (cy_a, cx_a)
            py, px = pred
            for j in range(n_cand):
                cy_b, cx_b, area_b = cand_stats.get(j + 1, (0.0, 0.0, 1))
                d2_pred = (cy_b - py) ** 2 + (cx_b - px) ** 2
                if d2_pred > r2:
                    continue
                size_sim = min(area_a, area_b) / max(area_a, area_b)
                prox = float(np.exp(-d2_pred / two_s2))
                mat[i, j] = 0.5 * size_sim + 0.5 * prox
        best_cand_idxs = np.argmax(mat, axis=1)
        best_scores = mat[np.arange(n_cur), best_cand_idxs]
        for c_idx in range(n_cur):
            if best_scores[c_idx] > best_for_cell[c_idx][0]:
                best_for_cell[c_idx] = (float(best_scores[c_idx]), p, int(best_cand_idxs[c_idx]))
    return _paint_composite(current, candidates, best_for_cell, H, W, n_cur)


def _paint_composite(
    current: np.ndarray,
    candidates: list[np.ndarray],
    best_for_cell: list[tuple[float, int, int]],
    H: int,
    W: int,
    n_cur: int,
) -> np.ndarray:
    """Paint cells into a composite frame, highest-confidence first."""
    order = sorted(range(n_cur), key=lambda i: best_for_cell[i][0], reverse=True)
    composite = np.zeros((H, W), dtype=np.uint32)
    occupied = np.zeros((H, W), dtype=bool)
    for c_idx in order:
        best_score, best_p, best_cand_idx = best_for_cell[c_idx]
        if best_score < MIN_MATCH_IOU or best_p < 0:
            continue
        cand = candidates[best_p]
        pixels = (cand == best_cand_idx + 1) & ~occupied
        composite[pixels] = c_idx + 1
        occupied |= pixels
    return composite


# ---------------------------------------------------------------------------
# Strategy 5: global LAP over flattened candidate pool
# ---------------------------------------------------------------------------

def build_composite_global_lap(
    current: np.ndarray,
    candidates: list[np.ndarray],
) -> np.ndarray:
    """Globally optimal per-cell assignment via LAP over all candidates across all hypotheses.

    Flattens every (hypothesis, candidate-label) pair into a single pool of K
    columns, builds an N×K IoU matrix, and solves one LAP to jointly maximise
    total IoU.  No two cells can win the same candidate, and the solution is
    globally optimal under the IoU objective.  Remaining cross-hypothesis pixel
    conflicts are resolved by painting in descending score order.
    """
    n_cur = int(current.max())
    if n_cur == 0:
        return np.zeros_like(current)
    H, W = current.shape

    flat: list[tuple[int, int]] = []   # (p, j_0based) for each column
    iou_by_p: dict[int, np.ndarray] = {}
    for p, cand in enumerate(candidates):
        if cand.max() == 0:
            continue
        iou = compute_iou_matrix(current, cand)
        iou_by_p[p] = iou
        for j in range(int(cand.max())):
            flat.append((p, j))

    if not flat:
        return np.zeros_like(current)

    K = len(flat)
    score_mat = np.zeros((n_cur, K), dtype=np.float32)
    for k, (p, j) in enumerate(flat):
        iou = iou_by_p[p]
        if j < iou.shape[1]:
            score_mat[:, k] = iou[:, j]

    row_ind, col_ind = linear_sum_assignment(score_mat, maximize=True)
    matched_scores = score_mat[row_ind, col_ind]

    order = np.argsort(-matched_scores)
    composite = np.zeros((H, W), dtype=np.uint32)
    occupied = np.zeros((H, W), dtype=bool)
    for idx in order:
        c_idx = int(row_ind[idx])
        k = int(col_ind[idx])
        score = float(matched_scores[idx])
        if score < MIN_MATCH_IOU:
            continue
        p, j = flat[k]
        pixels = (candidates[p] == j + 1) & ~occupied
        composite[pixels] = c_idx + 1
        occupied |= pixels

    return composite


# ---------------------------------------------------------------------------
# Strategy 6: per-cell Viterbi over the full prediction horizon (open-loop)
# ---------------------------------------------------------------------------

def build_viterbi_sequence(
    seed: np.ndarray,
    candidates_per_frame: list[list[np.ndarray]],
    *,
    seed_cents: np.ndarray,   # (n_cur, 2) — centroid of each cell in the seed frame
    seed_areas: np.ndarray,   # (n_cur,)   — area of each cell in the seed frame
    velocity: np.ndarray,     # (n_cur, 2) — pixels per frame estimated from seed history
    predict_ts: list[int],    # absolute frame indices of the prediction frames
    seed_t: int,              # absolute frame index of the seed frame
) -> list[np.ndarray]:
    """Globally optimal per-cell trajectory via Viterbi DP — open-loop.

    Scoring per DP step
    -------------------
    t = 0  IoU(seed_cell_i, k)
           + VITERBI_VEL_W  * exp(-d²(predicted_pos_i, centroid_k) / 2σ_vel²)
           + VITERBI_AREA_W * area_sim(seed_area_i, area_k)

    t > 0  max_k' [ dp[t-1][i,k'] + VITERBI_TRANS_W * (centroid_prox(k',k) + size_sim(k',k)) / 2 ]
           + VITERBI_VEL_W  * exp(-d²(predicted_pos_i_at_t, centroid_k) / 2σ_vel²)
           + VITERBI_AREA_W * area_sim(seed_area_i, area_k)

    The velocity term guides each cell toward its predicted position (estimated
    from seed-frame trajectory), while the transition term rewards smooth
    inter-frame movement.  Area consistency is a per-candidate unary that
    discourages large size changes from the known seed.

    Cells with no gated match receive a fallback assignment (best candidate by
    unary terms only, with VITERBI_FALLBACK_COST penalty), ensuring every cell
    is always painted — no disappearances.
    """
    from scipy.spatial import KDTree

    T = len(candidates_per_frame)
    if T == 0:
        return []
    n_cur = int(seed.max())
    if n_cur == 0:
        return [np.zeros_like(seed) for _ in range(T)]
    H, W = seed.shape
    two_s2_trans = 2.0 * VITERBI_TRANS_SIGMA ** 2
    two_s2_vel   = 2.0 * VITERBI_VEL_SIGMA   ** 2

    # Pre-compute flat (p, j) pool, centroids, and areas for every frame.
    flat_per_frame: list[list[tuple[int, int]]] = []
    cent_per_frame: list[np.ndarray] = []   # (K_t, 2)
    area_per_frame: list[np.ndarray] = []   # (K_t,)

    for candidates in candidates_per_frame:
        flat_t: list[tuple[int, int]] = []
        cy_t: list[float] = []
        cx_t: list[float] = []
        area_t: list[float] = []
        for p, cand in enumerate(candidates):
            cs = get_cell_stats(cand)
            for j in range(int(cand.max())):
                cy, cx, area = cs.get(j + 1, (0.0, 0.0, 1))
                flat_t.append((p, j))
                cy_t.append(cy)
                cx_t.append(cx)
                area_t.append(float(max(area, 1)))
        K_t = len(flat_t)
        flat_per_frame.append(flat_t)
        cent_per_frame.append(
            np.column_stack([cy_t, cx_t]) if K_t > 0 else np.empty((0, 2), dtype=np.float64)
        )
        area_per_frame.append(
            np.asarray(area_t, dtype=np.float64) if K_t > 0 else np.empty(0, dtype=np.float64)
        )

    seed_areas_safe = np.maximum(seed_areas, 1.0)  # guard against zero-area seed cells

    def _unary(t: int) -> np.ndarray:
        """Velocity + area unary for frame t: shape (n_cur, K_t)."""
        K_t = len(flat_per_frame[t])
        if K_t == 0:
            return np.empty((n_cur, 0), dtype=np.float64)
        dt = predict_ts[t] - seed_t
        pred = seed_cents + velocity * dt                    # (n_cur, 2) predicted centroids
        cur_cents = cent_per_frame[t]                        # (K_t, 2)
        d2_pred  = np.sum((pred[:, None, :] - cur_cents[None, :, :]) ** 2, axis=2)  # (n_cur, K_t)
        vel_u    = VITERBI_VEL_W  * np.exp(-d2_pred / two_s2_vel)
        cur_areas = area_per_frame[t]                        # (K_t,)
        area_u   = VITERBI_AREA_W * (
            np.minimum(seed_areas_safe[:, None], cur_areas[None, :]) /
            np.maximum(seed_areas_safe[:, None], cur_areas[None, :])
        )
        return vel_u + area_u                                # (n_cur, K_t)

    # ------------------------------------------------------------------
    # t = 0  unary: IoU(seed_i, k) + velocity + area
    # ------------------------------------------------------------------
    iou_by_p: dict[int, np.ndarray] = {}
    for p, cand in enumerate(candidates_per_frame[0]):
        if cand.max() == 0:
            continue
        iou_by_p[p] = compute_iou_matrix(seed, cand)

    K_0 = len(flat_per_frame[0])
    iou_scores = np.zeros((n_cur, K_0), dtype=np.float64)
    col = 0
    for p, cand in enumerate(candidates_per_frame[0]):
        n_cand_p = int(cand.max())
        if n_cand_p == 0:
            continue
        if p in iou_by_p:
            iou_mat = iou_by_p[p]
            n_valid = min(n_cand_p, iou_mat.shape[1])
            iou_scores[:, col:col + n_valid] = iou_mat[:, :n_valid]
        col += n_cand_p

    dp_0 = iou_scores + _unary(0)
    dp_all: list[np.ndarray] = [dp_0]
    bt_all: list[np.ndarray] = [np.zeros((n_cur, K_0), dtype=np.int32)]  # unused at t=0

    # ------------------------------------------------------------------
    # t >= 1  transition (KDTree-gated) + velocity/area unary.
    #
    # Transition gate (VITERBI_GATE_RADIUS) keeps memory O(n_cur × K_t).
    # Velocity and area unary are factored out of the argmax since they
    # don't depend on k_prev:
    #   dp[t][i,k] = max_{k'∈gate(k)} (dp[t-1][i,k'] + trans(k',k))
    #                + vel_unary(i,k) + area_unary(i,k)
    #
    # Cells with no gated match get a fallback: best candidate by unary
    # alone minus VITERBI_FALLBACK_COST, guaranteeing every cell is always
    # assigned (number consistency).
    # ------------------------------------------------------------------
    for t in range(1, T):
        K_t    = len(flat_per_frame[t])
        K_prev = len(flat_per_frame[t - 1])

        if K_t == 0 or K_prev == 0:
            sz = max(K_t, 1)
            dp_all.append(np.full((n_cur, sz), -np.inf, dtype=np.float64))
            bt_all.append(np.zeros((n_cur, sz), dtype=np.int32))
            continue

        dp_prev    = dp_all[t - 1]               # (n_cur, K_prev)
        prev_cents = cent_per_frame[t - 1]       # (K_prev, 2)
        prev_areas = area_per_frame[t - 1]       # (K_prev,)
        cur_cents  = cent_per_frame[t]           # (K_t,    2)
        cur_areas  = area_per_frame[t]           # (K_t,)
        unary_t    = _unary(t)                   # (n_cur,  K_t)

        tree   = KDTree(prev_cents)
        groups = tree.query_ball_point(cur_cents, r=float(VITERBI_GATE_RADIUS))

        dp_new = np.full((n_cur, K_t), -np.inf, dtype=np.float64)
        bt_new = np.zeros((n_cur, K_t), dtype=np.int32)

        for k, k_prev_list in enumerate(groups):
            if not k_prev_list:
                continue
            k_prev_arr = np.asarray(k_prev_list, dtype=np.int32)
            cy_k, cx_k = cur_cents[k]
            area_k     = float(cur_areas[k])

            d2       = np.sum((prev_cents[k_prev_arr] - [cy_k, cx_k]) ** 2, axis=1)
            area_p   = prev_areas[k_prev_arr]
            size_sim = np.minimum(area_p, area_k) / np.maximum(area_p, area_k)
            prox     = np.exp(-d2 / two_s2_trans)
            trans_k  = VITERBI_TRANS_W * 0.5 * (size_sim + prox)  # (neighbours,)

            scores   = dp_prev[:, k_prev_arr] + trans_k[None, :]   # (n_cur, neighbours)
            best_idx = np.argmax(scores, axis=1)                    # (n_cur,)
            # Unary factored out — doesn't affect argmax over k_prev
            dp_new[:, k] = scores[np.arange(n_cur), best_idx] + unary_t[:, k]
            bt_new[:, k] = k_prev_arr[best_idx]

        # Fallback: cells with no gated match at any k.
        # Assign to the best candidate by unary alone with a score penalty
        # so they paint after high-confidence cells but are never dropped.
        no_match = np.all(dp_new == -np.inf, axis=1)              # (n_cur,)
        if no_match.any():
            nm_idx   = np.where(no_match)[0]
            fk       = np.argmax(unary_t[no_match], axis=1)       # (n_no_match,)
            best_prev = np.argmax(dp_prev[nm_idx], axis=1)        # best k_prev regardless of gate
            dp_new[nm_idx, fk] = unary_t[nm_idx, fk] - VITERBI_FALLBACK_COST
            bt_new[nm_idx, fk] = best_prev

        dp_all.append(dp_new)
        bt_all.append(bt_new)

    # ------------------------------------------------------------------
    # Backtrack
    # ------------------------------------------------------------------
    last_dp = dp_all[T - 1]
    if last_dp.shape[1] == 0:
        return [np.zeros_like(seed) for _ in range(T)]

    optimal_k: list[np.ndarray] = [np.zeros(n_cur, dtype=np.int32)] * T
    optimal_k[T - 1] = np.argmax(last_dp, axis=1)
    for t in range(T - 2, -1, -1):
        optimal_k[t] = bt_all[t + 1][np.arange(n_cur), optimal_k[t + 1]]

    # ------------------------------------------------------------------
    # Assemble composite frames — always paint every cell (no threshold).
    # High-confidence cells (highest dp score) paint first and claim pixels;
    # fallback cells fill whatever remains.
    # ------------------------------------------------------------------
    result: list[np.ndarray] = []
    for t, candidates in enumerate(candidates_per_frame):
        K_t = len(flat_per_frame[t])
        composite = np.zeros((H, W), dtype=np.uint32)
        occupied  = np.zeros((H, W), dtype=bool)
        if K_t == 0:
            result.append(composite)
            continue

        cell_scores = dp_all[t][np.arange(n_cur), optimal_k[t]]
        order = np.argsort(-cell_scores)
        for i in order:
            k = int(optimal_k[t][i])
            p, j = flat_per_frame[t][k]
            pixels = (candidates[p] == j + 1) & ~occupied
            composite[pixels] = i + 1
            occupied |= pixels
        result.append(composite)

    return result


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark() -> None:
    tracked = read_full_tracked_stack(TRACKED_TIFF)
    non_empty = [t for t in range(tracked.shape[0]) if tracked[t].any()]

    if len(non_empty) < N_SEED + N_PREDICT:
        print(f"Need at least {N_SEED + N_PREDICT} non-empty frames, found {len(non_empty)}.")
        return

    seed_frames = non_empty[:N_SEED]
    predict_frames = non_empty[N_SEED: N_SEED + N_PREDICT]
    n_p, _ = list_hypotheses(HYPO_H5)

    traj_pcs = TrajectoryState()
    for t in seed_frames:
        traj_pcs.update(t, tracked[t])

    seed_labels = tracked[seed_frames[-1]]
    cur_single  = seed_labels
    cur_iou     = seed_labels
    cur_cs      = seed_labels
    cur_pcs     = seed_labels
    cur_lap     = seed_labels

    # Pre-load all candidates so Viterbi can plan the full horizon in one pass
    # and the per-frame strategies can reuse without re-reading from disk.
    shape = tracked[predict_frames[0]].shape
    all_candidates = [_load_candidates(t, n_p, shape) for t in predict_frames]

    # Compute per-cell velocity and seed geometry for Viterbi.
    # Use the last two seed frames to estimate per-cell displacement per frame.
    n_cur_vit    = int(seed_labels.max())
    stats_s1     = get_cell_stats(tracked[seed_frames[-2]])
    stats_s2     = get_cell_stats(tracked[seed_frames[-1]])
    dt_seed      = seed_frames[-1] - seed_frames[-2]
    seed_cents_v = np.array([[stats_s2.get(i+1,(0.,0.,1.))[0],
                              stats_s2.get(i+1,(0.,0.,1.))[1]] for i in range(n_cur_vit)])
    seed_areas_v = np.array([float(stats_s2.get(i+1,(0.,0.,1.))[2]) for i in range(n_cur_vit)])
    vel_v        = np.zeros((n_cur_vit, 2))
    for i in range(n_cur_vit):
        s1, s2 = stats_s1.get(i+1), stats_s2.get(i+1)
        if s1 and s2:
            vel_v[i] = [(s2[0]-s1[0])/dt_seed, (s2[1]-s1[1])/dt_seed]

    # Strategy 6: Viterbi — planned once over the full prediction horizon.
    viterbi_preds = build_viterbi_sequence(
        seed_labels, all_candidates,
        seed_cents=seed_cents_v,
        seed_areas=seed_areas_v,
        velocity=vel_v,
        predict_ts=predict_frames,
        seed_t=seed_frames[-1],
    )

    W = 12
    SEP = 116
    print(f"\n{'=' * SEP}")
    print(f"BENCHMARK  seed={seed_frames}  predict={predict_frames}  hypotheses={n_p}")
    print(f"  MIN_MATCH_IOU={MIN_MATCH_IOU}  ALPHA={ALPHA}  PROX_SIGMA={PROX_SIGMA}  GATE_RADIUS={GATE_RADIUS}")
    print(f"  scores normalized: total_iou / n_gt_cells  (1.0 = perfect match)")
    print(f"  * closed-loop strategies feed each predicted frame back as input")
    print(f"  * viterbi is open-loop: plans full trajectory from seed in one pass")
    print(f"{'=' * SEP}")
    print(f"{'t_tgt':>6}  {'gt_cells':>8}  "
          f"{'best_single':>{W}}  {'comp_iou':>{W}}  {'comp_cs':>{W}}  {'comp_pcs':>{W}}  {'comp_glap':>{W}}  {'viterbi':>{W}}")
    print("-" * SEP)

    totals = [0.0] * 6
    n_frames = 0

    for frame_idx, t_tgt in enumerate(predict_frames):
        gt_frame   = tracked[t_tgt]
        candidates = all_candidates[frame_idx]
        n_gt       = int(len({int(v) for v in np.unique(gt_frame) if v != 0}))
        norm       = n_gt if n_gt > 0 else 1

        # Strategy 1: best single hypothesis by IoU
        best_p, best_score = -1, -np.inf
        for p, cand in enumerate(candidates):
            s, *_ = score_hypothesis(cur_single, cand)
            if s > best_score:
                best_score, best_p = s, p
        pred_single = propagate_ids(cur_single, candidates[best_p])
        iou_single, *_ = evaluate_iou_against_gt(pred_single, gt_frame)

        # Strategy 2: composite by IoU
        pred_iou = build_composite_iou(cur_iou, candidates)
        iou_comp_iou, *_ = evaluate_iou_against_gt(pred_iou, gt_frame)

        # Strategy 3: composite by centroid proximity + size similarity
        cs_stats = get_cell_stats(cur_cs)
        pred_cs = build_composite_centroid_size(cur_cs, candidates, cs_stats)
        iou_comp_cs, *_ = evaluate_iou_against_gt(pred_cs, gt_frame)

        # Strategy 4: composite by predicted centroid proximity + size similarity
        pcs_stats = get_cell_stats(cur_pcs)
        pred_pcs = build_composite_pred_centroid_size(cur_pcs, candidates, traj_pcs, t_tgt, pcs_stats)
        iou_comp_pcs, *_ = evaluate_iou_against_gt(pred_pcs, gt_frame)

        # Strategy 5: global LAP over flattened candidate pool
        pred_lap = build_composite_global_lap(cur_lap, candidates)
        iou_comp_lap, *_ = evaluate_iou_against_gt(pred_lap, gt_frame)

        # Strategy 6: Viterbi (pre-computed above)
        pred_viterbi = viterbi_preds[frame_idx]
        iou_viterbi, *_ = evaluate_iou_against_gt(pred_viterbi, gt_frame)

        s1 = iou_single   / norm
        s2 = iou_comp_iou / norm
        s3 = iou_comp_cs  / norm
        s4 = iou_comp_pcs / norm
        s5 = iou_comp_lap / norm
        s6 = iou_viterbi  / norm

        for idx, s in enumerate([s1, s2, s3, s4, s5, s6]):
            totals[idx] += s
        n_frames += 1

        print(f"{t_tgt:>6}  {n_gt:>8d}  "
              f"{s1:>{W}.4f}  {s2:>{W}.4f}  {s3:>{W}.4f}  {s4:>{W}.4f}  {s5:>{W}.4f}  {s6:>{W}.4f}")

        cur_single = pred_single
        cur_iou    = pred_iou
        cur_cs     = pred_cs
        cur_pcs    = pred_pcs
        cur_lap    = pred_lap
        traj_pcs.update(t_tgt, pred_pcs)

    mean = [v / n_frames for v in totals] if n_frames > 0 else totals
    print("-" * SEP)
    print(f"{'MEAN':>6}  {'':>8}  "
          f"{mean[0]:>{W}.4f}  {mean[1]:>{W}.4f}  {mean[2]:>{W}.4f}  {mean[3]:>{W}.4f}  {mean[4]:>{W}.4f}  {mean[5]:>{W}.4f}")
    print()


# ---------------------------------------------------------------------------
# Visual inspection (napari)
# ---------------------------------------------------------------------------

def load_zavg(path: str, t_src: int, t_tgt: int) -> np.ndarray:
    raw = np.asarray(tifffile.imread(path))
    if raw.ndim == 2:
        return np.stack([raw, raw], axis=0).copy()
    if raw.ndim == 3:
        return np.stack([raw[t_src], raw[t_tgt]], axis=0).copy()
    raise ValueError(f"Unexpected zavg shape {raw.shape}")


def main():
    tracked = read_full_tracked_stack(TRACKED_TIFF)
    non_empty = [t for t in range(tracked.shape[0]) if tracked[t].any()]
    if len(non_empty) < 2:
        raise RuntimeError(f"Need at least 2 non-empty tracked frames; found {non_empty}")
    t_src = non_empty[-2]
    t_tgt = non_empty[-1]
    print(f"t_src={t_src}  t_tgt={t_tgt}")

    current = tracked[t_src]
    gt      = tracked[t_tgt]
    n_p, _  = list_hypotheses(HYPO_H5)
    print(f"Hypothesis pool: {n_p} parameter sets")

    candidates = _load_candidates(t_tgt, n_p, current.shape)

    best_p, best_score = -1, -np.inf
    for p, cand in enumerate(candidates):
        s, *_ = score_hypothesis(current, cand)
        if s > best_score:
            best_score, best_p = s, p
    pred_single = propagate_ids(current, candidates[best_p])
    pred_iou    = build_composite_iou(current, candidates)

    print(f"Best single: p={best_p}  score={best_score:.3f}")

    cell_zavg = load_zavg(CELL_ZAVG_TIFF, t_src, t_tgt)
    nuc_zavg  = load_zavg(NUC_ZAVG_TIFF,  t_src, t_tgt)

    viewer = napari.Viewer()
    viewer.add_image(cell_zavg, name="cell_zavg",    colormap="gray",       blending="additive")
    viewer.add_image(nuc_zavg,  name="nucleus_zavg", colormap="bop orange", blending="additive")
    viewer.add_labels(np.stack([current, gt],          axis=0), name="ground truth")
    viewer.add_labels(np.stack([current, pred_single], axis=0), name=f"best_single (p={best_p})")
    viewer.add_labels(np.stack([current, pred_iou],    axis=0), name="composite_iou")
    napari.run()


if __name__ == "__main__":
    run_benchmark()
    main()
