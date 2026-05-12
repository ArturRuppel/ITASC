"""Cell label ICM solver with staged API (Initialize → Refine → Commit).

Provides a decomposed pipeline for cell boundary optimization:

- ``initialize_icm``: compute geodesic unary costs and spatial/temporal
  pairwise weights, then build initial labels (nucleus-only, argmin, or
  watershed).  Returns a :class:`CellICMState` that caches all
  energy-landscape data, plus the initial label array.

- ``refine_icm``: run *N* Iterated Conditional Modes (ICM) sweeps on an
  existing label array using a previously computed ``CellICMState``.
  Can be called repeatedly for incremental refinement, interleaved with
  manual corrections in the napari viewer.

- ``commit_labels``: write the current label array to a TIFF file.

Backward-compatible monolithic entry points are preserved:

- ``segment_cells_icm``: run the full pipeline on in-memory arrays.
- ``run_cell_icm_from_pos_dir``: load TIFFs from disk, run full pipeline.

The ICM solver uses a sequential Gauss-Seidel raster sweep (Numba JIT)
with 8-connected spatial neighbours and spatiotemporal face-diagonal edges.
"""
from __future__ import annotations

import hashlib
import math
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

import h5py
import numba
import numpy as np
import tifffile
from skimage.graph import MCP_Geometric

__all__ = [
    "CellLabelICMParams",
    "CellICMState",
    "initialize_icm",
    "refine_icm",
    "commit_labels",
    "segment_cells_icm",
    "run_cell_icm_from_pos_dir",
]

# ── Constants ────────────────────────────────────────────────────────────────

_INF: float = 1e9
_DIAG_SCALE: float = 1.0 / math.sqrt(2.0)


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class CellLabelICMParams:
    """Parameters for the cell label ICM pipeline."""

    alpha_unary: float = 4.0
    """Contour weight in the geodesic cost field: ``1 + alpha_unary * contour``."""

    lambda_s: float = 1.0
    """Spatial pairwise Potts weight."""

    beta_s: float = 5.0
    """Contour sensitivity in spatial pairwise: ``exp(-beta_s * avg_contour)``."""

    lambda_t: float = 1.0
    """Temporal pairwise Potts weight."""

    n_iters: int = 3
    """Number of ICM sweeps."""

    min_round_flips: int = 0
    """Stop early if a round has fewer flips than this."""

    lambda_area: float = 0.0
    """Weight for per-label frame-to-frame area-change penalty."""

    gamma_unary: float = 0.0
    """Weight for ``(1 - foreground_score)`` added to the geodesic cost field."""

    init_mode: str = "nuclei"
    """Label initialisation: ``"nuclei"`` | ``"unary"`` | ``"watershed"``."""

    n_workers: int = 1
    """Parallel worker processes for geodesic unary computation.
    1 = sequential.  Values > 1 use fork-based multiprocessing
    to compute frames in parallel."""


@dataclass
class CellICMState:
    """Cached energy-landscape data for incremental ICM refinement.

    Created by :func:`initialize_icm` and consumed by :func:`refine_icm`.
    All arrays are stored as their solver-ready dtypes (float32 / uint32 /
    bool).  The state is **read-only** once created — it describes the energy
    landscape, not the current solution.
    """

    fg_mask: np.ndarray = field(repr=False)
    """(T, Y, X) bool — foreground mask (includes nucleus pixels)."""

    nuc_tracks: np.ndarray = field(repr=False)
    """(T, Y, X) uint32 — nucleus track IDs (0 = no nucleus)."""

    label_ids: np.ndarray = field(repr=False)
    """(K,) uint32 — sorted global set of label (track) IDs."""

    unary_dense: np.ndarray = field(repr=False)
    """(T, Y, X, K) float32 — dense unary cost array.  Dead / background
    entries are ``_INF``."""

    # Spatial pairwise weights — all (T, Y, X) float32
    h: np.ndarray = field(repr=False)
    v: np.ndarray = field(repr=False)
    dr: np.ndarray = field(repr=False)
    dl: np.ndarray = field(repr=False)

    # Temporal pairwise weights — all (T, Y, X) float32
    tw: np.ndarray = field(repr=False)
    tw_ty_dn: np.ndarray = field(repr=False)
    tw_ty_up: np.ndarray = field(repr=False)
    tw_tx_r: np.ndarray = field(repr=False)
    tw_tx_l: np.ndarray = field(repr=False)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.fg_mask.shape  # type: ignore[return-value]

    @property
    def n_labels(self) -> int:
        return len(self.label_ids)


# ── Internal: pairwise weights ───────────────────────────────────────────────

def _compute_pairwise_weights(
    fg_mask: np.ndarray,
    boundary_signal: np.ndarray,
    lambda_s: float,
    beta_s: float,
    lambda_t: float,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Compute per-pixel spatial + temporal Potts pairwise weights.

    Returns ``(h, v, dr, dl, tw, tw_ty_dn, tw_ty_up, tw_tx_r, tw_tx_l)``
    — nine ``(T, Y, X)`` float32 arrays.
    """
    T, Y, X = fg_mask.shape
    fg = fg_mask.astype(bool)
    c = boundary_signal

    h = np.zeros((T, Y, X), dtype=np.float32)
    h[:, :, :-1] = (
        lambda_s * np.exp(-beta_s * 0.5 * (c[:, :, :-1] + c[:, :, 1:]))
        * (fg[:, :, :-1] & fg[:, :, 1:])
    ).astype(np.float32)

    v = np.zeros((T, Y, X), dtype=np.float32)
    v[:, :-1, :] = (
        lambda_s * np.exp(-beta_s * 0.5 * (c[:, :-1, :] + c[:, 1:, :]))
        * (fg[:, :-1, :] & fg[:, 1:, :])
    ).astype(np.float32)

    dr = np.zeros((T, Y, X), dtype=np.float32)
    dr[:, :-1, :-1] = (
        _DIAG_SCALE * lambda_s
        * np.exp(-beta_s * 0.5 * (c[:, :-1, :-1] + c[:, 1:, 1:]))
        * (fg[:, :-1, :-1] & fg[:, 1:, 1:])
    ).astype(np.float32)

    dl = np.zeros((T, Y, X), dtype=np.float32)
    dl[:, :-1, 1:] = (
        _DIAG_SCALE * lambda_s
        * np.exp(-beta_s * 0.5 * (c[:, :-1, 1:] + c[:, 1:, :-1]))
        * (fg[:, :-1, 1:] & fg[:, 1:, :-1])
    ).astype(np.float32)

    tw = np.zeros((T, Y, X), dtype=np.float32)
    if T > 1:
        tw[:-1, :, :] = (lambda_t * (fg[:-1] & fg[1:])).astype(np.float32)

    tw_ty_dn = np.zeros((T, Y, X), dtype=np.float32)
    tw_ty_up = np.zeros((T, Y, X), dtype=np.float32)
    tw_tx_r = np.zeros((T, Y, X), dtype=np.float32)
    tw_tx_l = np.zeros((T, Y, X), dtype=np.float32)
    if T > 1:
        tw_ty_dn[:-1, :-1, :] = (
            _DIAG_SCALE * lambda_t * (fg[:-1, :-1, :] & fg[1:, 1:, :])
        ).astype(np.float32)
        tw_ty_up[:-1, 1:, :] = (
            _DIAG_SCALE * lambda_t * (fg[:-1, 1:, :] & fg[1:, :-1, :])
        ).astype(np.float32)
        tw_tx_r[:-1, :, :-1] = (
            _DIAG_SCALE * lambda_t * (fg[:-1, :, :-1] & fg[1:, :, 1:])
        ).astype(np.float32)
        tw_tx_l[:-1, :, 1:] = (
            _DIAG_SCALE * lambda_t * (fg[:-1, :, 1:] & fg[1:, :, :-1])
        ).astype(np.float32)

    return h, v, dr, dl, tw, tw_ty_dn, tw_ty_up, tw_tx_r, tw_tx_l


# ── Internal: geodesic unaries ───────────────────────────────────────────────

def _compute_frame_geodesic(
    contours_t: np.ndarray,
    fg_t: np.ndarray,
    nuc_t: np.ndarray,
    label_ids: np.ndarray,
    alpha_unary: float,
    fg_scores_t: np.ndarray | None = None,
    gamma_unary: float = 0.0,
) -> dict[int, np.ndarray]:
    """Compute normalised geodesic unaries for all alive labels in one frame.

    Returns ``{k: (Y, X) float32}`` — normalised geodesic distance per label.
    Dead / background entries are ``_INF``.

    The MCP object is created once and reused for all labels in the frame
    (the cost field depends only on the contour map, not the label).
    """
    Y, X = fg_t.shape

    # Build cost field — shared across all labels
    cost_field = np.full((Y, X), np.inf, dtype=np.float32)
    c = 1.0 + alpha_unary * contours_t[fg_t]
    if gamma_unary != 0.0 and fg_scores_t is not None:
        c = c + gamma_unary * (1.0 - np.clip(fg_scores_t[fg_t], 0.0, 1.0))
    cost_field[fg_t] = c

    alive = [int(k) for k in label_ids if np.any(nuc_t == k)]
    if not alive:
        return {}

    # Single MCP object reused for all labels in this frame
    mcp = MCP_Geometric(cost_field, fully_connected=True)

    raw: dict[int, np.ndarray] = {}
    for k in alive:
        starts = [
            tuple(int(v) for v in coord)
            for coord in np.argwhere(nuc_t == k)
        ]
        cum, _ = mcp.find_costs(starts)
        d = cum.astype(np.float32)
        d[~fg_t] = np.inf
        raw[k] = d

    # Per-frame median normalisation
    all_finite = np.concatenate([d[np.isfinite(d)] for d in raw.values()])
    med = float(np.median(all_finite)) if all_finite.size > 0 else 1.0
    if med <= 0.0:
        med = 1.0

    result: dict[int, np.ndarray] = {}
    for k, d in raw.items():
        nd = d / med
        nd[~np.isfinite(nd)] = _INF
        result[k] = nd

    # Hard nucleus anchors within this frame
    for k in alive:
        k_pix = nuc_t == k
        if not k_pix.any():
            continue
        for j in alive:
            if j != k and j in result:
                result[j][k_pix] = _INF

    return result


# ── Parallel worker globals (fork-inherited, no pickling) ────────────────────

_MP_CONTOURS: np.ndarray | None = None
_MP_FG_MASK: np.ndarray | None = None
_MP_NUC_TRACKS: np.ndarray | None = None
_MP_LABEL_IDS: np.ndarray | None = None
_MP_ALPHA_UNARY: float = 0.0
_MP_GAMMA_UNARY: float = 0.0
_MP_FG_SCORES: np.ndarray | None = None


def _geodesic_frame_worker(t: int) -> tuple[int, dict[int, np.ndarray]]:
    """Multiprocessing worker: compute geodesic unaries for frame *t*.

    Reads from module-level globals set before ``Pool`` creation —
    inherited via fork-COW on Linux, zero pickling overhead.
    """
    result = _compute_frame_geodesic(
        _MP_CONTOURS[t],
        _MP_FG_MASK[t],
        _MP_NUC_TRACKS[t],
        _MP_LABEL_IDS,
        _MP_ALPHA_UNARY,
        fg_scores_t=_MP_FG_SCORES[t] if _MP_FG_SCORES is not None else None,
        gamma_unary=_MP_GAMMA_UNARY,
    )
    return t, result


def _compute_geodesic_unaries(
    nuc_tracks: np.ndarray,
    fg_mask: np.ndarray,
    contours: np.ndarray,
    label_ids: np.ndarray,
    alpha_unary: float,
    *,
    foreground_scores: np.ndarray | None = None,
    gamma_unary: float = 0.0,
    n_workers: int = 1,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    """Compute normalised geodesic unary costs for each alive (frame, label).

    When ``n_workers > 1``, frames are computed in parallel using
    fork-based multiprocessing (Linux).  Each worker inherits the input
    arrays via copy-on-write — only the frame index is sent through the
    pipe per task.
    """
    T = fg_mask.shape[0]
    _report = progress_cb or (lambda msg: None)

    if n_workers > 1:
        return _compute_geodesic_unaries_parallel(
            nuc_tracks, fg_mask, contours, label_ids, alpha_unary,
            foreground_scores=foreground_scores,
            gamma_unary=gamma_unary,
            n_workers=n_workers,
            progress_cb=progress_cb,
        )

    # ── Sequential path ──────────────────────────────────────────────
    unary: dict[tuple[int, int], np.ndarray] = {}
    for t in range(T):
        frame_result = _compute_frame_geodesic(
            contours[t], fg_mask[t], nuc_tracks[t], label_ids,
            alpha_unary,
            fg_scores_t=(
                foreground_scores[t] if foreground_scores is not None else None
            ),
            gamma_unary=gamma_unary,
        )
        for k, d in frame_result.items():
            unary[(t, k)] = d
        if progress_cb and ((t + 1) % 10 == 0 or t + 1 == T):
            alive = len(frame_result)
            _report(f"Geodesic unaries: frame {t + 1}/{T}, {alive} alive")

    return unary


def _compute_geodesic_unaries_parallel(
    nuc_tracks: np.ndarray,
    fg_mask: np.ndarray,
    contours: np.ndarray,
    label_ids: np.ndarray,
    alpha_unary: float,
    *,
    foreground_scores: np.ndarray | None = None,
    gamma_unary: float = 0.0,
    n_workers: int = 4,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    """Parallel geodesic unary computation across frames.

    Uses fork-based multiprocessing: input arrays are set as module-level
    globals and inherited by worker processes via COW.  Only the frame
    index (a single int) is sent per task; results (sparse dicts of
    float32 arrays) are returned through the pipe.
    """
    global _MP_CONTOURS, _MP_FG_MASK, _MP_NUC_TRACKS, _MP_LABEL_IDS
    global _MP_ALPHA_UNARY, _MP_GAMMA_UNARY, _MP_FG_SCORES

    T = fg_mask.shape[0]
    _report = progress_cb or (lambda msg: None)
    n_workers = min(n_workers, T, os.cpu_count() or 1)

    # Set globals before fork — workers inherit via COW
    _MP_CONTOURS = contours
    _MP_FG_MASK = fg_mask
    _MP_NUC_TRACKS = nuc_tracks
    _MP_LABEL_IDS = label_ids
    _MP_ALPHA_UNARY = alpha_unary
    _MP_GAMMA_UNARY = gamma_unary
    _MP_FG_SCORES = foreground_scores

    _report(f"Computing geodesic unaries ({n_workers} workers, {T} frames)...")

    unary: dict[tuple[int, int], np.ndarray] = {}
    done = 0

    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers) as pool:
            for t, frame_result in pool.imap_unordered(
                _geodesic_frame_worker, range(T)
            ):
                for k, d in frame_result.items():
                    unary[(t, k)] = d
                done += 1
                if progress_cb and (done % 10 == 0 or done == T):
                    _report(
                        f"Geodesic unaries: {done}/{T} frames "
                        f"({len(frame_result)} labels in frame {t})"
                    )
    finally:
        # Clear globals — don't keep references to large arrays
        _MP_CONTOURS = None
        _MP_FG_MASK = None
        _MP_NUC_TRACKS = None
        _MP_LABEL_IDS = None
        _MP_FG_SCORES = None

    return unary


def _apply_nucleus_anchors(
    unary: dict[tuple[int, int], np.ndarray],
    nuc_tracks: np.ndarray,
    label_ids: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """Re-apply hard nucleus anchors: cost=0 for own label, INF for others."""
    T = nuc_tracks.shape[0]
    label_list = [int(k) for k in label_ids]
    cached_by_t: dict[int, list[int]] = {}
    for (t, j) in unary:
        cached_by_t.setdefault(int(t), []).append(int(j))

    for t in range(T):
        alive = [k for k in label_list if int((nuc_tracks[t] == k).sum()) > 0]
        cached = cached_by_t.get(t, [])
        for k in alive:
            k_pix = nuc_tracks[t] == k
            if (t, k) in unary:
                unary[(t, k)][k_pix] = 0.0
            for j in cached:
                if j != k:
                    unary[(t, j)][k_pix] = _INF
    return unary


def _dict_to_dense_unary(
    unary: dict[tuple[int, int], np.ndarray],
    fg_mask: np.ndarray,
    label_ids: np.ndarray,
) -> np.ndarray:
    """Convert sparse ``{(t, k): (Y, X)}`` to ``(T, Y, X, K)`` float32."""
    T, Y, X = fg_mask.shape
    K = len(label_ids)
    dense = np.full((T, Y, X, K), _INF, dtype=np.float32)
    for ki, k in enumerate(label_ids):
        for t in range(T):
            u = unary.get((t, int(k)))
            if u is not None:
                dense[t, :, :, ki] = u
    return dense


# ── Internal: HDF5 unary cache ───────────────────────────────────────────────

def _unary_cache_key(
    shape: tuple[int, int, int],
    alpha_unary: float,
    gamma_unary: float,
) -> str:
    raw = f"{shape[0]}x{shape[1]}x{shape[2]}_a{alpha_unary:g}_g{gamma_unary:g}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:12]
    return f"unary_{digest}"


def _unary_cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.h5"


def _read_unary_cache(
    cache_dir: Path,
    key: str,
) -> dict[tuple[int, int], np.ndarray] | None:
    path = _unary_cache_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        unary: dict[tuple[int, int], np.ndarray] = {}
        with h5py.File(path, "r") as f:
            grp = f["unaries"]
            for name in grp:
                t_s, k_s = name.split("_", 1)
                unary[(int(t_s), int(k_s))] = grp[name][...].astype(
                    np.float32, copy=False
                )
        return unary
    except Exception:
        return None


def _write_unary_cache(
    cache_dir: Path,
    key: str,
    unary: dict[tuple[int, int], np.ndarray],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _unary_cache_path(cache_dir, key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with h5py.File(tmp, "w") as f:
            grp = f.create_group("unaries")
            grp.attrs["cache_key"] = key
            for (t, k), arr in unary.items():
                grp.create_dataset(
                    f"{int(t)}_{int(k)}",
                    data=np.asarray(arr, dtype=np.float32),
                    compression="lzf",
                )
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


# ── Internal: initialisation helpers ─────────────────────────────────────────

def _unary_elevation_from_dense(
    unary_dense: np.ndarray,
    fg_mask: np.ndarray,
) -> np.ndarray:
    T, Y, X = fg_mask.shape
    elevation = np.min(unary_dense, axis=3)
    for t in range(T):
        finite_fg = fg_mask[t] & (elevation[t] < _INF)
        if finite_fg.any():
            cap = float(np.max(elevation[t][finite_fg]))
            inf_fg = fg_mask[t] & (elevation[t] >= _INF)
            elevation[t][inf_fg] = cap
    return elevation


def _watershed_init(
    fg_mask: np.ndarray,
    nuc_tracks: np.ndarray,
    elevation: np.ndarray,
) -> np.ndarray:
    from skimage.segmentation import watershed

    T, Y, X = fg_mask.shape
    labels = np.zeros((T, Y, X), dtype=np.uint32)
    for t in range(T):
        markers = nuc_tracks[t].astype(np.int32)
        labels[t] = watershed(
            elevation[t].astype(np.float32),
            markers=markers,
            mask=fg_mask[t],
        ).astype(np.uint32)
    return labels


def _argmin_init(
    unary_dense: np.ndarray,
    fg_mask: np.ndarray,
    label_ids: np.ndarray,
) -> np.ndarray:
    best_ki = np.argmin(unary_dense, axis=3)
    return np.where(fg_mask, label_ids[best_ki], 0).astype(np.uint32)


# ── Internal: Numba ICM kernel ───────────────────────────────────────────────

@numba.njit(cache=True)
def _nb_icm_round(
    labels: np.ndarray,
    unary_dense: np.ndarray,
    h_w: np.ndarray,
    v_w: np.ndarray,
    dr_w: np.ndarray,
    dl_w: np.ndarray,
    tw_w: np.ndarray,
    tw_ty_dn_w: np.ndarray,
    tw_ty_up_w: np.ndarray,
    tw_tx_r_w: np.ndarray,
    tw_tx_l_w: np.ndarray,
    fg_mask: np.ndarray,
    label_ids: np.ndarray,
    anchor_label: np.ndarray,
    areas: np.ndarray,
    lambda_area: np.float32,
) -> int:
    """One sequential Gauss-Seidel ICM sweep — raster scan, in-place."""
    T, Y, X = fg_mask.shape
    K = len(label_ids)
    n_flips = 0

    for t in range(T):
        for y in range(Y):
            for x in range(X):
                if anchor_label[t, y, x] > np.uint32(0):
                    continue
                if not fg_mask[t, y, x]:
                    continue

                old_label = labels[t, y, x]

                ji = np.int32(-1)
                for kk in range(K):
                    if label_ids[kk] == old_label:
                        ji = np.int32(kk)
                        break

                best_cost = np.float32(1e30)
                best_k = old_label
                best_ki = np.int32(-1)

                for ki in range(K):
                    u = unary_dense[t, y, x, ki]
                    if u >= np.float32(1e8):
                        continue
                    k = label_ids[ki]

                    if k != old_label:
                        adj = False
                        if x + 1 < X and labels[t, y, x + 1] == k:
                            adj = True
                        elif x > 0 and labels[t, y, x - 1] == k:
                            adj = True
                        elif y + 1 < Y and labels[t, y + 1, x] == k:
                            adj = True
                        elif y > 0 and labels[t, y - 1, x] == k:
                            adj = True
                        elif y + 1 < Y and x + 1 < X and labels[t, y + 1, x + 1] == k:
                            adj = True
                        elif y > 0 and x > 0 and labels[t, y - 1, x - 1] == k:
                            adj = True
                        elif y + 1 < Y and x > 0 and labels[t, y + 1, x - 1] == k:
                            adj = True
                        elif y > 0 and x + 1 < X and labels[t, y - 1, x + 1] == k:
                            adj = True
                        if not adj:
                            continue

                    cost = u

                    if x + 1 < X and fg_mask[t, y, x + 1] and labels[t, y, x + 1] != k:
                        cost += h_w[t, y, x]
                    if x > 0 and fg_mask[t, y, x - 1] and labels[t, y, x - 1] != k:
                        cost += h_w[t, y, x - 1]
                    if y + 1 < Y and fg_mask[t, y + 1, x] and labels[t, y + 1, x] != k:
                        cost += v_w[t, y, x]
                    if y > 0 and fg_mask[t, y - 1, x] and labels[t, y - 1, x] != k:
                        cost += v_w[t, y - 1, x]

                    if y + 1 < Y and x + 1 < X and fg_mask[t, y + 1, x + 1] and labels[t, y + 1, x + 1] != k:
                        cost += dr_w[t, y, x]
                    if y > 0 and x > 0 and fg_mask[t, y - 1, x - 1] and labels[t, y - 1, x - 1] != k:
                        cost += dr_w[t, y - 1, x - 1]
                    if y + 1 < Y and x > 0 and fg_mask[t, y + 1, x - 1] and labels[t, y + 1, x - 1] != k:
                        cost += dl_w[t, y, x]
                    if y > 0 and x + 1 < X and fg_mask[t, y - 1, x + 1] and labels[t, y - 1, x + 1] != k:
                        cost += dl_w[t, y - 1, x + 1]

                    if t + 1 < T and fg_mask[t + 1, y, x] and labels[t + 1, y, x] != k:
                        cost += tw_w[t, y, x]
                    if t > 0 and fg_mask[t - 1, y, x] and labels[t - 1, y, x] != k:
                        cost += tw_w[t - 1, y, x]

                    if t + 1 < T:
                        if y + 1 < Y and fg_mask[t + 1, y + 1, x] and labels[t + 1, y + 1, x] != k:
                            cost += tw_ty_dn_w[t, y, x]
                        if y > 0 and fg_mask[t + 1, y - 1, x] and labels[t + 1, y - 1, x] != k:
                            cost += tw_ty_up_w[t, y, x]
                        if x + 1 < X and fg_mask[t + 1, y, x + 1] and labels[t + 1, y, x + 1] != k:
                            cost += tw_tx_r_w[t, y, x]
                        if x > 0 and fg_mask[t + 1, y, x - 1] and labels[t + 1, y, x - 1] != k:
                            cost += tw_tx_l_w[t, y, x]

                    if t > 0:
                        if y + 1 < Y and fg_mask[t - 1, y + 1, x] and labels[t - 1, y + 1, x] != k:
                            cost += tw_ty_up_w[t - 1, y + 1, x]
                        if y > 0 and fg_mask[t - 1, y - 1, x] and labels[t - 1, y - 1, x] != k:
                            cost += tw_ty_dn_w[t - 1, y - 1, x]
                        if x + 1 < X and fg_mask[t - 1, y, x + 1] and labels[t - 1, y, x + 1] != k:
                            cost += tw_tx_l_w[t - 1, y, x + 1]
                        if x > 0 and fg_mask[t - 1, y, x - 1] and labels[t - 1, y, x - 1] != k:
                            cost += tw_tx_r_w[t - 1, y, x - 1]

                    if lambda_area > np.float32(0.0) and ki != ji:
                        ak = areas[ki, t]
                        aj = areas[ji, t] if ji >= np.int32(0) else np.int32(0)
                        area_delta = np.float32(0.0)
                        if t > 0:
                            area_delta += np.float32(2 * (ak - areas[ki, t - 1]) + 1)
                            if ji >= np.int32(0):
                                area_delta += np.float32(1 - 2 * (aj - areas[ji, t - 1]))
                        if t + 1 < T:
                            area_delta += np.float32(2 * (ak - areas[ki, t + 1]) + 1)
                            if ji >= np.int32(0):
                                area_delta += np.float32(1 - 2 * (aj - areas[ji, t + 1]))
                        cost += lambda_area * area_delta

                    if cost < best_cost:
                        best_cost = cost
                        best_k = k
                        best_ki = np.int32(ki)

                labels[t, y, x] = best_k
                if best_k != old_label:
                    n_flips += 1
                    if best_ki >= np.int32(0):
                        areas[best_ki, t] += np.int32(1)
                    if ji >= np.int32(0):
                        areas[ji, t] -= np.int32(1)

    return n_flips


def _build_areas(labels: np.ndarray, label_ids: np.ndarray) -> np.ndarray:
    K = len(label_ids)
    T = labels.shape[0]
    areas = np.zeros((K, T), dtype=np.int32)
    for ki, k in enumerate(label_ids):
        for t in range(T):
            areas[ki, t] = int(np.count_nonzero(labels[t] == k))
    return areas


# ══════════════════════════════════════════════════════════════════════════════
# Public API — Staged
# ══════════════════════════════════════════════════════════════════════════════

def initialize_icm(
    nuc_tracks: np.ndarray,
    fg_mask: np.ndarray,
    contours: np.ndarray,
    params: CellLabelICMParams,
    *,
    foreground_scores: np.ndarray | None = None,
    cache_dir: Path | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[CellICMState, np.ndarray]:
    """Compute energy terms and build initial labels.

    Parameters
    ----------
    nuc_tracks : (T, Y, X) uint32
    fg_mask : (T, Y, X) bool
    contours : (T, Y, X) float32
    params : CellLabelICMParams
    foreground_scores : (T, Y, X) float32, optional
    cache_dir : Path, optional
        HDF5 unary cache directory.
    progress_cb : callable, optional

    Returns
    -------
    state : CellICMState
    init_labels : (T, Y, X) uint32
    """
    _report = progress_cb or (lambda msg: None)

    fg_mask = fg_mask | (nuc_tracks > 0)

    label_ids = np.array(
        sorted(int(k) for k in np.unique(nuc_tracks) if k > 0),
        dtype=np.uint32,
    )
    T, Y, X = fg_mask.shape
    _report(
        f"Label set: {len(label_ids)} track IDs, "
        f"shape {T}×{Y}×{X}, "
        f"fg_voxels={int(np.count_nonzero(fg_mask))}"
    )

    # ── Pairwise weights (cheap — always recompute) ───────────────────
    _report("Computing pairwise weights...")
    boundary_signal = np.clip(contours, 0.0, 1.0).astype(np.float32, copy=False)
    h, v, dr, dl, tw, tw_ty_dn, tw_ty_up, tw_tx_r, tw_tx_l = (
        _compute_pairwise_weights(
            fg_mask, boundary_signal,
            params.lambda_s, params.beta_s, params.lambda_t,
        )
    )

    # ── Geodesic unaries (expensive — cache + parallel) ───────────────
    cache_key = _unary_cache_key((T, Y, X), params.alpha_unary, params.gamma_unary)
    unary_dict: dict[tuple[int, int], np.ndarray] | None = None

    if cache_dir is not None:
        _report(f"Checking unary cache: {cache_key}")
        unary_dict = _read_unary_cache(cache_dir, cache_key)
        if unary_dict is not None:
            _report(f"Cache hit: {len(unary_dict)} entries loaded.")

    if unary_dict is None:
        t0 = perf_counter()
        unary_dict = _compute_geodesic_unaries(
            nuc_tracks, fg_mask, contours, label_ids, params.alpha_unary,
            foreground_scores=foreground_scores,
            gamma_unary=params.gamma_unary,
            n_workers=params.n_workers,
            progress_cb=progress_cb,
        )
        elapsed = perf_counter() - t0
        _report(f"Geodesic unaries: {len(unary_dict)} entries in {elapsed:.1f}s")
        if cache_dir is not None:
            _report("Writing unary cache...")
            _write_unary_cache(cache_dir, cache_key, unary_dict)

    _apply_nucleus_anchors(unary_dict, nuc_tracks, label_ids)

    # ── Dense unary ───────────────────────────────────────────────────
    _report("Building dense unary array...")
    unary_dense = _dict_to_dense_unary(unary_dict, fg_mask, label_ids)
    del unary_dict

    # ── Initial labels ────────────────────────────────────────────────
    mode = params.init_mode
    if mode == "unary":
        _report("Initialising labels from unary argmin...")
        init_labels = _argmin_init(unary_dense, fg_mask, label_ids)
    elif mode == "watershed":
        _report("Initialising labels via seeded watershed...")
        elevation = _unary_elevation_from_dense(unary_dense, fg_mask)
        init_labels = _watershed_init(fg_mask, nuc_tracks, elevation)
    else:
        _report("Initialising labels from nucleus pixels only...")
        init_labels = np.zeros((T, Y, X), dtype=np.uint32)

    nuc_mask = nuc_tracks > 0
    init_labels[nuc_mask] = nuc_tracks[nuc_mask].astype(np.uint32)

    state = CellICMState(
        fg_mask=fg_mask.astype(bool, copy=False),
        nuc_tracks=nuc_tracks.astype(np.uint32, copy=False),
        label_ids=label_ids,
        unary_dense=unary_dense,
        h=h, v=v, dr=dr, dl=dl,
        tw=tw, tw_ty_dn=tw_ty_dn, tw_ty_up=tw_ty_up,
        tw_tx_r=tw_tx_r, tw_tx_l=tw_tx_l,
    )

    _report("Initialisation complete.")
    return state, init_labels


def refine_icm(
    state: CellICMState,
    labels: np.ndarray,
    n_iters: int = 1,
    *,
    min_round_flips: int = 0,
    lambda_area: float = 0.0,
    progress_cb: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Run *n_iters* ICM sweeps on *labels*.

    Input ``labels`` is **not** modified — a copy is returned.
    """
    _report = progress_cb or (lambda msg: None)

    labels = labels.copy().astype(np.uint32)
    nuc_mask = state.nuc_tracks > 0
    labels[nuc_mask] = state.nuc_tracks[nuc_mask].astype(np.uint32)

    anchor_label = state.nuc_tracks.astype(np.uint32)
    h32 = state.h.astype(np.float32, copy=False)
    v32 = state.v.astype(np.float32, copy=False)
    dr32 = state.dr.astype(np.float32, copy=False)
    dl32 = state.dl.astype(np.float32, copy=False)
    tw32 = state.tw.astype(np.float32, copy=False)
    tw_ty_dn32 = state.tw_ty_dn.astype(np.float32, copy=False)
    tw_ty_up32 = state.tw_ty_up.astype(np.float32, copy=False)
    tw_tx_r32 = state.tw_tx_r.astype(np.float32, copy=False)
    tw_tx_l32 = state.tw_tx_l.astype(np.float32, copy=False)
    lids32 = state.label_ids.astype(np.uint32, copy=False)
    lambda_area32 = np.float32(lambda_area)

    areas = _build_areas(labels, state.label_ids)

    energy_log: list[dict] = []
    for iteration in range(n_iters):
        _report(f"ICM round {iteration + 1}/{n_iters}...")
        t0 = perf_counter()
        n_flips = _nb_icm_round(
            labels, state.unary_dense,
            h32, v32, dr32, dl32, tw32,
            tw_ty_dn32, tw_ty_up32, tw_tx_r32, tw_tx_l32,
            state.fg_mask, lids32, anchor_label, areas, lambda_area32,
        )
        elapsed = perf_counter() - t0
        _report(f"ICM round {iteration + 1}/{n_iters}: {n_flips} flips, {elapsed:.1f}s")
        energy_log.append({
            "iteration": iteration + 1,
            "flips": int(n_flips),
            "elapsed_s": round(elapsed, 2),
        })
        if n_flips == 0:
            _report(f"Converged after round {iteration + 1}.")
            break
        if min_round_flips > 0 and n_flips < min_round_flips:
            _report(f"Stopping: {n_flips} < min_round_flips={min_round_flips}")
            break

    return labels, energy_log


def commit_labels(labels: np.ndarray, output_path: Path | str) -> None:
    """Write label array to TIFF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(output_path),
        labels.astype(np.uint16, copy=False),
        compression="zlib",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Public API — Legacy monolithic wrappers
# ══════════════════════════════════════════════════════════════════════════════

def segment_cells_icm(
    nuc_tracks: np.ndarray,
    fg_mask: np.ndarray,
    contours: np.ndarray,
    params: CellLabelICMParams,
    *,
    foreground_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Run the full pipeline on in-memory arrays."""
    state, init_labels = initialize_icm(
        nuc_tracks, fg_mask, contours, params,
        foreground_scores=foreground_scores,
        progress_cb=lambda msg: print(msg, flush=True),
    )
    labels, _ = refine_icm(
        state, init_labels,
        n_iters=params.n_iters,
        min_round_flips=params.min_round_flips,
        lambda_area=params.lambda_area,
        progress_cb=lambda msg: print(msg, flush=True),
    )
    return labels


def run_cell_icm_from_pos_dir(
    pos_dir: Path | str,
    params: CellLabelICMParams,
) -> np.ndarray:
    """Load TIFFs from disk, run full pipeline."""
    pos_dir = Path(pos_dir)
    nuc_tracks, fg_mask, contours, foreground_scores = _load_pos_dir_inputs(pos_dir)
    return segment_cells_icm(
        nuc_tracks, fg_mask, contours, params,
        foreground_scores=foreground_scores,
    )


def _read_tiff(path: Path, dtype) -> np.ndarray:
    a = np.asarray(tifffile.imread(str(path)), dtype=dtype)
    if a.ndim == 4 and a.shape[1] == 1:
        a = a[:, 0]
    return a


def _load_pos_dir_inputs(
    pos_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    nuc = _read_tiff(pos_dir / "2_nucleus" / "tracked_labels.tif", np.uint32)
    fg = _read_tiff(pos_dir / "3_cell" / "foreground_masks.tif", np.uint8) > 0
    ct = _read_tiff(pos_dir / "3_cell" / "contour_maps.tif", np.float32)
    fg = fg | (nuc > 0)
    fg_score_path = pos_dir / "3_cell" / "foreground_scores.tif"
    fg_scores = (
        _read_tiff(fg_score_path, np.float32) if fg_score_path.exists() else None
    )
    return nuc, fg, ct, fg_scores