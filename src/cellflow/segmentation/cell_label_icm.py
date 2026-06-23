"""Unary-only cell label segmentation (contour-aware geodesic Voronoi).

- ``initialize_icm``: compute geodesic unary costs from the contour/foreground
  cost field and assign each foreground pixel to its nearest nucleus seed
  (per-pixel argmin). Returns a :class:`CellICMState` caching the unary and a
  hard-anchored initial label array.

- ``assemble_cost_field``: build the per-frame geodesic cost field the walk
  traverses (shared with the cell widget's preview).

- ``commit_labels``: write the label array to a TIFF file.

The spatial/temporal Potts pairwise terms and the iterated-conditional-modes
refinement sweep have been removed — with the pairwise weights zeroed the ICM
optimum is exactly the argmin of the unary, so the initialisation *is* the
answer.
"""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from collections.abc import Callable

import h5py
import numpy as np
from skimage.graph import MCP_Geometric

from cellflow.core.tiff import imwrite_grayscale

__all__ = [
    "CellLabelICMParams",
    "CellICMState",
    "assemble_cost_field",
    "balance_strength_to_weights",
    "initialize_icm",
    "commit_labels",
]

# ── Constants ────────────────────────────────────────────────────────────────

_INF: float = 1e9


# ── Cost-field parameterisation ──────────────────────────────────────────────

def balance_strength_to_weights(
    balance: float, feature_strength: float
) -> tuple[float, float]:
    """Map the (balance, feature_strength) knobs to raw cost-field weights.

    The geodesic cost field is ``1 + alpha * contour + gamma * (1 - fg_score)``.
    Because the final labels come from a per-pixel ``argmin`` over geodesic
    distances, multiplying the whole field by any positive constant leaves the
    result unchanged — overall scale is a free gauge.  That leaves exactly two
    observable degrees of freedom, exposed here as:

    - ``balance`` (``r`` in ``[0, 1]``) — the contour↔foreground split
      (``1`` = pure contour, ``0`` = pure foreground).
    - ``feature_strength`` (``s >= 0``) — how strongly either feature bends the
      walk away from a plain distance Voronoi, relative to the fixed base of 1
      (``0`` = pure distance Voronoi).

    with ``alpha = s * r`` and ``gamma = s * (1 - r)``.
    """
    r = min(1.0, max(0.0, float(balance)))
    s = max(0.0, float(feature_strength))
    return s * r, s * (1.0 - r)


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class CellLabelICMParams:
    """Parameters for the unary-only geodesic cell segmentation."""

    balance: float = 1.0
    """Contour↔foreground split ``r`` in ``[0, 1]``: ``1`` = pure contour,
    ``0`` = pure foreground.  See :func:`balance_strength_to_weights`."""

    feature_strength: float = 4.0
    """Overall feature weight ``s >= 0`` relative to the fixed base cost of 1:
    how strongly contour/foreground bend the walk away from a plain geodesic
    distance Voronoi.  ``0`` = pure distance Voronoi."""

    n_workers: int = 1
    """Parallel worker processes for geodesic unary computation.
    1 = sequential.  Values > 1 use fork-based multiprocessing
    to compute frames in parallel."""


@dataclass
class CellICMState:
    """Cached energy-landscape data for the unary segmentation.

    Created by :func:`initialize_icm`.  All arrays are stored as their
    solver-ready dtypes (float32 / uint32 / bool).
    """

    fg_mask: np.ndarray = field(repr=False)
    """(T, Y, X) bool — foreground mask (includes nucleus pixels)."""

    nuc_tracks: np.ndarray = field(repr=False)
    """(T, Y, X) uint32 — nucleus track IDs (0 = no nucleus)."""

    label_ids: np.ndarray = field(repr=False)
    """(K,) uint32 — sorted global set of label (track) IDs."""

    unary_dense: np.ndarray | None = field(default=None, repr=False)
    """Deprecated. Previously held the dense ``(T, Y, X, K)`` unary cost
    volume; initial labels are now computed with a streaming argmin
    (:func:`_argmin_init_from_dict`) so this is left ``None`` to avoid the
    multi-gigabyte allocation."""

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.fg_mask.shape  # type: ignore[return-value]

    @property
    def n_labels(self) -> int:
        return len(self.label_ids)


# ── Cost field assembly (shared by solver + previews) ────────────────────────

def assemble_cost_field(
    contours_t: np.ndarray,
    fg_t: np.ndarray,
    alpha_unary: float,
    fg_scores_t: np.ndarray | None = None,
    gamma_unary: float = 0.0,
) -> np.ndarray:
    """Per-pixel geodesic cost over a single frame's foreground mask.

    ``cost = 1 + alpha_unary * contour + gamma_unary * (1 - fg_score)`` inside
    the mask, ``inf`` elsewhere.  This is the exact field the geodesic walk
    traverses; sharing it between :func:`_compute_frame_geodesic` and the cell
    widget's live preview guarantees the preview shows the same array the
    solver uses rather than a re-derivation.
    """
    Y, X = fg_t.shape
    cost_field = np.full((Y, X), np.inf, dtype=np.float32)
    c = 1.0 + alpha_unary * contours_t[fg_t]
    if gamma_unary != 0.0 and fg_scores_t is not None:
        c = c + gamma_unary * (1.0 - np.clip(fg_scores_t[fg_t], 0.0, 1.0))
    cost_field[fg_t] = c
    return cost_field


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
    # Build cost field — shared across all labels (and with the widget preview)
    cost_field = assemble_cost_field(
        contours_t, fg_t, alpha_unary, fg_scores_t, gamma_unary
    )

    # Locate every nucleus pixel in a single pass and group by label, rather
    # than rescanning the whole frame once per label (which was O(K * Y * X)).
    nz_y, nz_x = np.nonzero(nuc_t)
    if nz_y.size == 0:
        return {}
    nz_lab = nuc_t[nz_y, nz_x]
    order = np.argsort(nz_lab, kind="stable")  # stable → row-major within label
    nz_lab = nz_lab[order]
    nz_y = nz_y[order]
    nz_x = nz_x[order]
    uniq, starts_idx = np.unique(nz_lab, return_index=True)
    bounds = np.append(starts_idx, nz_lab.size)
    alive = [int(k) for k in uniq]

    # Single MCP object reused for all labels in this frame
    mcp = MCP_Geometric(cost_field, fully_connected=True)

    raw: dict[int, np.ndarray] = {}
    for i, k in enumerate(alive):
        s, e = int(starts_idx[i]), int(bounds[i + 1])
        ys = nz_y[s:e]
        xs = nz_x[s:e]
        # Seed the geodesic front from the nucleus centroid only, not the
        # whole body, so pixels are drawn toward the centre rather than
        # captured by the nearest point of the full nucleus footprint.
        cy = ys.mean()
        cx = xs.mean()
        nearest = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        cum, _ = mcp.find_costs([(int(ys[nearest]), int(xs[nearest]))])
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

    # Hard nucleus anchors within this frame: each label is forbidden (_INF)
    # on every *other* label's nucleus pixels.  O(K) full-frame writes instead
    # of the previous O(K^2) label-pair loop.
    nuc_pos = nuc_t > 0
    for k in alive:
        result[k][nuc_pos & (nuc_t != k)] = _INF

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


def _argmin_init_from_dict(
    unary: dict[tuple[int, int], np.ndarray],
    fg_mask: np.ndarray,
    label_ids: np.ndarray,
) -> np.ndarray:
    """Per-pixel nearest-seed assignment over the sparse unary dict.

    Equivalent to ``argmin`` over the label axis of the dense
    ``(T, Y, X, K)`` cost volume, but computed as a running minimum so the
    full volume is never materialised (it can be tens of GB for large
    stacks with many labels).  Missing ``(t, k)`` entries are treated as
    ``_INF``, matching the dense fill value.
    """
    T, Y, X = fg_mask.shape
    best_cost = np.full((T, Y, X), _INF, dtype=np.float32)
    best_ki = np.zeros((T, Y, X), dtype=np.intp)
    for ki, k in enumerate(label_ids):
        for t in range(T):
            u = unary.get((t, int(k)))
            if u is None:
                continue
            # Strict ``<`` keeps the lowest ki on ties, matching np.argmin.
            better = u < best_cost[t]
            np.copyto(best_cost[t], u, where=better)
            best_ki[t][better] = ki
    return np.where(fg_mask, label_ids[best_ki], 0).astype(np.uint32)


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


# ══════════════════════════════════════════════════════════════════════════════
# Public API
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
    """Compute geodesic unaries and build labels (per-pixel argmin).

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

    # ── Geodesic unaries (expensive — cache + parallel) ───────────────
    alpha_unary, gamma_unary = balance_strength_to_weights(
        params.balance, params.feature_strength
    )
    cache_key = _unary_cache_key((T, Y, X), alpha_unary, gamma_unary)
    unary_dict: dict[tuple[int, int], np.ndarray] | None = None

    if cache_dir is not None:
        _report(f"Checking unary cache: {cache_key}")
        unary_dict = _read_unary_cache(cache_dir, cache_key)
        if unary_dict is not None:
            _report(f"Cache hit: {len(unary_dict)} entries loaded.")

    if unary_dict is None:
        t0 = perf_counter()
        unary_dict = _compute_geodesic_unaries(
            nuc_tracks, fg_mask, contours, label_ids, alpha_unary,
            foreground_scores=foreground_scores,
            gamma_unary=gamma_unary,
            n_workers=params.n_workers,
            progress_cb=progress_cb,
        )
        elapsed = perf_counter() - t0
        _report(f"Geodesic unaries: {len(unary_dict)} entries in {elapsed:.1f}s")
        if cache_dir is not None:
            _report("Writing unary cache...")
            _write_unary_cache(cache_dir, cache_key, unary_dict)

    _apply_nucleus_anchors(unary_dict, nuc_tracks, label_ids)

    # ── Initial labels (streaming argmin; no dense (T, Y, X, K) volume) ──
    _report("Initialising labels from unary argmin...")
    init_labels = _argmin_init_from_dict(unary_dict, fg_mask, label_ids)
    del unary_dict

    nuc_mask = nuc_tracks > 0
    init_labels[nuc_mask] = nuc_tracks[nuc_mask].astype(np.uint32)

    state = CellICMState(
        fg_mask=fg_mask.astype(bool, copy=False),
        nuc_tracks=nuc_tracks.astype(np.uint32, copy=False),
        label_ids=label_ids,
    )

    _report("Initialisation complete.")
    return state, init_labels


def commit_labels(labels: np.ndarray, output_path: Path | str) -> None:
    """Write label array to TIFF.

    Labels are stored as ``uint16`` when they fit (compact, backward
    compatible) and promoted to ``uint32`` otherwise. Casting a track id
    above 65535 down to ``uint16`` would silently wrap and merge distinct
    cells, so the dtype is chosen from the actual maximum label.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_label = int(labels.max()) if labels.size else 0
    out_dtype = np.uint16 if max_label <= np.iinfo(np.uint16).max else np.uint32
    imwrite_grayscale(
        output_path,
        labels.astype(out_dtype, copy=False),
        compression="zlib",
    )
