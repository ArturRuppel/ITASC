"""Unary-only cell segmentation from cached divergence maps.

Chains the simplified cell pipeline validated in
``scripts/experiment_divergence_icm.py`` into a single, Qt-free helper so the
cell widget can drive both its live single-frame preview and its full-stack run
through the same code path:

    1. Map cleanup     — local-mean residual + threshold on foreground and
                         contours (the nucleus/atom ``residual`` scheme applied
                         symmetrically to both maps).
    2. Temporal smooth — bidirectional signal-adaptive EMA on the cleaned
                         contours (full-stack only; needs the whole movie).
    3. Foreground mask — ``(foreground_clean > fg_threshold) | (nucleus > 0)``.
    4. Segmentation    — unary-only geodesic Voronoi: ``initialize_icm`` assigns
                         each foreground pixel to its nearest nucleus seed
                         through the contour-aware cost field (per-pixel argmin
                         of the unary).

The divergence maps themselves (``cell_contours.tif`` + ``cell_foreground.tif``)
are produced upstream by ``DivergenceMapsWidget``; this helper only consumes
them.

The returned :class:`CellDivergenceResult` carries every intermediate plus the
weighted cost field, so the widget can drop each into a preview layer without
recomputing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from cellflow.segmentation.cell_label_icm import (
    CellLabelICMParams,
    assemble_cost_field,
    balance_strength_to_weights,
    initialize_icm,
)
from cellflow.segmentation.contour_filtering import contour_memory_filter
from cellflow.tracking_ultrack.atoms import residual

__all__ = [
    "CellDivergenceParams",
    "CellDivergenceResult",
    "clean_and_smooth_contours",
    "segment_cells_divergence",
]


@dataclass
class CellDivergenceParams:
    """Parameters for the unary-only divergence cell pipeline.

    Defaults match the values that held across pos00/pos01 in the prototype.
    """

    # ── Map cleanup (same trio per map as the nucleus/atom widget) ──────────
    fg_window: int = 51
    """Local-mean window for the foreground residual (px, forced odd)."""
    fg_strength: float = 0.0
    """Foreground residual strength: 0 = raw sigmoid, 1 = full subtraction."""
    fg_threshold: float = 0.1
    """Cleaned-foreground cutoff producing the fill mask (sigmoid scale)."""
    contour_window: int = 51
    """Local-mean window for the contour residual (px, forced odd)."""
    contour_strength: float = 1.0
    """Contour residual strength: 0 = raw, 1 = full local-mean subtraction."""
    contour_threshold: float = 0.0
    """Noise floor on the normalized contour [0, 1]; below → 0."""
    contour_norm_pct: float = 99.0
    """Percentile of the positive contour signal mapped to 1.0 in [0, 1]."""

    # ── Temporal smoothing ──────────────────────────────────────────────────
    memory_tau: float = 0.0
    """EMA crossover (~the contour value you call "weak"). 0 = off."""
    memory_floor: float = 0.01
    """Minimum per-frame alpha; ghost half-life (~69 frames @ 0.01)."""

    # ── Segmentation ────────────────────────────────────────────────────────
    balance: float = 0.98
    """Contour↔foreground split ``r`` in ``[0, 1]`` (``1`` = pure contour).
    See :func:`balance_strength_to_weights`."""
    feature_strength: float = 100.0
    """Overall feature weight ``s >= 0`` relative to the base cost of 1."""
    n_workers: int = 4
    """Parallel workers for geodesic computation (compute only)."""


@dataclass
class CellDivergenceResult:
    """All pipeline intermediates plus the final labels.

    Arrays are ``(T, Y, X)`` for a full-stack run and ``(Y, X)`` for a
    single-frame (``frame`` given) run.
    """

    foreground_raw: np.ndarray
    """Raw input foreground map (sigmoid)."""
    foreground_clean: np.ndarray
    """Foreground after residual cleanup (sigmoid scale)."""
    contours_raw: np.ndarray
    """Raw input contour map (positive divergence)."""
    contours_clean: np.ndarray
    """Contours after residual + normalize + floor (and temporal smoothing
    for a full-stack run)."""
    foreground_mask: np.ndarray
    """Fill territory: ``(foreground_clean > fg_threshold) | (nucleus > 0)``."""
    cost_field: np.ndarray
    """Weighted geodesic cost over the mask; ``inf`` outside."""
    labels: np.ndarray | None
    """Cell labels — the unary argmin (tracked nucleus IDs).  ``None`` when the
    geodesic label assignment was skipped (``with_labels=False``)."""


def _robust_normalize_01(contours: np.ndarray, pct: float) -> np.ndarray:
    """Scale positive-divergence contours into [0, 1] by a high percentile.

    Raw positive divergence has an arbitrary positive scale, so dividing by a
    high percentile of the nonzero signal keeps ``alpha`` interpretable and
    comparable across frames/datasets.  Mirrors the prototype.
    """
    c = np.clip(np.asarray(contours, dtype=np.float32), 0.0, None)
    nz = c[c > 0]
    if nz.size == 0:
        return c
    hi = float(np.percentile(nz, pct))
    if hi <= 0.0:
        return c
    return np.clip(c / hi, 0.0, 1.0).astype(np.float32)


def _clean_foreground(fg: np.ndarray, params: CellDivergenceParams) -> np.ndarray:
    """Per-frame local-mean residual; stays on the native sigmoid scale.

    ``fg_strength=0`` makes ``residual`` a no-op (returns the raw, non-negative
    map), so the baseline reproduces the raw-sigmoid foreground.
    """
    return np.stack([
        residual(fg[t], params.fg_window, params.fg_strength)
        for t in range(fg.shape[0])
    ]).astype(np.float32)


def _clean_contours(contours: np.ndarray, params: CellDivergenceParams) -> np.ndarray:
    """Per-frame residual → [0, 1] normalize → noise floor."""
    cleaned = np.stack([
        residual(contours[t], params.contour_window, params.contour_strength)
        for t in range(contours.shape[0])
    ]).astype(np.float32)
    cleaned = _robust_normalize_01(cleaned, params.contour_norm_pct)
    if params.contour_threshold > 0.0:
        cleaned = np.where(cleaned < params.contour_threshold, 0.0, cleaned)
    return cleaned.astype(np.float32)


def clean_and_smooth_contours(
    contours: np.ndarray, params: CellDivergenceParams
) -> np.ndarray:
    """Full-stack contour cleanup + temporal smoothing — pipeline stages 1+2.

    Returns the ``(T, Y, X)`` cleaned (residual → global-percentile normalize →
    floor) and, when ``memory_tau > 0`` and there is more than one frame,
    temporally smoothed contour stack — exactly the ``contours_clean`` the
    full run feeds the segmenter.

    The widget's live preview computes this once over the whole movie, caches it,
    and slices the current frame back into :func:`segment_cells_divergence` via
    ``contours_clean_override`` so the previewed cost field / labels for a frame
    match the full run (which the per-frame path cannot, since both the global
    percentile and the bidirectional EMA need every frame).
    """
    contours = _to_tyx(contours, np.float32)
    cleaned = _clean_contours(contours, params)
    if params.memory_tau > 0.0 and cleaned.shape[0] > 1:
        cleaned = contour_memory_filter(
            cleaned, tau=params.memory_tau, floor=params.memory_floor,
        )
    return cleaned.astype(np.float32)


def _to_tyx(arr: np.ndarray, dtype) -> np.ndarray:
    a = np.asarray(arr, dtype=dtype)
    if a.ndim == 4 and a.shape[1] == 1:
        a = a[:, 0]
    if a.ndim == 2:
        a = a[np.newaxis]
    return a


def segment_cells_divergence(
    contours: np.ndarray,
    foreground: np.ndarray,
    nuc: np.ndarray,
    params: CellDivergenceParams,
    *,
    frame: int | None = None,
    with_labels: bool = True,
    contours_clean_override: np.ndarray | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> CellDivergenceResult:
    """Run the unary-only divergence pipeline and return all intermediates.

    Parameters
    ----------
    contours, foreground : (T, Y, X) float
        Cached divergence maps (raw positive divergence and the sigmoid
        foreground, respectively).
    nuc : (T, Y, X) integer
        Tracked nucleus seeds.
    params : CellDivergenceParams
    frame : int, optional
        When given, only that frame is processed and **temporal smoothing is
        skipped** (it needs the whole stack); the result arrays are 2-D.  When
        ``None``, the whole stack is processed including temporal smoothing.
    with_labels : bool, default True
        When ``False``, the geodesic Voronoi label assignment (the single
        slowest step) is skipped: every cleanup intermediate plus the weighted
        cost field is still returned, but ``result.labels`` is ``None``.  The
        live preview uses this to stay responsive — the cost field already
        explains every boundary the labels would land on.
    contours_clean_override : (Y, X) float, optional
        Single-frame-only.  When given (with ``frame`` set), this pre-cleaned —
        and, when temporal smoothing is on, pre-smoothed — contour frame is used
        as ``contours_clean`` instead of re-running the per-frame cleanup.  The
        widget passes a frame sliced from :func:`clean_and_smooth_contours` so
        the single-frame cost field / labels match the full run exactly (the
        per-frame path cannot, as it lacks the whole-movie percentile and EMA).
        Ignored when ``frame`` is ``None``.
    progress_cb : callable, optional
        Receives short status strings.

    Returns
    -------
    CellDivergenceResult
    """
    _report = progress_cb or (lambda _msg: None)

    contours = _to_tyx(contours, np.float32)
    foreground = _to_tyx(foreground, np.float32)
    nuc = _to_tyx(nuc, np.uint32)

    T = min(len(contours), len(foreground), len(nuc))
    contours, foreground, nuc = contours[:T], foreground[:T], nuc[:T]

    single = frame is not None
    if single:
        t = max(0, min(int(frame), T - 1))
        contours = contours[t:t + 1]
        foreground = foreground[t:t + 1]
        nuc = nuc[t:t + 1]

    contours_raw = contours.copy()
    foreground_raw = np.clip(foreground, 0.0, 1.0).astype(np.float32)

    # ── 1. Map cleanup ──────────────────────────────────────────────────────
    _report("Cleaning maps…")
    foreground_clean = _clean_foreground(foreground, params)

    if single and contours_clean_override is not None:
        # Caller supplied the already cleaned (+ smoothed) frame — use it
        # verbatim so the single-frame result matches the full run for this
        # frame.  Stage 2 is folded into the override and skipped here.
        override = np.asarray(contours_clean_override, dtype=np.float32)
        if override.shape != contours.shape[1:]:
            raise ValueError(
                "contours_clean_override shape "
                f"{override.shape} does not match frame shape {contours.shape[1:]}"
            )
        contours_clean = override[np.newaxis]
    else:
        contours_clean = _clean_contours(contours, params)
        # ── 2. Temporal contour smoothing (full-stack only) ─────────────────
        if not single and params.memory_tau > 0.0 and contours_clean.shape[0] > 1:
            _report(f"Temporal contour smoothing (τ={params.memory_tau})…")
            contours_clean = contour_memory_filter(
                contours_clean, tau=params.memory_tau, floor=params.memory_floor,
            )

    # ── 3. Foreground mask ──────────────────────────────────────────────────
    foreground_mask = (foreground_clean > params.fg_threshold) | (nuc > 0)

    # ── Weighted cost field (same construction the solver traverses) ────────
    # Cheap (`1 + α·contour + γ·(1 − fg)`); built first so it is available even
    # when the geodesic label assignment below is skipped.
    alpha, gamma = balance_strength_to_weights(
        params.balance, params.feature_strength
    )
    cost_field = np.stack([
        assemble_cost_field(
            contours_clean[i], foreground_mask[i],
            alpha, foreground_clean[i], gamma,
        )
        for i in range(contours_clean.shape[0])
    ]).astype(np.float32)

    # ── 4. Unary-only segmentation (the slow geodesic walk) ─────────────────
    labels: np.ndarray | None = None
    if with_labels:
        _report("Segmenting (unary geodesic Voronoi)…")
        icm_params = CellLabelICMParams(
            balance=params.balance,
            feature_strength=params.feature_strength,
            n_workers=1 if single else max(1, params.n_workers),
        )
        _state, labels = initialize_icm(
            nuc, foreground_mask, contours_clean, icm_params,
            foreground_scores=foreground_clean,
            progress_cb=lambda m: _report(str(m)),
        )
        labels = labels.astype(np.uint32, copy=False)

    if single:
        return CellDivergenceResult(
            foreground_raw=foreground_raw[0],
            foreground_clean=foreground_clean[0],
            contours_raw=contours_raw[0],
            contours_clean=contours_clean[0],
            foreground_mask=foreground_mask[0],
            cost_field=cost_field[0],
            labels=None if labels is None else labels[0],
        )
    return CellDivergenceResult(
        foreground_raw=foreground_raw,
        foreground_clean=foreground_clean,
        contours_raw=contours_raw,
        contours_clean=contours_clean,
        foreground_mask=foreground_mask,
        cost_field=cost_field,
        labels=labels,
    )
