"""Radial contour-registration refinement of tracked nucleus labels.

Each existing tracked label is treated as an identity-preserving shape. Its
boundary is sampled by ``n_rays`` radial rays from the centroid, and each ray
searches within a local displacement window for a better contour location using
``contours.tif`` and a foreground-score image. Hard guarantees: original label
IDs persist, one 8-connected component per label per frame, no holes, no
overwriting of neighboring labels, mutation constrained to a dilation band
around the original object.

This module is pure (no Qt, no napari). The napari widget calls
:func:`refine_stack` and :func:`write_refinement_outputs`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Callable
from collections.abc import Iterable

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import polygon
from skimage.measure import perimeter
from skimage.morphology import remove_small_holes

from cellflow.core.tiff import imwrite_grayscale


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RadialRefineConfig:
    """Tunable knobs for radial contour-registration refinement."""

    n_rays: int = 64
    max_outward: int = 10
    max_inward: int = 10
    wc: float = 5.0
    wi: float = 2.0
    we: float = 1.0
    wd: float = 1.0
    smooth: int = 2
    orig_pull: float = 0.15

    def with_(self, **kwargs) -> RadialRefineConfig:
        return replace(self, **kwargs)


PRESETS: dict[str, RadialRefineConfig] = {
    "balanced": RadialRefineConfig(
        wc=5.0, wi=2.0, we=1.0, wd=1.0, smooth=2, orig_pull=0.15
    ),
    "conservative": RadialRefineConfig(
        wc=4.0, wi=2.0, we=2.0, wd=2.0, smooth=3, orig_pull=0.30
    ),
    "contour": RadialRefineConfig(
        wc=8.0, wi=1.0, we=1.0, wd=1.0, smooth=2, orig_pull=0.10
    ),
}


def preset_name(cfg: RadialRefineConfig) -> str | None:
    for name, preset in PRESETS.items():
        if cfg == preset:
            return name
    return None


def param_hash(cfg: RadialRefineConfig, length: int = 8) -> str:
    """Short stable hash of the config for filenames."""
    blob = json.dumps(asdict(cfg), sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:length]


def config_label(cfg: RadialRefineConfig) -> str:
    """Filename-friendly label combining preset name (or 'custom') and hash."""
    return f"{preset_name(cfg) or 'custom'}_{param_hash(cfg)}"


# ── Core per-frame primitives ─────────────────────────────────────────────────


def _angle_basis(n_rays: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    return (
        angles,
        np.sin(angles).astype(np.float32),
        np.cos(angles).astype(np.float32),
    )


def _radial_profile(
    mask: np.ndarray, cy: float, cx: float, n_rays: int
) -> np.ndarray:
    yy, xx = np.nonzero(mask)
    dy = yy.astype(np.float32) - cy
    dx = xx.astype(np.float32) - cx
    rr = np.sqrt(dy * dy + dx * dx)
    aa = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
    bins = np.floor(aa / (2 * np.pi) * n_rays).astype(np.int32) % n_rays
    radii = np.zeros(n_rays, dtype=np.float32)
    for i in range(n_rays):
        vals = rr[bins == i]
        if vals.size:
            radii[i] = np.percentile(vals, 98)
    known = radii > 0
    if not known.all():
        idx = np.arange(n_rays)
        if known.any():
            ki = idx[known]
            kv = radii[known]
            radii = np.interp(
                idx,
                np.r_[ki - n_rays, ki, ki + n_rays],
                np.r_[kv, kv, kv],
            ).astype(np.float32)
        else:
            radii[:] = 2
    return np.maximum(radii, 2)


def _mask_from_radii(
    cy: float,
    cx: float,
    radii: np.ndarray,
    sin_a: np.ndarray,
    cos_a: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    yy = cy + radii * sin_a
    xx = cx + radii * cos_a
    rr, cc = polygon(yy, xx, shape=shape)
    m = np.zeros(shape, bool)
    m[rr, cc] = True
    return m


def _clean_seed_component(mask: np.ndarray, sy: int, sx: int) -> np.ndarray:
    mask = ndi.binary_fill_holes(mask)
    mask = remove_small_holes(mask, area_threshold=8192)
    mask[sy, sx] = True
    cc, _ = ndi.label(mask, structure=np.ones((3, 3), np.uint8))
    sid = cc[sy, sx]
    if sid == 0:
        return mask
    return ndi.binary_fill_holes(cc == sid)


def _optimize_radii(
    mask: np.ndarray,
    cy: float,
    cx: float,
    fg_t: np.ndarray,
    cont_t: np.ndarray,
    cfg: RadialRefineConfig,
    sin_a: np.ndarray,
    cos_a: np.ndarray,
) -> np.ndarray:
    H, W = fg_t.shape
    n_rays = cfg.n_rays
    orig = _radial_profile(mask, cy, cx, n_rays)
    offsets = np.arange(
        -int(cfg.max_inward), int(cfg.max_outward) + 1, dtype=np.float32
    )
    move_norm = max(int(cfg.max_inward), int(cfg.max_outward), 1)

    cand = np.maximum(2.0, orig[:, None] + offsets[None, :])
    ys = np.rint(cy + cand * sin_a[:, None]).astype(np.int32)
    xs = np.rint(cx + cand * cos_a[:, None]).astype(np.int32)
    valid = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    c = np.zeros_like(cand, dtype=np.float32)
    fedge = np.zeros_like(cand, dtype=np.float32)
    c[valid] = cont_t[ys[valid], xs[valid]]
    fedge[valid] = fg_t[ys[valid], xs[valid]]
    yin = np.rint(cy + np.maximum(1.0, cand - 2.0) * sin_a[:, None]).astype(np.int32)
    xin = np.rint(cx + np.maximum(1.0, cand - 2.0) * cos_a[:, None]).astype(np.int32)
    vin = (yin >= 0) & (yin < H) & (xin >= 0) & (xin < W)
    fin = np.zeros_like(cand, dtype=np.float32)
    fin[vin] = fg_t[yin[vin], xin[vin]]
    score = (
        cfg.wc * c
        + cfg.wi * fin
        + cfg.we * fedge
        - cfg.wd * (np.abs(offsets)[None, :] / move_norm)
    )
    score[~valid] = -1e9
    idx = np.argmax(score, axis=1)
    r = cand[np.arange(n_rays), idx]
    for _ in range(int(cfg.smooth)):
        r = (1 - cfg.orig_pull) * (
            0.5 * r + 0.25 * np.roll(r, 1) + 0.25 * np.roll(r, -1)
        ) + cfg.orig_pull * orig
    return np.clip(
        r,
        np.maximum(2.0, orig - float(cfg.max_inward)),
        orig + float(cfg.max_outward),
    )


# ── Frame-level refinement ────────────────────────────────────────────────────


@dataclass
class PerObjectRow:
    t: int
    label: int
    inside_score: float
    edge_contour_mean: float
    area: int
    perimeter: float
    circularity: float


def _compute_seeds(labels_t: np.ndarray) -> dict[int, tuple[int, int]]:
    """Return {label: (sy, sx)} using the pixel nearest the centroid."""
    seeds: dict[int, tuple[int, int]] = {}
    for lab in np.unique(labels_t):
        if lab == 0:
            continue
        yy, xx = np.nonzero(labels_t == lab)
        cy, cx = float(yy.mean()), float(xx.mean())
        j = int(np.argmin((yy - cy) ** 2 + (xx - cx) ** 2))
        seeds[int(lab)] = (int(yy[j]), int(xx[j]))
    return seeds


def refine_frame(
    labels_t: np.ndarray,
    contours_t: np.ndarray,
    fg_t: np.ndarray,
    cfg: RadialRefineConfig,
    *,
    frozen_labels: Iterable[int] = (),
) -> tuple[np.ndarray, list[PerObjectRow]]:
    """Refine one frame. Returns (out_labels_t, per_object_rows).

    Labels listed in ``frozen_labels`` are copied through unchanged.
    """
    H, W = labels_t.shape
    _, sin_a, cos_a = _angle_basis(int(cfg.n_rays))
    out = np.zeros_like(labels_t, dtype=np.uint32)
    rows: list[PerObjectRow] = []
    seeds = _compute_seeds(labels_t)
    frozen = {int(x) for x in frozen_labels}
    band_iters = max(int(cfg.max_outward), 1)

    # First pass: stamp frozen labels so the proposal pass can't steal pixels.
    for lab in frozen:
        if lab not in seeds:
            continue
        m = labels_t == lab
        if not m.any():
            continue
        out[m] = np.uint32(lab)
        area = int(m.sum())
        peri = float(perimeter(m, neighborhood=8)) if area else 0.0
        circ = float(4 * np.pi * area / (peri * peri)) if peri > 0 else 0.0
        inside = float(fg_t[m].mean()) if area else 0.0
        edge = m ^ ndi.binary_erosion(m, structure=np.ones((3, 3), bool))
        edge_cont = float(contours_t[edge].mean()) if edge.any() else 0.0
        rows.append(
            PerObjectRow(
                t=-1,
                label=int(lab),
                inside_score=inside,
                edge_contour_mean=edge_cont,
                area=area,
                perimeter=peri,
                circularity=circ,
            )
        )

    proposals: list[tuple[float, int, np.ndarray, float, float]] = []
    for lab, (sy, sx) in seeds.items():
        if lab in frozen:
            continue
        mask = labels_t == lab
        if not mask.any():
            continue
        yy, xx = np.nonzero(mask)
        cy, cx = float(yy.mean()), float(xx.mean())
        radii = _optimize_radii(
            mask, cy, cx, fg_t, contours_t, cfg, sin_a, cos_a
        )
        pmask = _mask_from_radii(cy, cx, radii, sin_a, cos_a, (H, W))
        band = ndi.binary_dilation(mask, iterations=band_iters)
        pmask &= band
        pmask = _clean_seed_component(pmask, sy, sx)
        pmask &= band
        pmask = _clean_seed_component(pmask, sy, sx)
        inside = float(fg_t[pmask].mean()) if pmask.any() else 0.0
        edge = pmask ^ ndi.binary_erosion(pmask, structure=np.ones((3, 3), bool))
        edge_cont = float(contours_t[edge].mean()) if edge.any() else 0.0
        proposals.append((inside + edge_cont, int(lab), pmask, inside, edge_cont))

    proposals.sort(reverse=True, key=lambda x: x[0])
    for _, lab, pmask, inside, edge_cont in proposals:
        sy, sx = seeds[lab]
        pmask = pmask & (out == 0)
        pmask[sy, sx] = True
        pmask = _clean_seed_component(pmask, sy, sx)
        pmask = pmask & (out == 0)
        pmask[sy, sx] = True
        out[pmask] = np.uint32(lab)
        area = int(pmask.sum())
        peri = float(perimeter(pmask, neighborhood=8)) if area else 0.0
        circ = float(4 * np.pi * area / (peri * peri)) if peri > 0 else 0.0
        rows.append(
            PerObjectRow(
                t=-1,
                label=int(lab),
                inside_score=inside,
                edge_contour_mean=edge_cont,
                area=area,
                perimeter=peri,
                circularity=circ,
            )
        )
    return out, rows


# ── Stack-level refinement ────────────────────────────────────────────────────


@dataclass
class RefineSummary:
    name: str
    path: str
    median_pixels: float
    median_ratio_vs_original: float
    hole_pixels: int
    fragmented_label_frames: int
    missing_seed_label_frames: int
    mean_inside_score: float
    mean_edge_contour: float
    mean_circularity: float


def _structural_qc(
    labels_t_original: np.ndarray,
    out_t: np.ndarray,
    seeds: dict[int, tuple[int, int]],
) -> tuple[int, int, int]:
    hole_pixels = 0
    fragmented = 0
    missing_seed = 0
    for lab, (sy, sx) in seeds.items():
        if out_t[sy, sx] != lab:
            missing_seed += 1
    for lab in np.unique(out_t):
        if lab == 0:
            continue
        m = out_t == lab
        filled = ndi.binary_fill_holes(m)
        hole_pixels += int(filled.sum() - m.sum())
        _, n = ndi.label(m, structure=np.ones((3, 3), np.uint8))
        if n != 1:
            fragmented += 1
    return hole_pixels, fragmented, missing_seed


def refine_stack(
    labels: np.ndarray,
    contours: np.ndarray,
    fg: np.ndarray,
    cfg: RadialRefineConfig,
    *,
    frozen_frames: Iterable[int] = (),
    frozen_labels: Iterable[int] = (),
    progress_cb: Callable[[int, int, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, list[PerObjectRow], RefineSummary]:
    """Refine every frame in ``labels``.

    Validated frames listed in ``frozen_frames`` are copied through unchanged.
    Labels listed in ``frozen_labels`` are copied through unchanged in every
    frame they appear (validated tracks).

    ``progress_cb(done, total, msg)`` is called per frame when supplied;
    ``should_stop`` is checked between frames to allow cancellation.
    """
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    labels = labels.astype(np.uint32, copy=False)
    contours = contours.astype(np.float32, copy=False)
    fg = fg.astype(np.float32, copy=False)
    if labels.shape != contours.shape or labels.shape != fg.shape:
        raise ValueError(
            f"shape mismatch: labels={labels.shape} "
            f"contours={contours.shape} fg={fg.shape}"
        )

    T = labels.shape[0]
    frozen_frames_set = {int(t) for t in frozen_frames}
    frozen_labels_set = {int(x) for x in frozen_labels}

    out = np.zeros_like(labels, dtype=np.uint32)
    all_rows: list[PerObjectRow] = []
    hole_pixels = 0
    fragmented = 0
    missing_seed = 0

    name = config_label(cfg)
    if progress_cb is not None:
        progress_cb(0, T, f"Refining ({name})...")

    for t in range(T):
        if should_stop is not None and should_stop():
            break
        if t in frozen_frames_set:
            out[t] = labels[t]
            seeds = _compute_seeds(labels[t])
            for lab, (sy, sx) in seeds.items():
                m = labels[t] == lab
                area = int(m.sum())
                peri = float(perimeter(m, neighborhood=8)) if area else 0.0
                circ = (
                    float(4 * np.pi * area / (peri * peri)) if peri > 0 else 0.0
                )
                inside = float(fg[t][m].mean()) if area else 0.0
                edge = m ^ ndi.binary_erosion(m, structure=np.ones((3, 3), bool))
                edge_cont = (
                    float(contours[t][edge].mean()) if edge.any() else 0.0
                )
                all_rows.append(
                    PerObjectRow(
                        t=t,
                        label=int(lab),
                        inside_score=inside,
                        edge_contour_mean=edge_cont,
                        area=area,
                        perimeter=peri,
                        circularity=circ,
                    )
                )
        else:
            out_t, rows_t = refine_frame(
                labels[t],
                contours[t],
                fg[t],
                cfg,
                frozen_labels=frozen_labels_set,
            )
            out[t] = out_t
            for r in rows_t:
                all_rows.append(
                    PerObjectRow(
                        t=t,
                        label=r.label,
                        inside_score=r.inside_score,
                        edge_contour_mean=r.edge_contour_mean,
                        area=r.area,
                        perimeter=r.perimeter,
                        circularity=r.circularity,
                    )
                )

        seeds = _compute_seeds(labels[t])
        h, f, m = _structural_qc(labels[t], out[t], seeds)
        hole_pixels += h
        fragmented += f
        missing_seed += m

        if progress_cb is not None:
            progress_cb(t + 1, T, f"Refined frame {t + 1}/{T}")

    orig_counts = np.array([np.count_nonzero(labels[t]) for t in range(T)])
    out_counts = np.array([np.count_nonzero(out[t]) for t in range(T)])
    orig_med = float(np.median(orig_counts)) if orig_counts.size else 0.0
    median_pixels = float(np.median(out_counts)) if out_counts.size else 0.0
    median_ratio = float(median_pixels / orig_med) if orig_med > 0 else 0.0

    mean_inside = (
        float(np.mean([r.inside_score for r in all_rows])) if all_rows else 0.0
    )
    mean_edge = (
        float(np.mean([r.edge_contour_mean for r in all_rows])) if all_rows else 0.0
    )
    mean_circ = (
        float(np.mean([r.circularity for r in all_rows])) if all_rows else 0.0
    )

    summary = RefineSummary(
        name=name,
        path="",
        median_pixels=median_pixels,
        median_ratio_vs_original=median_ratio,
        hole_pixels=int(hole_pixels),
        fragmented_label_frames=int(fragmented),
        missing_seed_label_frames=int(missing_seed),
        mean_inside_score=mean_inside,
        mean_edge_contour=mean_edge,
        mean_circularity=mean_circ,
    )
    return out, all_rows, summary


# ── I/O ───────────────────────────────────────────────────────────────────────


PER_OBJECT_HEADER = [
    "t",
    "label",
    "inside_score",
    "edge_contour_mean",
    "area",
    "perimeter",
    "circularity",
]

SUMMARY_HEADER = [
    "name",
    "path",
    "median_pixels",
    "median_ratio_vs_original",
    "hole_pixels",
    "fragmented_label_frames",
    "missing_seed_label_frames",
    "mean_inside_score",
    "mean_edge_contour",
    "mean_circularity",
]


def write_refinement_outputs(
    out_dir: Path,
    cfg: RadialRefineConfig,
    labels: np.ndarray,
    per_object: list[PerObjectRow],
    summary: RefineSummary,
) -> Path:
    """Write candidate TIFF + per_object CSV; append/update summary CSV.

    Returns the candidate TIFF path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = config_label(cfg)
    tif_path = out_dir / f"refined_labels_{name}.tif"
    imwrite_grayscale(tif_path, labels.astype(np.uint32))

    per_obj_path = out_dir / f"per_object_{name}.csv"
    with open(per_obj_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(PER_OBJECT_HEADER)
        for r in per_object:
            w.writerow([
                r.t,
                r.label,
                r.inside_score,
                r.edge_contour_mean,
                r.area,
                r.perimeter,
                r.circularity,
            ])

    summary = RefineSummary(**{**asdict(summary), "path": str(tif_path)})

    config_path = out_dir / f"config_{name}.json"
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)

    summary_path = out_dir / "summary.csv"
    existing: list[list[str]] = []
    if summary_path.exists():
        with open(summary_path, newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if rows and rows[0] == SUMMARY_HEADER:
            existing = [r for r in rows[1:] if r and r[0] != name]
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_HEADER)
        for r in existing:
            w.writerow(r)
        w.writerow([
            summary.name,
            summary.path,
            summary.median_pixels,
            summary.median_ratio_vs_original,
            summary.hole_pixels,
            summary.fragmented_label_frames,
            summary.missing_seed_label_frames,
            summary.mean_inside_score,
            summary.mean_edge_contour,
            summary.mean_circularity,
        ])

    return tif_path


def promote_refinement_to_tracked(
    refined_path: Path,
    tracked_path: Path,
    *,
    backup_suffix: str = ".prev.tif",
) -> Path | None:
    """Replace ``tracked_path`` with ``refined_path``.

    The current tracked file (if any) is moved aside to
    ``tracked_path.with_suffix(backup_suffix)`` as a one-level undo. Returns
    the backup path (or ``None`` if there was no prior tracked file).
    """
    refined_path = Path(refined_path)
    tracked_path = Path(tracked_path)
    if not refined_path.exists():
        raise FileNotFoundError(refined_path)
    tracked_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path: Path | None = None
    if tracked_path.exists():
        backup_path = tracked_path.with_name(tracked_path.stem + backup_suffix)
        shutil.move(str(tracked_path), str(backup_path))
    shutil.copy2(str(refined_path), str(tracked_path))
    return backup_path
