"""Nucleus-seeded watershed segmentation hypothesis sweep.

Runs skimage.segmentation.watershed seeded from tracked nuclear labels, using
the cell-channel probability map and/or flow-field magnitude as the height map.
All (cellprob_threshold × compactness × smooth_sigma) parameter combinations
are swept and each produces a separate hypothesis label array.

Weight-map conventions
----------------------
- "prob"     : image = 1 - prob   (low cost in cell interior)
- "flow_mag" : image = 1 - mag    (flow magnitude ≈ 1 in interior, ≈ 0 at
                                   boundaries due to conflicting flows)
- "both"     : run both and include both as separate hypotheses
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Generator

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

from cellflow.cellpose.config import SeededWatershedConfig
from cellflow.cellpose.stages.contours import discover_dp_files, discover_prob_files


# ── Contour extraction ────────────────────────────────────────────────────────


def _contours_from_labels(labels: np.ndarray, smooth_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Extract foreground + contour maps from a label array."""
    fg = (labels > 0).astype(np.float32)

    ct = np.zeros_like(labels, dtype=np.float32)
    for axis in range(labels.ndim):
        sl_a = [slice(None)] * labels.ndim
        sl_b = [slice(None)] * labels.ndim
        sl_a[axis] = slice(None, -1)
        sl_b[axis] = slice(1, None)
        diff = (labels[tuple(sl_a)] != labels[tuple(sl_b)]).astype(np.float32)
        ct[tuple(sl_a)] = np.maximum(ct[tuple(sl_a)], diff)
        ct[tuple(sl_b)] = np.maximum(ct[tuple(sl_b)], diff)

    if smooth_sigma > 0:
        ct = gaussian_filter(ct, sigma=smooth_sigma)
        ct_max = float(ct.max())
        if ct_max > 0:
            ct /= ct_max

    return fg, ct.astype(np.float32)


# ── Flow magnitude ────────────────────────────────────────────────────────────


def _flow_mag(dp: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Compute normalized flow magnitude; same spatial shape as prob."""
    z_first = dp.ndim == 4 and dp.shape[0] == prob.shape[0]

    if dp.ndim <= 2:
        mag = np.abs(dp).astype(np.float32)
    elif dp.ndim == 3:
        mag = np.abs(dp).astype(np.float32) if z_first else np.sqrt(np.sum(dp.astype(np.float32) ** 2, axis=0))
    elif dp.ndim == 4:
        axis = 1 if z_first else 0
        mag = np.sqrt(np.sum(dp.astype(np.float32) ** 2, axis=axis))
    else:
        raise ValueError(f"Unexpected dp shape: {dp.shape}")

    mag_max = float(mag.max())
    if mag_max > 0:
        mag /= mag_max
    return mag.astype(np.float32)


# ── Sweep builder ─────────────────────────────────────────────────────────────


def _build_sweeps(cfg: SeededWatershedConfig) -> list[tuple[float, float, float]]:
    """Return (cellprob_threshold, compactness, smooth_sigma) combos."""
    def _vals(v, lo, hi, step):
        if step > 0 and lo != hi:
            return list(np.arange(lo, hi + step / 2, step))
        return [v]

    cp_vals = _vals(cfg.cellprob_threshold, cfg.cellprob_min, cfg.cellprob_max, cfg.cellprob_step)
    comp_vals = _vals(cfg.compactness, cfg.compactness_min, cfg.compactness_max, cfg.compactness_step)
    sigma_vals = _vals(cfg.smooth_sigma, cfg.smooth_min, cfg.smooth_max, cfg.smooth_step)

    return [(cp, comp, sigma) for cp in cp_vals for comp in comp_vals for sigma in sigma_vals]


# ── Core computation ──────────────────────────────────────────────────────────


def compute_hypotheses(
    prob: np.ndarray,
    dp: np.ndarray | None,
    nucleus_labels: np.ndarray,
    cfg: SeededWatershedConfig,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Run seeded watershed sweep for one timepoint.

    Parameters
    ----------
    prob : (Z, Y, X) or (Y, X) float32
    dp : (Z, C, Y, X) or (C, Y, X) or None
    nucleus_labels : (Z, Y, X) or (Y, X) int32 — non-zero pixels are seeds
    cfg : SeededWatershedConfig

    Returns
    -------
    foreground : float32 — mean across all hypotheses
    contours : float32 — mean across all hypotheses
    all_labels : list of uint32 arrays, one per (combo × weight-source) run
    """
    from skimage.segmentation import watershed

    # 2D seeds + 3D prob: run per Z-slice
    is_per_z = prob.ndim == 3 and nucleus_labels.ndim == 2

    flow_mag = _flow_mag(dp, prob) if (dp is not None and cfg.weight_source in ("flow_mag", "both")) else None

    combos = _build_sweeps(cfg)
    fg_list: list[np.ndarray] = []
    ct_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []

    _smoothed: dict[float, tuple] = {}

    for cp_thresh, compactness, smooth_sigma in combos:
        if smooth_sigma not in _smoothed:
            prob_s = gaussian_filter(prob, sigma=smooth_sigma) if smooth_sigma > 0 else prob
            fm_s = (gaussian_filter(flow_mag, sigma=smooth_sigma) if smooth_sigma > 0 else flow_mag) if flow_mag is not None else None
            _smoothed[smooth_sigma] = (prob_s, fm_s)
        prob_s, fm_s = _smoothed[smooth_sigma]

        mask = prob > cp_thresh

        images: list[np.ndarray] = []
        if cfg.weight_source in ("prob", "both"):
            images.append((1.0 - prob_s).astype(np.float32))
        if cfg.weight_source in ("flow_mag", "both") and fm_s is not None:
            images.append((1.0 - fm_s).astype(np.float32))
        if not images:
            images.append((1.0 - prob_s).astype(np.float32))

        for image in images:
            if is_per_z:
                slices = [
                    watershed(image[z], markers=nucleus_labels.astype(np.int32), mask=mask[z], compactness=compactness, watershed_line=False).astype(np.uint32)
                    for z in range(image.shape[0])
                ]
                labels = np.stack(slices, axis=0)
            else:
                labels = watershed(image, markers=nucleus_labels.astype(np.int32), mask=mask, compactness=compactness, watershed_line=False).astype(np.uint32)

            labels_list.append(labels)
            fg, ct = _contours_from_labels(labels, cfg.smooth_contour_sigma)
            fg_list.append(fg)
            ct_list.append(ct)

    if not fg_list:
        zero = np.zeros_like(prob, dtype=np.float32)
        return zero, zero, []

    return (
        np.mean(fg_list, axis=0).astype(np.float32),
        np.mean(ct_list, axis=0).astype(np.float32),
        labels_list,
    )


# ── Parallel worker ───────────────────────────────────────────────────────────


def _worker(args: tuple) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    dp_t, prob_t, nuc_t, cfg = args
    if nuc_t.shape[-2:] != prob_t.shape[-2:]:
        zero = np.zeros(prob_t.shape, dtype=np.float32)
        return zero, zero, []
    return compute_hypotheses(prob_t, dp_t, nuc_t, cfg)


# ── Batch run ─────────────────────────────────────────────────────────────────


def run(
    input_dir: str | Path,
    nucleus_labels_path: str | Path,
    output_dir: str | Path,
    cfg: SeededWatershedConfig,
    overwrite: bool = False,
) -> Generator[tuple[int, int, str], None, None]:
    """Process all timepoints and write foreground/contour/hypothesis files.

    Yields (done, total, status_label).
    """
    inp = Path(input_dir)
    out = Path(output_dir)
    fg_path = out / "foreground.tif"
    ct_path = out / "contours.tif"

    if not overwrite and fg_path.exists() and ct_path.exists():
        yield (0, 1, "foreground.tif and contours.tif already exist, skipping")
        return

    nuc_path = Path(nucleus_labels_path)
    if not nuc_path.exists():
        yield (0, 0, f"Nucleus labels not found: {nuc_path}")
        return

    prob_files = discover_prob_files(inp)
    dp_files = discover_dp_files(inp)

    if not prob_files:
        yield (0, 0, "No *_prob.tif files found")
        return

    if dp_files and len(dp_files) != len(prob_files):
        yield (0, 0, f"Mismatch: {len(dp_files)} dp vs {len(prob_files)} prob files")
        return

    yield (0, len(prob_files), "Loading nucleus labels…")
    nucleus_labels = tifffile.imread(str(nuc_path)).astype(np.int32)

    n_combos = len(_build_sweeps(cfg)) * (2 if cfg.weight_source == "both" else 1)
    n_files = len(prob_files)

    fg_frames: list[np.ndarray] = []
    ct_frames: list[np.ndarray] = []
    combo_labels_all: list[list[np.ndarray]] = [[] for _ in range(n_combos)] if cfg.save_all_hypotheses else []
    t_global = 0

    if cfg.n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        for file_idx, prob_path in enumerate(prob_files):
            t_str = prob_path.name.split("_")[0]
            dp_path = dp_files[file_idx] if dp_files else None
            yield (file_idx, n_files, f"Loading {t_str}…")

            prob = tifffile.imread(str(prob_path)).astype(np.float32)
            dp = tifffile.imread(str(dp_path)).astype(np.float32) if dp_path else None

            pairs = [(dp[t], prob[t]) for t in range(dp.shape[0])] if (dp is not None and dp.ndim == 5) else \
                    [(None, prob[t]) for t in range(prob.shape[0])] if (dp is None and prob.ndim == 4) else \
                    [(dp, prob)]

            args = []
            for dp_t, prob_t in pairs:
                nuc_t = nucleus_labels[t_global] if t_global < nucleus_labels.shape[0] else np.zeros(prob_t.shape[-2:], dtype=np.int32)
                args.append((dp_t, prob_t, nuc_t, cfg))
                t_global += 1

            with ProcessPoolExecutor(max_workers=cfg.n_workers) as executor:
                for fg, ct, all_labels in executor.map(_worker, args):
                    fg_frames.append(fg)
                    ct_frames.append(ct)
                    if cfg.save_all_hypotheses:
                        for i, lbl in enumerate(all_labels):
                            combo_labels_all[i].append(lbl)

            yield (file_idx + 1, n_files, f"{t_str} done ({len(args)} timepoints, {n_combos} combo(s), {cfg.n_workers} workers)")

    else:
        def _n_timepoints(path: Path) -> int:
            with tifffile.TiffFile(str(path)) as tf:
                s = tf.series[0].shape
            return s[0] if len(s) >= 4 else 1

        total = sum(_n_timepoints(p) for p in prob_files)

        for file_idx, prob_path in enumerate(prob_files):
            t_str = prob_path.name.split("_")[0]
            dp_path = dp_files[file_idx] if dp_files else None

            try:
                prob = tifffile.imread(str(prob_path)).astype(np.float32)
                dp = tifffile.imread(str(dp_path)).astype(np.float32) if dp_path else None

                pairs = [(dp[t], prob[t]) for t in range(dp.shape[0])] if (dp is not None and dp.ndim == 5) else \
                        [(None, prob[t]) for t in range(prob.shape[0])] if (dp is None and prob.ndim == 4) else \
                        [(dp, prob)]

                for dp_t, prob_t in pairs:
                    nuc_t = nucleus_labels[t_global] if t_global < nucleus_labels.shape[0] else np.zeros(prob_t.shape[-2:], dtype=np.int32)
                    fg, ct, all_labels = _worker((dp_t, prob_t, nuc_t, cfg))
                    fg_frames.append(fg)
                    ct_frames.append(ct)
                    if cfg.save_all_hypotheses:
                        for i, lbl in enumerate(all_labels):
                            combo_labels_all[i].append(lbl)
                    t_global += 1
                    yield (t_global, total, f"t={t_global}/{total} — {t_str} ({n_combos} combo(s))")

            except Exception as e:
                print(f"  {t_str}: error: {e}", flush=True)
                zero = np.zeros_like(prob if 'prob' in dir() else np.zeros(1), dtype=np.float32)
                fg_frames.append(zero)
                ct_frames.append(zero)
                if cfg.save_all_hypotheses:
                    for lst in combo_labels_all:
                        lst.append(np.zeros_like(zero, dtype=np.uint32))
                t_global += 1
                yield (t_global, total, f"t={t_global} — {t_str} ERROR: {e}")

    out.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(fg_path), np.stack(fg_frames, axis=0), compression="zlib")
    tifffile.imwrite(str(ct_path), np.stack(ct_frames, axis=0), compression="zlib")

    if cfg.save_all_hypotheses and combo_labels_all:
        hyp_dir = out / "hypotheses"
        hyp_dir.mkdir(parents=True, exist_ok=True)
        for i, frames in enumerate(combo_labels_all):
            tifffile.imwrite(
                str(hyp_dir / f"hypothesis_{i:03d}.tif"),
                np.stack(frames, axis=0).astype(np.uint32),
                compression="zlib",
            )

    total_written = len(fg_frames)
    yield (total_written, total_written, "Done")


# ── CLI ───────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nucleus-seeded watershed hypothesis sweep")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--nucleus-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg_dict: dict = {}
    if args.config:
        cfg_dict = json.loads(Path(args.config).read_text())
    cfg = SeededWatershedConfig(**cfg_dict)

    for done, total, label in run(args.input_dir, args.nucleus_labels, args.output_dir, cfg, overwrite=args.overwrite):
        print(f"[{done}/{total}] {label}", flush=True)

    sys.exit(0)
