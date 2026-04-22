"""s02c — Cellpose-native contours from flow fields and probability maps.

Uses cellpose.dynamics.compute_masks to generate label maps from flow (dP) and
probability maps, then ultrack.utils.labels_to_contours to derive foreground and
contour maps.

This is an alternative to s02/s02b that leverages cellpose's own segmentation
logic rather than custom thresholding and watershed.

Usage
-----
    python -m ultrack_wrapper.stages.s02c_cellpose_contours \\
        --input-dir /path/to/cellpose_output \\
        --output-dir /path/to/output \\
        --config /tmp/cfg.json \\
        [--overwrite]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Generator

import numpy as np
import tifffile
from scipy import ndimage as ndi

from cellflow.cellpose.config import CellposeContoursConfig
from cellflow.core.paths import stage_dir
from cellflow.core.protocol import StageProgress, ValidationResult


# ── File discovery ──────────────────────────────────────────────────────────


def discover_dp_files(input_dir: str | Path) -> list[Path]:
    """Return the Cellpose cell flow stack in *input_dir*."""
    input_dir = Path(input_dir)
    path = input_dir / "cell_dp.tif"
    if path.exists():
        return [path]
    return []


def discover_prob_files(input_dir: str | Path) -> list[Path]:
    """Return the Cellpose cell probability stack in *input_dir*."""
    input_dir = Path(input_dir)
    path = input_dir / "cell_prob.tif"
    if path.exists():
        return [path]
    return []


# ── Core computation ────────────────────────────────────────────────────────


def compute_labels_single(
    dp_path: str | Path,
    prob_path: str | Path,
    cfg: CellposeContoursConfig,
) -> np.ndarray:
    """Compute label map using cellpose.dynamics.compute_masks.

    Parameters
    ----------
    dp_path : path to *_dp.tif (flow field)
    prob_path : path to *_prob.tif (cell probability map)
    cfg : CellposeContoursConfig

    Returns
    -------
    labels : np.ndarray, uint32
        Labeled segmentation; 0 = background, 1+ = cell IDs.
    """
    from cellpose.dynamics import compute_masks
    import torch

    dp = tifffile.imread(str(dp_path)).astype(np.float32)
    prob = tifffile.imread(str(prob_path)).astype(np.float32)

    try:
        device = torch.device(cfg.device)
        masks = compute_masks(
            dp,
            prob,
            cellprob_threshold=cfg.cellprob_threshold,
            do_3D=cfg.do_3D,
            device=device,
        )
    except Exception as e:
        print(f"  [warn] compute_masks failed: {e}", flush=True)
        masks = np.zeros(prob.shape, dtype=np.uint16)

    return masks.astype(np.uint32)


def compute_contours_from_labels(
    labels: np.ndarray,
    smooth_sigma: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert label map to foreground and contour maps via ultrack's labels_to_contours.

    Parameters
    ----------
    labels : (Z, Y, X) or (Y, X) uint32
        Label array from cellpose.dynamics.compute_masks
    smooth_sigma : float
        Gaussian sigma for smoothing the contour map (0 = no smoothing)

    Returns
    -------
    foreground : np.ndarray, float32 in [0, 1]
        Foreground probability (1 = cell interior, 0 = background)
    contours : np.ndarray, float32 in [0, 1]
        Contour map (1 = strong boundary, 0 = cell interior)
    """
    from ultrack.utils import labels_to_contours

    # labels_to_contours expects a list of label arrays
    fg, ucm = labels_to_contours([labels])
    fg = np.asarray(fg, dtype=np.float32)
    ucm = np.asarray(ucm, dtype=np.float32)

    # Optional smoothing of the contour map
    if smooth_sigma > 0:
        ucm = ndi.gaussian_filter(ucm, sigma=smooth_sigma)
        ucm_max = ucm.max()
        if ucm_max > 0:
            ucm = ucm / ucm_max

    return fg, ucm


def compute_single(
    dp_path: str | Path,
    prob_path: str | Path,
    cfg: CellposeContoursConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute labels, foreground, and contours for a single timepoint.

    Returns
    -------
    labels : np.ndarray, uint32
    foreground : np.ndarray, float32
    contours : np.ndarray, float32
    """
    labels = compute_labels_single(dp_path, prob_path, cfg)
    foreground, contours = compute_contours_from_labels(labels, smooth_sigma=cfg.smooth_sigma)
    return labels, foreground, contours


def compute_single_from_arrays(
    dp: np.ndarray,
    prob: np.ndarray,
    cfg: CellposeContoursConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute labels, foreground, and contours from pre-loaded arrays.

    Parameters
    ----------
    dp : np.ndarray, float32
        Flow field (dP)
    prob : np.ndarray, float32
        Probability map
    cfg : CellposeContoursConfig

    Returns
    -------
    labels : np.ndarray, uint32
    foreground : np.ndarray, float32
    contours : np.ndarray, float32
    """
    from cellpose.dynamics import compute_masks
    import torch

    try:
        device = torch.device(cfg.device)
        masks = compute_masks(
            dp,
            prob,
            cellprob_threshold=cfg.cellprob_threshold,
            do_3D=cfg.do_3D,
            device=device,
        )
    except Exception as e:
        print(f"[warn] compute_masks failed: {e}", flush=True)
        masks = np.zeros(prob.shape, dtype=np.uint16)

    labels = masks.astype(np.uint32)
    foreground, contours = compute_contours_from_labels(labels, smooth_sigma=cfg.smooth_sigma)
    return labels, foreground, contours


# ── Batch run ───────────────────────────────────────────────────────────────


def run(
    input_dir: str | Path,
    output_dir: str | Path,
    cfg: CellposeContoursConfig,
    overwrite: bool = False,
) -> Generator[tuple[int, int, str], None, None]:
    """Process the Cellpose cell stack and write foreground.tif and contours.tif.

    Yields ``(done, total, status_label)`` for progress reporting.
    If both output files already exist and *overwrite* is False, skips immediately.
    """
    inp = Path(input_dir)
    out = Path(output_dir)
    fg_path = out / "foreground.tif"
    ct_path = out / "contours.tif"

    # Early exit if outputs exist
    if not overwrite and fg_path.exists() and ct_path.exists():
        yield (0, 1, "foreground.tif and contours.tif already exist, skipping")
        return

    # Discover the stacked Cellpose outputs.
    dp_files = discover_dp_files(inp)
    prob_files = discover_prob_files(inp)

    if not dp_files or not prob_files:
        yield (0, 0, "No cell_dp.tif or cell_prob.tif files found")
        return

    if len(dp_files) != len(prob_files):
        yield (0, 0, f"Mismatch: {len(dp_files)} dp files vs {len(prob_files)} prob files")
        return

    # Build threshold sweep from config sweep fields
    if cfg.cellprob_step > 0 and cfg.cellprob_min != cfg.cellprob_max:
        thresholds = list(np.arange(cfg.cellprob_min, cfg.cellprob_max + cfg.cellprob_step / 2, cfg.cellprob_step))
    else:
        thresholds = [cfg.cellprob_threshold]

    if not thresholds:
        yield (0, 0, "Error: no cellprob thresholds generated (check cellprob_min/max/step)")
        return

    dp_stack = tifffile.imread(str(dp_files[0])).astype(np.float32)
    prob_stack = tifffile.imread(str(prob_files[0])).astype(np.float32)
    if dp_stack.ndim == 4:
        dp_stack = dp_stack[np.newaxis]
    if prob_stack.ndim == 3:
        prob_stack = prob_stack[np.newaxis]
    if dp_stack.shape[0] != prob_stack.shape[0]:
        yield (0, 0, f"Timepoint mismatch: dp={dp_stack.shape[0]} prob={prob_stack.shape[0]}")
        return

    total = dp_stack.shape[0]
    out.mkdir(parents=True, exist_ok=True)

    fg_frames: list[np.ndarray] = []
    ct_frames: list[np.ndarray] = []
    # maps threshold → list of per-timepoint label arrays (only populated when save_masks)
    masks_per_thresh: dict[float, list[np.ndarray]] = {t: [] for t in thresholds} if cfg.save_masks else {}

    for i in range(total):
        t_str = f"t{i:03d}"
        try:
            dp = dp_stack[i]
            prob = prob_stack[i]
            fg_list = []
            ct_list = []
            for thresh in thresholds:
                cfg.cellprob_threshold = thresh
                labels, fg, ct = compute_single_from_arrays(dp, prob, cfg)
                fg_list.append(fg)
                ct_list.append(ct)
                if cfg.save_masks:
                    masks_per_thresh[thresh].append(labels)
            fg_frames.append(np.mean(fg_list, axis=0).astype(np.float32))
            ct_frames.append(np.mean(ct_list, axis=0).astype(np.float32))
        except Exception as e:
            print(f"  {t_str}: error: {e}", flush=True)
            spatial = prob_stack[i].shape
            fg_frames.append(np.zeros(spatial, dtype=np.float32))
            ct_frames.append(np.zeros(spatial, dtype=np.float32))
            if cfg.save_masks:
                for thresh in thresholds:
                    masks_per_thresh[thresh].append(np.zeros(spatial, dtype=np.uint32))

        yield (i + 1, total, t_str)

    fg_stack = np.stack(fg_frames, axis=0)
    ct_stack = np.stack(ct_frames, axis=0)

    tifffile.imwrite(str(fg_path), fg_stack, compression="zlib")
    tifffile.imwrite(str(ct_path), ct_stack, compression="zlib")

    if cfg.save_masks:
        masks_3d_dir = out / "masks" / "3d"
        masks_2d_dir = out / "masks" / "2d"
        masks_3d_dir.mkdir(parents=True, exist_ok=True)
        masks_2d_dir.mkdir(parents=True, exist_ok=True)
        for thresh, frames in masks_per_thresh.items():
            thresh_str = f"{thresh:.2f}".replace("-", "m").replace(".", "p")
            stack_3d = np.stack(frames, axis=0)  # (T, Z, Y, X) or (T, Y, X)
            tifffile.imwrite(str(masks_3d_dir / f"thresh_{thresh_str}_masks.tif"), stack_3d, compression="zlib")
            # Max-projection over Z for 4-D stacks, pass through for 3-D
            proj_2d = stack_3d.max(axis=1) if stack_3d.ndim == 4 else stack_3d
            tifffile.imwrite(str(masks_2d_dir / f"thresh_{thresh_str}_masks.tif"), proj_2d, compression="zlib")

    yield (total, total, "Done")


# ── CLI entry point ─────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="s02c — compute contours from Cellpose flow fields and probability maps",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing cell_dp.tif and cell_prob.tif files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write foreground.tif and contours.tif",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON file with CellposeContoursConfig fields (optional)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    args = parser.parse_args()

    cfg_dict: dict = {}
    if args.config:
        cfg_dict = json.loads(Path(args.config).read_text())
    cfg = CellposeContoursConfig(**cfg_dict)

    for done, total, label in run(args.input_dir, args.output_dir, cfg, overwrite=args.overwrite):
        print(f"[{done}/{total}] {label}", flush=True)

    sys.exit(0)


# ── StageProtocol wrapper ────────────────────────────────────────────────────


class _ContoursStageClass:
    name = "contours"
    display_name = "Cellpose Contours"

    def __init__(self):
        self.config = CellposeContoursConfig()

    def run(self, input_dir, output_dir, cfg: CellposeContoursConfig = None, overwrite: bool = False):
        from cellflow.core.logging import StageLogger
        from cellflow.core.paths import log_path

        cfg = cfg or self.config
        log = StageLogger(log_path(output_dir, 0), self.name)
        with log:
            for progress in run(input_dir=input_dir, output_dir=output_dir, cfg=cfg, overwrite=overwrite):
                yield StageProgress(*progress)

    def validate_inputs(self, schema, root_dir, pos) -> ValidationResult:
        from cellflow.core.validation import validate_inputs

        cell_dir = stage_dir(root_dir, pos, "cellpose_nucleus")
        dp_files = [cell_dir / "cell_dp.tif"] if (cell_dir / "cell_dp.tif").exists() else []
        if not dp_files:
            return ValidationResult(ok=False, errors=[f"No cell_dp.tif in {cell_dir}"])
        return ValidationResult(ok=True, errors=[])

    def is_complete(self, root_dir, pos) -> bool:
        d = stage_dir(root_dir, pos, "contours")
        return (d / "foreground.tif").exists() and (d / "contours.tif").exists()


ContoursStage = _ContoursStageClass()
