"""Progressive Ultrack database helpers."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import tifffile
from skimage.segmentation import find_boundaries

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.db_build import (
    UltrackDatabaseBuildReport,
    _notify,
    build_ultrack_database,
)


def foreground_scores_from_logits(prob_3dt: np.ndarray) -> np.ndarray:
    """Convert 3D per-frame logits to continuous 2D foreground scores."""
    prob = np.asarray(prob_3dt, dtype=np.float32)
    if prob.ndim != 4:
        raise ValueError(
            f"Expected probability logits shaped (T, Z, Y, X), got {prob.shape}"
        )
    scores = 1.0 / (1.0 + np.exp(-prob))
    return scores.mean(axis=1).astype(np.float32, copy=False)


def contour_maps_from_masks(masks: np.ndarray) -> np.ndarray:
    """Extract inner contour maps from label masks.

    Four-dimensional masks are max-projected over Z before boundary extraction,
    preserving any labeled pixel visible in the stack.
    """
    labels = np.asarray(masks)
    if labels.ndim == 4:
        labels = labels.max(axis=1)
    elif labels.ndim != 3:
        raise ValueError(f"Expected masks shaped (T, Y, X) or (T, Z, Y, X), got {labels.shape}")

    contours = np.zeros(labels.shape, dtype=np.float32)
    for t in range(labels.shape[0]):
        contours[t] = find_boundaries(labels[t], mode="inner").astype(np.float32)
    return contours


def write_progressive_inputs(
    prob_3dt_path: str | Path,
    masks_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write continuous foreground scores and contour maps for progressive Ultrack."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    foreground = foreground_scores_from_logits(tifffile.imread(prob_3dt_path))
    contours = contour_maps_from_masks(tifffile.imread(masks_path))

    foreground_path = out_dir / "foreground_scores.tif"
    contour_path = out_dir / "contour_maps.tif"
    tifffile.imwrite(foreground_path, foreground.astype(np.float32, copy=False))
    tifffile.imwrite(contour_path, contours.astype(np.float32, copy=False))
    return foreground_path, contour_path


def build_progressive_ultrack_database(
    foreground_scores_path: str | Path,
    contour_maps_path: str | Path,
    nucleus_prob_zavg_path: str | Path,
    working_dir: str | Path,
    cfg: TrackingConfig,
    *,
    validated_tracks: dict[int, set[int]] | None = None,
    tracked_labels: np.ndarray | None = None,
    use_validated: bool = False,
    progress_cb: Callable[[str], None] | None = None,
) -> UltrackDatabaseBuildReport:
    """Build one Ultrack DB from continuous foreground scores and contour maps."""
    _notify(
        progress_cb,
        "Building continuous foreground / progressive hierarchy Ultrack database …",
    )
    return build_ultrack_database(
        contour_maps_path=contour_maps_path,
        foreground_masks_path=foreground_scores_path,
        nucleus_prob_zavg_path=nucleus_prob_zavg_path,
        working_dir=working_dir,
        cfg=cfg,
        validated_tracks=validated_tracks,
        tracked_labels=tracked_labels,
        use_validated=use_validated,
        progress_cb=progress_cb,
    )
