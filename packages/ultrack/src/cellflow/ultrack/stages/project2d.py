"""s04 — 3D to 2D label projection (nearest-centroid assignment for conflicts)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from cellflow.core.paths import stage_dir
from cellflow.core.protocol import StageProgress, ValidationResult


def _project_frame(vol: np.ndarray) -> np.ndarray:
    """Project a single (Z, Y, X) label frame to (Y, X).

    For each (y, x) pixel, if only one non-zero label appears across Z slices it
    is used directly.  When multiple labels overlap at the same pixel the one
    whose 2D centroid (averaged over all its voxels in this frame) is closest to
    (y, x) wins.
    """
    Z, Y, X = vol.shape
    flat = vol.reshape(Z, -1)           # (Z, Y*X)
    max_proj = flat.max(axis=0)         # (Y*X,)  — correct for non-conflicting pixels

    # Conflict pixels: at least one z has a non-zero label != the max-projected label
    has_other = np.any((flat > 0) & (flat != max_proj[np.newaxis, :]), axis=0)
    conflict_pixels = np.flatnonzero(has_other)

    if conflict_pixels.size == 0:
        return max_proj.reshape(Y, X).astype(np.uint32)

    # 2D centroids for every label present in this frame
    unique_labels = np.unique(vol)
    unique_labels = unique_labels[unique_labels > 0]

    # voxel coordinates
    zz, yy, xx = np.nonzero(vol)
    voxel_labels = vol[zz, yy, xx]

    # centroid_y[i], centroid_x[i]  for  unique_labels[i]
    centroid_y = np.zeros(len(unique_labels))
    centroid_x = np.zeros(len(unique_labels))
    for i, lab in enumerate(unique_labels):
        mask = voxel_labels == lab
        centroid_y[i] = yy[mask].mean()
        centroid_x[i] = xx[mask].mean()

    # For each conflict pixel, find which labels are present and pick the nearest
    conflict_cols = flat[:, conflict_pixels]            # (Z, n_conflicts)
    conflict_ys, conflict_xs = np.unravel_index(conflict_pixels, (Y, X))
    pix_y = conflict_ys.astype(float)                  # (n_conflicts,)
    pix_x = conflict_xs.astype(float)

    # label_present[i, j] = True if unique_labels[i] appears at conflict pixel j
    label_present = np.zeros((len(unique_labels), conflict_pixels.size), dtype=bool)
    for i, lab in enumerate(unique_labels):
        label_present[i] = np.any(conflict_cols == lab, axis=0)

    # squared distance from each label centroid to each conflict pixel  (n_labels, n_conflicts)
    dy = centroid_y[:, np.newaxis] - pix_y[np.newaxis, :]
    dx = centroid_x[:, np.newaxis] - pix_x[np.newaxis, :]
    dists = dy ** 2 + dx ** 2
    dists[~label_present] = np.inf     # ignore labels absent at this pixel

    best_idx = dists.argmin(axis=0)    # (n_conflicts,)
    best_labels = unique_labels[best_idx]

    result = max_proj.copy()
    result[conflict_pixels] = best_labels
    return result.reshape(Y, X).astype(np.uint32)


def project_labels_to_2d(labels: np.ndarray) -> np.ndarray:
    """Project a (T, Z, Y, X) label volume to (T, Y, X).

    Conflict pixels — where multiple non-zero labels overlap in the Z column —
    are resolved by assigning each pixel to the label whose 2D centroid is closest.

    Parameters
    ----------
    labels : np.ndarray
        Shape ``(T, Z, Y, X)`` or ``(T, Y, X)``.  If already (T, Y, X) the
        array is returned as uint32 unchanged.

    Returns
    -------
    np.ndarray
        Shape ``(T, Y, X)`` uint32.
    """
    if labels.ndim == 3:
        return labels.astype(np.uint32)
    if labels.ndim != 4:
        raise ValueError(f"Expected 3D or 4D label array, got shape {labels.shape}")

    T = labels.shape[0]
    frames = [_project_frame(labels[t]) for t in range(T)]
    return np.stack(frames, axis=0)


def export_nuclear_labels_2d(
    tracked_labels_path: str | Path,
    output_path: str | Path,
) -> None:
    """Load ``tracked_labels.tif`` and write a centroid-projected ``nuclear_labels_2d.tif``.

    Parameters
    ----------
    tracked_labels_path : path-like
        Path to ``tracked_labels.tif`` (T, Z, Y, X) or (T, Y, X).
    output_path : path-like
        Destination file, typically ``<working_dir>/nuclear_labels_2d.tif``.
    """
    labels = tifffile.imread(str(tracked_labels_path))
    proj = project_labels_to_2d(labels)
    tifffile.imwrite(str(output_path), proj, compression="zlib")


class _Project2DStageClass:
    name = "project2d"
    display_name = "3D → 2D Projection"

    def __init__(self):
        self.config = None

    def run(
        self,
        working_dir: str | Path,
        **kwargs,
    ):
        """Project ``tracked_labels.tif`` to 2D and write ``nuclear_labels_2d.tif``."""
        wd = Path(working_dir)
        src = wd / "tracked_labels.tif"
        dst = wd / "nuclear_labels_2d.tif"

        if not src.exists():
            raise FileNotFoundError(
                f"tracked_labels.tif not found in {wd}. Run the Tracking stage first."
            )

        yield StageProgress(0, 2, "Projecting tracked labels to 2D…")
        export_nuclear_labels_2d(src, dst)
        yield StageProgress(2, 2, "2D projection done.")

    def validate_inputs(self, schema, root_dir, pos) -> ValidationResult:
        from cellflow.core.validation import validate_inputs

        tracking_dir = stage_dir(root_dir, pos, "tracking")
        return validate_inputs([tracking_dir / "tracked_labels.tif"])

    def is_complete(self, root_dir, pos) -> bool:
        d = stage_dir(root_dir, pos, "tracking")
        return (d / "nuclear_labels_2d.tif").exists()


Project2DStage = _Project2DStageClass()
