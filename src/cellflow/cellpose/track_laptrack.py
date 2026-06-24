"""Link per-frame native masks across time with laptrack, Qt-free.

The standalone ``cellflow-cellpose`` tool segments each frame independently
(:mod:`cellflow.cellpose.native_masks`), producing labels that are *not*
consistent across time. This module closes that gap: it computes per-frame
centroids, runs a linear-assignment tracker (``laptrack``) to link objects
between consecutive frames, and relabels the stack so a tracked object keeps one
id over its whole lifetime.

``laptrack`` (and ``pandas``) are imported lazily inside :func:`track_masks`, so
importing this module — and the centroid / relabel helpers, which only need
numpy + scikit-image — does not require the optional ``[laptrack]`` extra.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "build_track_dataframe",
    "relabel_by_tracks",
    "track_masks",
    "COORDINATE_COLS",
]

#: Centroid coordinate columns used for linking; ``z`` is constant for 2D input.
COORDINATE_COLS = ["z", "y", "x"]


def build_track_dataframe(masks_tzyx: np.ndarray):
    """Build a per-object centroid table from a ``(T, Z, Y, X)`` label stack.

    Returns a :class:`pandas.DataFrame` with columns ``frame, label, z, y, x``
    (one row per labelled object per frame). ``z`` is always present and is
    constant ``0`` for single-slice (2D) input, so linking is uniform.
    """
    import pandas as pd
    from skimage.measure import regionprops_table

    if masks_tzyx.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X), got shape {masks_tzyx.shape}")
    rows = []
    for t in range(masks_tzyx.shape[0]):
        volume = np.asarray(masks_tzyx[t])
        if not volume.any():
            continue
        table = regionprops_table(volume, properties=("label", "centroid"))
        n = len(table["label"])
        # centroid-0/1/2 -> z/y/x (3D volume always yields three centroid cols).
        z = table.get("centroid-0", np.zeros(n))
        y = table.get("centroid-1", np.zeros(n))
        x = table.get("centroid-2", np.zeros(n))
        for i in range(n):
            rows.append(
                {
                    "frame": t,
                    "label": int(table["label"][i]),
                    "z": float(z[i]),
                    "y": float(y[i]),
                    "x": float(x[i]),
                }
            )
    return pd.DataFrame(rows, columns=["frame", "label", "z", "y", "x"])


def relabel_by_tracks(masks_tzyx: np.ndarray, track_of: dict) -> np.ndarray:
    """Relabel ``(T, Z, Y, X)`` masks by a ``(frame, orig_label) -> track_id`` map.

    Output labels are ``track_id + 1`` (so tracks start at ``1`` and background
    stays ``0``). Original labels absent from ``track_of`` are dropped to ``0``.
    """
    if masks_tzyx.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X), got shape {masks_tzyx.shape}")
    out = np.zeros_like(masks_tzyx, dtype=np.int32)
    for t in range(masks_tzyx.shape[0]):
        frame = np.asarray(masks_tzyx[t])
        max_label = int(frame.max())
        if max_label == 0:
            continue
        lut = np.zeros(max_label + 1, dtype=np.int32)
        for orig_label in range(1, max_label + 1):
            track_id = track_of.get((t, orig_label))
            if track_id is not None:
                lut[orig_label] = int(track_id) + 1
        out[t] = lut[frame]
    return out


def _run_laptrack(df, *, max_distance: float, max_frame_gap: int):
    """Run laptrack on a centroid dataframe; return it with a ``track_id`` column.

    Isolated so :func:`track_masks` orchestration can be tested without the
    optional dependency installed (tests monkeypatch this).
    """
    from laptrack import LapTrack

    cutoff = float(max_distance) ** 2  # sqeuclidean metric
    lt = LapTrack(
        track_dist_metric="sqeuclidean",
        track_cost_cutoff=cutoff,
        gap_closing_dist_metric="sqeuclidean",
        gap_closing_cost_cutoff=cutoff if max_frame_gap > 0 else False,
        gap_closing_max_frame_count=int(max_frame_gap),
        splitting_cost_cutoff=False,
        merging_cost_cutoff=False,
    )
    track_df, _split_df, _merge_df = lt.predict_dataframe(
        df,
        coordinate_cols=COORDINATE_COLS,
        frame_col="frame",
        only_coordinate_cols=False,
    )
    return track_df.reset_index(drop=True)


def track_masks(
    masks_tzyx: np.ndarray,
    *,
    max_distance: float = 15.0,
    max_frame_gap: int = 0,
) -> np.ndarray:
    """Link per-frame masks across time and return a track-consistent stack.

    ``max_distance`` is the maximum centroid displacement (in pixels, over
    ``z/y/x``) allowed for a link; ``max_frame_gap`` > 0 enables gap-closing over
    that many missed frames. Output is ``(T, Z, Y, X)`` ``int32`` with labels
    stable across time (background ``0``).
    """
    if masks_tzyx.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X), got shape {masks_tzyx.shape}")
    df = build_track_dataframe(masks_tzyx)
    if df.empty:
        return np.zeros_like(masks_tzyx, dtype=np.int32)
    tracked_df = _run_laptrack(df, max_distance=max_distance, max_frame_gap=max_frame_gap)
    track_of = {
        (int(r.frame), int(r.label)): int(r.track_id)
        for r in tracked_df.itertuples(index=False)
    }
    return relabel_by_tracks(masks_tzyx, track_of)
