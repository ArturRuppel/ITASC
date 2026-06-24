"""Joint nucleus-anchored cell segmentation + tracking, Qt-free.

When the standalone tool has both channels, this composes the existing pieces
into one nucleus-anchored path:

1. **Nucleus native masks** (:func:`native_masks.run_nucleus_masks_stack`) →
   **track** them (:func:`track_laptrack.track_masks`) → tracked nucleus labels.
2. **Cell flows** (:func:`cellpose_runner.run_cell_stack` → ``prob``, ``dp``);
   the cell foreground is ``sigmoid(prob) > fg_threshold``.
3. Per z-plane, **flow-follow** each cell-foreground pixel to a nucleus
   (:func:`flow_following.flow_follow_movie`). Cell labels inherit the *tracked*
   nucleus ids, so the cell stack is tracked by construction — one cell per
   nucleus, sharing its id.

Everything here reuses functions the app also imports (read-only); it adds no
new behaviour to the app's Cellpose stage.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from cellflow.cellpose import cellpose_runner, native_masks, track_laptrack
from cellflow.cellpose.cellpose_runner import CellParams, NucleusParams
from cellflow.cellpose.flow_following import FlowFollowingParams, flow_follow_movie

__all__ = ["cell_foreground_from_prob", "joint_segment_track"]


def cell_foreground_from_prob(prob: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean cell foreground from a Cellpose probability map via the sigmoid.

    Matches the app's ``sigmoid(prob)`` foreground convention; ``threshold`` is
    the exposed cutoff on the ``[0, 1]`` sigmoid.
    """
    sig = 1.0 / (1.0 + np.exp(-np.asarray(prob, dtype=np.float32)))
    return sig > float(threshold)


def joint_segment_track(
    nucleus_stack: np.ndarray,
    cell_stack: np.ndarray,
    nucleus_params: NucleusParams,
    cell_params: CellParams,
    flow_params: FlowFollowingParams,
    *,
    max_distance: float = 15.0,
    max_frame_gap: int = 0,
    progress_cb: Callable[[int, int, str], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the joint path; return ``(nucleus_tracked, cell_tracked)`` ``(T,Z,Y,X)``.

    Both outputs are ``int32`` and share label ids: a cell carries the id of the
    nucleus it was assigned to. ``max_distance``/``max_frame_gap`` tune the
    nucleus tracker; the cell stack inherits its tracks.
    """
    nucleus_stack = np.asarray(nucleus_stack)
    cell_stack = np.asarray(cell_stack)
    if nucleus_stack.ndim != 4 or cell_stack.ndim != 4:
        raise ValueError("nucleus_stack and cell_stack must be (T, Z, Y, X)")

    def _report(frac_done: int, total: int, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(frac_done, total, msg)

    # 1. Nucleus masks → tracked nucleus labels.
    _report(0, 4, "Nucleus masks...")
    nuc_masks = native_masks.run_nucleus_masks_stack(
        nucleus_stack, nucleus_params, cancel_cb=cancel_cb,
    )
    _report(1, 4, "Tracking nuclei...")
    nuc_tracked = track_laptrack.track_masks(
        nuc_masks, max_distance=max_distance, max_frame_gap=max_frame_gap,
    ).astype(np.int32)

    # 2. Cell flows + foreground.
    _report(2, 4, "Cell flows...")
    prob, dp = cellpose_runner.run_cell_stack(
        cell_stack, cell_params, cancel_cb=cancel_cb,
    )
    foreground = cell_foreground_from_prob(prob, flow_params.fg_threshold)

    # 3. Flow-follow per z-plane; cell labels inherit the tracked nucleus ids.
    _report(3, 4, "Assigning cells to nuclei...")
    T, Z = nuc_tracked.shape[:2]
    cell_tracked = np.zeros_like(nuc_tracked, dtype=np.int32)
    for z in range(Z):
        # dp is (T, Z, 2, Y, X) → per-plane (T, 2, Y, X).
        cell_tracked[:, z] = flow_follow_movie(
            foreground[:, z],
            dp[:, z],
            nuc_tracked[:, z],
            flow_params,
        )
    _report(4, 4, "Joint segmentation complete.")
    return nuc_tracked, cell_tracked
