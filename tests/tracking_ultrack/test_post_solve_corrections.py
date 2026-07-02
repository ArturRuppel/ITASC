"""Post-solve correction relabeling — the array-only path (no DB / ultrack).

``apply_post_solve_corrections`` with ``working_dir=None`` never touches the
solver database, so these run everywhere regardless of the optional ``ultrack``
dependency.
"""
from __future__ import annotations

import numpy as np

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.corrections import (
    Correction,
    apply_post_solve_corrections,
)


def _two_track_frame() -> np.ndarray:
    labels = np.zeros((1, 6, 12), dtype=np.int32)
    labels[0, 1:4, 1:4] = 20  # solver track 20, centroid ~ (2, 2)
    labels[0, 1:4, 8:11] = 30  # solver track 30, centroid ~ (2, 9)
    return labels


def test_collision_pass_preserves_anchored_track_under_id_coincidence():
    """A solver track whose numeric id coincides with a *different* anchored
    cell id must survive the reserved-id collision pass.

    Anchor cell 10 matches solver track 20; anchor cell 20 matches solver track
    30. This yields solver_track_remap = {20: 10, 30: 20}. The collision loop
    reaches reserved id 20 and, without the source guard, would relabel the
    pixels of solver track 20 (destined to become cell 10) to a fresh id — so
    the later 20->10 remap finds nothing and anchored cell 10 vanishes.
    """
    corrections = [
        Correction(cell_id=10, t=0, kind="anchor", y=2.0, x=2.0),
        Correction(cell_id=20, t=0, kind="anchor", y=2.0, x=9.0),
    ]
    cfg = TrackingConfig(anchor_radius_px=3.0)

    labels, report = apply_post_solve_corrections(
        _two_track_frame(), corrections, _two_track_frame(), cfg, working_dir=None
    )
    present = {int(v) for v in np.unique(labels)}

    # Both anchored identities survive...
    assert 10 in present, "anchored cell 10 lost its guaranteed identity"
    assert 20 in present
    # ...and land on the correct regions (10 where solver track 20 was).
    assert labels[0, 2, 2] == 10
    assert labels[0, 2, 9] == 20
    assert report.remapped_anchor_tracks == 2
