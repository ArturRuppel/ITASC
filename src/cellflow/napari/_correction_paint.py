"""Paint label assignments into a single tracked-labels frame.

Shared by the extend paths (keyboard A/D and the candidate-gallery pick): each
assignment carries a ``cell_id`` and a boolean ``mask_2d``. The frame is mutated
in place — every assigned id is first cleared, then repainted either greedily
(overwriting anything not protected) or only into background — and the set of
ids whose pixels actually changed is returned so the caller can recolour just
those. Kept pure (no widget / napari state) so it is unit-tested directly.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def paint_assignments(
    frame: np.ndarray,
    assignments: Iterable,
    protected_mask: np.ndarray,
    *,
    greedy: bool,
) -> set[int]:
    """Paint ``assignments`` into ``frame``; return the changed cell ids.

    ``greedy`` overwrites everything outside ``protected_mask``; otherwise an
    assignment only fills background (``frame == 0``) pixels.
    """
    assignments = tuple(assignments)
    before = np.asarray(frame).copy()
    for a in assignments:
        frame[frame == int(a.cell_id)] = 0
    if greedy:
        for a in assignments:
            frame[a.mask_2d & ~protected_mask] = int(a.cell_id)
    else:
        for a in assignments:
            frame[a.mask_2d & (frame == 0)] = int(a.cell_id)
    changed = before != frame
    changed_ids = (
        set(int(v) for v in np.unique(before[changed]))
        | set(int(v) for v in np.unique(np.asarray(frame)[changed]))
    )
    changed_ids.discard(0)
    return changed_ids
