from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from cellflow.tracking_ultrack.corrections import Correction


@dataclass(frozen=True)
class AnchorRemovalResult:
    remaining: list[Correction]
    removed: bool


def without_anchor_correction(
    corrections: Iterable[Correction],
    *,
    cell_id: int,
    frame: int,
) -> AnchorRemovalResult:
    """Return corrections with the matching anchor removed, if present."""
    selected_cell_id = int(cell_id)
    selected_frame = int(frame)
    original = list(corrections)
    remaining = [
        correction
        for correction in original
        if not (
            int(correction.cell_id) == selected_cell_id
            and int(correction.t) == selected_frame
            and correction.kind == "anchor"
        )
    ]
    return AnchorRemovalResult(
        remaining=remaining,
        removed=len(remaining) != len(original),
    )


def anchor_correction(
    *,
    cell_id: int,
    frame: int,
    y: float,
    x: float,
) -> Correction:
    """Build an anchor correction for the selected cell and frame."""
    return Correction(
        cell_id=int(cell_id),
        t=int(frame),
        kind="anchor",
        y=float(y),
        x=float(x),
    )
