"""Unit tests for cellflow.segmentation.error_scan."""
from __future__ import annotations

import numpy as np

from cellflow.segmentation.error_scan import scan_errors


def _stack(t: int = 4, size: int = 8) -> np.ndarray:
    return np.zeros((t, size, size), dtype=np.uint32)


def test_high_divergence_cell_is_flagged() -> None:
    arr = _stack()
    for t in range(4):
        arr[t, 0:2, 0:2] = 1
    contours = np.zeros_like(arr, dtype=np.float32)
    contours[0, 0:2, 0:2] = 9.0  # cell 1 sits on strong boundary signal at t=0

    errors = scan_errors(arr, contours)

    hit = next(e for e in errors if e.t == 0 and e.cell_id == 1)
    assert "high boundary divergence" in hit.reasons
    assert hit.score == 1.0  # normalised against the max per-cell mean


def test_orphan_track_is_flagged_at_first_frame() -> None:
    arr = _stack()
    arr[2, 5, 5] = 7  # exists for a single frame

    errors = scan_errors(arr, None)

    hit = next(e for e in errors if e.cell_id == 7)
    assert hit.t == 2
    assert hit.reasons == ("short track (1 frame)",)


def test_gap_in_track_is_flagged_on_reappearance() -> None:
    arr = _stack()
    arr[0, 4:6, 4:6] = 3
    arr[1, 4:6, 4:6] = 3
    # frame 2 missing
    arr[3, 4:6, 4:6] = 3

    errors = scan_errors(arr, None)

    hit = next(e for e in errors if e.cell_id == 3 and e.t == 3)
    assert hit.reasons == ("reappears after 1-frame gap",)


def test_area_jump_between_adjacent_frames_is_flagged() -> None:
    arr = _stack()
    arr[0, 0:1, 0:1] = 4  # area 1
    arr[1, 0:4, 0:4] = 4  # area 16 -> ×16 jump

    errors = scan_errors(arr, None)

    hit = next(e for e in errors if e.cell_id == 4 and e.t == 1)
    assert any("area ×" in r for r in hit.reasons)


def test_results_sorted_by_descending_score() -> None:
    arr = _stack()
    for t in range(4):
        arr[t, 0:2, 0:2] = 1
    arr[0, 6, 6] = 2  # orphan
    contours = np.zeros_like(arr, dtype=np.float32)
    contours[0, 0:2, 0:2] = 9.0

    errors = scan_errors(arr, contours)

    scores = [e.score for e in errors]
    assert scores == sorted(scores, reverse=True)


def test_max_results_is_respected() -> None:
    arr = _stack(t=1, size=40)
    # 100 single-pixel orphan cells.
    for i in range(100):
        arr[0, i // 40 * 2, i % 40] = i + 1

    errors = scan_errors(arr, None, max_results=10)

    assert len(errors) == 10


def test_singleton_z_axis_is_squeezed() -> None:
    arr = np.zeros((3, 1, 8, 8), dtype=np.uint32)
    arr[0, 0, 5, 5] = 9

    errors = scan_errors(arr, None)

    assert any(e.cell_id == 9 for e in errors)


def test_no_errors_on_clean_stable_stack() -> None:
    arr = _stack()
    for t in range(4):
        arr[t, 0:3, 0:3] = 1  # one stable cell, no gaps/jumps
    errors = scan_errors(arr, None)
    assert errors == []
