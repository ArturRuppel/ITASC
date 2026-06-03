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
    arr[1, 0, 5, 5] = 9  # interior frame so the short-track flag still fires

    errors = scan_errors(arr, None)

    assert any(e.cell_id == 9 for e in errors)


def test_no_errors_on_clean_stable_stack() -> None:
    arr = _stack()
    for t in range(4):
        arr[t, 0:3, 0:3] = 1  # one stable cell, no gaps/jumps
    errors = scan_errors(arr, None)
    assert errors == []


def _stable_cells(*, n: int, t: int = 3, size: int = 16) -> np.ndarray:
    """``n`` stable 2x2 cells present in every frame (no gaps/jumps/orphans)."""
    arr = np.zeros((t, size, size), dtype=np.uint32)
    for i in range(n):
        y, x = 1 + 3 * (i // 4), 1 + 3 * (i % 4)
        arr[:, y : y + 2, x : x + 2] = i + 1
    return arr


def test_uniform_divergence_flags_nothing() -> None:
    # A genuinely uniform contour field must not flag a fixed top-decile: with
    # every cell equally "divergent" there is no outlier to surface.
    arr = _stable_cells(n=6)
    contours = np.full_like(arr, 5.0, dtype=np.float32)

    errors = scan_errors(arr, contours)

    assert not any("high boundary divergence" in e.reasons for e in errors)


def test_only_a_clear_divergence_outlier_is_flagged() -> None:
    # Most cells sit on low contour; one sits on a strong boundary. Only the
    # outlier is flagged for divergence — not the low-signal majority.
    arr = _stable_cells(n=6)
    contours = np.ones_like(arr, dtype=np.float32)  # baseline signal under all
    contours[:, 1:3, 1:3] = 50.0  # cell 1 sits on a strong boundary

    errors = scan_errors(arr, contours)

    div = [e for e in errors if "high boundary divergence" in e.reasons]
    assert {e.cell_id for e in div} == {1}


def test_short_track_at_fov_boundary_is_not_flagged() -> None:
    # 1-frame tracks at the first/last frame are field-of-view entry/exit, not
    # errors; an interior 1-frame track still is.
    arr = _stack(t=4)
    arr[0, 1, 1] = 10   # touches first frame
    arr[3, 2, 2] = 11   # touches last frame
    arr[1, 3, 3] = 12   # interior

    errors = scan_errors(arr, None)
    flagged = {e.cell_id for e in errors}

    assert 10 not in flagged and 11 not in flagged
    assert 12 in flagged


def test_tiny_area_wobble_is_not_flagged() -> None:
    # A 1->3 px wobble clears the ratio but not the area floor: it's noise, not
    # a merge/split, so nothing is flagged.
    arr = _stack(t=2)
    arr[0, 0, 0] = 4          # area 1
    arr[1, 0:1, 0:3] = 4      # area 3 (-> ×3, but max mask is sub-floor)

    errors = scan_errors(arr, None)

    assert errors == []


def test_all_scores_are_within_the_unit_interval() -> None:
    arr = _stable_cells(n=6)
    contours = np.ones_like(arr, dtype=np.float32)
    contours[:, 1:3, 1:3] = 50.0
    arr[1, 12, 12] = 99  # add an interior orphan for reason variety

    errors = scan_errors(arr, contours)

    assert errors  # the fixture produces at least one flag
    assert all(0.0 <= e.score <= 1.0 for e in errors)
