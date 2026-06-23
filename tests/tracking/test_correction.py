
from cellflow.correction.labels import (
    _free_label,
    add_cell,
    carve_into_selected,
    clean_stranded_pixels,
    draw_cell_path,
    expand_label_to_foreground,
    fill_label_holes,
    fix_label_semiholes,
    split_across,
)
from cellflow.segmentation import apply_gamma
import numpy as np
import pytest


def test_correction_package_does_not_reexport_apply_gamma():
    import cellflow.correction as correction

    assert not hasattr(correction, "apply_gamma")


def test_gamma_identity():
    logits = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(apply_gamma(logits, 1.0), logits)


def test_free_label_returns_next_id():
    seg = np.zeros((4, 4), dtype=np.uint16)
    seg[1, 1] = 5
    assert _free_label(seg) == 6


def test_free_label_raises_at_dtype_ceiling():
    """max+1 must not wrap to 0/collide when seg is at the uint16 ceiling."""
    seg = np.zeros((4, 4), dtype=np.uint16)
    seg[0, 0] = np.iinfo(np.uint16).max
    with pytest.raises(OverflowError):
        _free_label(seg)

def test_gamma_values():
    logits = np.array([0.0]) # prob 0.5
    # gamma 2.0 -> prob 0.25 -> logit log(0.25/0.75) = log(1/3) = -1.0986
    corrected = apply_gamma(logits, 2.0)
    assert np.isclose(corrected, np.log(1/3))
    
    # gamma 0.5 -> prob 0.707 -> logit log(0.707/0.293) = 0.857
    corrected = apply_gamma(logits, 0.5)
    assert np.isclose(corrected, np.log(np.sqrt(0.5) / (1 - np.sqrt(0.5))))


def test_draw_cell_path_rejects_disconnected_extension():
    seg = np.zeros((12, 12), dtype=np.uint32)
    seg[1:3, 1:3] = 1
    before = seg.copy()

    positions = [
        (7, 7),
        (7, 10),
        (10, 10),
        (10, 7),
    ]

    ok = draw_cell_path(seg, positions, curlabel=1)

    assert ok is False
    assert np.array_equal(seg, before)


def test_draw_cell_path_extension_fills_enclosed_holes():
    seg = np.zeros((12, 12), dtype=np.uint32)
    seg[4:8, 4:8] = 1
    seg[5:7, 5:7] = 0

    positions = [
        (4, 7),
        (4, 9),
        (7, 9),
        (7, 7),
    ]

    ok = draw_cell_path(seg, positions, curlabel=1)

    assert ok is True
    assert seg[5, 5] == 1
    assert seg[0, 0] == 0


def test_split_across_uses_caller_supplied_fresh_label():
    stack = np.zeros((2, 12, 12), dtype=np.uint32)
    stack[0, 2:10, 2:10] = 5
    stack[1, 1:3, 1:3] = 6

    ok = split_across(stack[0], None, (3, 3), (8, 8), new_label=7)

    assert ok is True
    assert 7 in stack[0]
    assert 6 not in stack[0]


def test_draw_cell_path_uses_caller_supplied_fresh_label_for_new_cell():
    stack = np.zeros((2, 12, 12), dtype=np.uint32)
    stack[0, 1:3, 1:3] = 5
    stack[1, 8:10, 8:10] = 6

    ok = draw_cell_path(stack[0], [(5, 1), (5, 4), (8, 4), (8, 1)], new_label=7)

    assert ok is True
    assert 7 in stack[0]
    assert 6 not in stack[0]


def test_draw_cell_path_extension_preserves_protected_pixels():
    seg = np.zeros((12, 12), dtype=np.uint32)
    seg[3:6, 3:6] = 1
    seg[4:8, 6:9] = 9
    seg[7:10, 6:9] = 8
    protected_mask = seg == 9

    ok = draw_cell_path(
        seg,
        [(3, 5), (3, 10), (10, 10), (10, 5)],
        curlabel=1,
        protected_mask=protected_mask,
    )

    assert ok is True
    assert np.all(seg[protected_mask] == 9)
    assert np.any(seg[7:10, 6:9] == 1)


def test_carve_into_selected_cuts_neighbour_and_merges_near_piece():
    # A (selected, cols 0-5) borders neighbour B (cols 6-14). A line drawn the
    # full height of B at col 10 splits B in two; only the piece touching A joins.
    seg = np.zeros((20, 20), dtype=np.uint16)
    seg[:, 0:6] = 1
    seg[:, 6:15] = 2
    bg_before = int(np.sum(seg == 0))

    ok = carve_into_selected(seg, [(0, 10), (19, 10)], selected_label=1)

    assert ok is True
    assert np.all(seg[:, 0:6] == 1)        # selected cell intact
    assert np.all(seg[:, 6:9] == 1)        # near half of B annexed into A
    assert np.all(seg[:, 11:15] == 2)      # far half of B survives as B
    assert sorted(np.unique(seg).tolist()) == [0, 1, 2]  # no label created/deleted
    assert int(np.sum(seg == 0)) == bg_before  # background never annexed


def test_carve_into_selected_merges_smaller_piece_when_both_touch():
    # A spans the full height alongside B, so both halves of a horizontal cut
    # touch A. The smaller (top) piece is annexed; B keeps its bulk.
    seg = np.zeros((20, 20), dtype=np.uint16)
    seg[:, 0:6] = 1
    seg[:, 6:15] = 2

    ok = carve_into_selected(seg, [(3, 6), (3, 14)], selected_label=1)

    assert ok is True
    assert np.all(seg[0:3, 6:15] == 1)     # smaller top piece annexed
    assert np.all(seg[5:20, 6:15] == 2)    # larger bottom piece stays B


def test_carve_into_selected_merges_iteratively_across_two_neighbours():
    # A (bottom-left block) touches B but not C; B sits between them. One line
    # cutting both B and C annexes B's lower half first, which then brings C's
    # lower half into contact with A so it is annexed on the next pass.
    seg = np.zeros((20, 20), dtype=np.uint16)
    seg[16:20, 0:5] = 1   # A
    seg[0:16, 0:5] = 2    # B (above A)
    seg[0:16, 5:10] = 3   # C (right of B, not touching A)

    ok = carve_into_selected(seg, [(8, 0), (8, 9)], selected_label=1)

    assert ok is True
    assert np.all(seg[9:16, 0:5] == 1)     # B's near half annexed
    assert np.all(seg[9:16, 5:10] == 1)    # C's near half annexed only after B's
    assert np.all(seg[0:8, 0:5] == 2)      # B keeps its far half
    assert np.all(seg[0:8, 5:10] == 3)     # C keeps its far half
    assert sorted(np.unique(seg).tolist()) == [0, 1, 2, 3]


def test_carve_into_selected_trims_selected_cell_dropping_smaller_piece():
    # A line drawn clean through the selected cell itself splits it and removes
    # the smaller piece to background; the larger piece stays as the cell.
    seg = np.zeros((20, 20), dtype=np.uint16)
    seg[:, 0:15] = 1  # selected cell spans cols 0-14

    ok = carve_into_selected(seg, [(3, 0), (3, 14)], selected_label=1)

    assert ok is True
    assert np.all(seg[0:3, 0:15] == 0)     # smaller top piece dropped to bg
    assert np.all(seg[5:20, 0:15] == 1)    # larger bottom piece stays selected
    assert sorted(np.unique(seg).tolist()) == [0, 1]


def test_carve_into_selected_trim_takes_priority_over_neighbour_cut():
    # A line crossing both the selected cell and a neighbour only trims the
    # selection; the neighbour is left completely untouched.
    seg = np.zeros((20, 20), dtype=np.uint16)
    seg[:, 0:7] = 1   # selected cell
    seg[:, 7:15] = 2  # neighbour
    neighbour_before = seg[:, 7:15].copy()

    ok = carve_into_selected(seg, [(3, 0), (3, 14)], selected_label=1)

    assert ok is True
    assert np.all(seg[0:3, 0:7] == 0)              # smaller piece of selection dropped
    assert np.all(seg[5:20, 0:7] == 1)             # larger piece kept
    assert np.array_equal(seg[:, 7:15], neighbour_before)  # neighbour untouched


def test_carve_into_selected_keeps_selected_cell_a_line_only_clips():
    # A stroke that enters the selected cell without crossing it leaves it whole.
    seg = np.zeros((20, 20), dtype=np.uint16)
    seg[:, 0:15] = 1
    before = seg.copy()

    ok = carve_into_selected(seg, [(5, 0), (5, 7)], selected_label=1)

    assert ok is False
    assert np.array_equal(seg, before)


def test_carve_into_selected_ignores_line_that_only_clips_a_neighbour():
    # A short stroke that enters B but never runs through it leaves B whole.
    seg = np.zeros((20, 20), dtype=np.uint16)
    seg[:, 0:6] = 1
    seg[:, 6:15] = 2
    before = seg.copy()

    ok = carve_into_selected(seg, [(5, 6), (5, 9)], selected_label=1)

    assert ok is False
    assert np.array_equal(seg, before)


def test_carve_into_selected_ignores_line_over_background():
    # A swipe that crosses no other cell is a no-op.
    seg = np.zeros((20, 20), dtype=np.uint16)
    seg[:, 0:6] = 1
    seg[:, 6:10] = 2  # cols 10-19 are background
    before = seg.copy()

    ok = carve_into_selected(seg, [(5, 17), (14, 17)], selected_label=1)

    assert ok is False
    assert np.array_equal(seg, before)


def test_carve_into_selected_requires_present_selection():
    seg = np.zeros((20, 20), dtype=np.uint16)
    seg[:, 0:5] = 2  # only a neighbour, no selected label present
    before = seg.copy()

    assert carve_into_selected(
        seg, [(0, 3), (19, 3)], selected_label=1,
    ) is False
    assert np.array_equal(seg, before)


def test_clean_stranded_pixels_cleans_fragments_without_filling_background_holes():
    seg = np.ones((8, 8), dtype=np.uint32)
    seg[2:6, 2:6] = 2
    seg[3, 3] = 0
    seg[0, 7] = 2

    changed = clean_stranded_pixels(seg, min_size=4)

    assert changed == 1
    assert seg[3, 3] == 0
    assert seg[0, 7] != 2


def test_fill_label_holes_fills_hole_enclosed_by_single_cell():
    seg = np.zeros((12, 12), dtype=np.uint32)
    seg[2:10, 2:10] = 1
    seg[5:7, 5:7] = 0  # background hole fully inside cell 1

    result = fill_label_holes(seg, radius=999)

    assert np.all(result[5:7, 5:7] == 1)
    # surrounding background outside the cell is untouched
    assert result[0, 0] == 0


def test_fill_label_holes_leaves_gap_between_two_cells_unchanged():
    # The enclosed background here borders both cell 1 and cell 2, so it is an
    # inter-cellular gap, not a hole within one cell: it must NOT be filled
    # (no expanding cells into the space between them).
    seg = np.zeros((12, 12), dtype=np.uint32)
    seg[1:11, 1:4] = 1
    seg[1:11, 8:11] = 2
    seg[1:3, 4:8] = 1
    seg[9:11, 4:8] = 2
    seg[3:9, 4:8] = 0
    before = seg.copy()

    result = fill_label_holes(seg, radius=999)

    np.testing.assert_array_equal(result, before)


def test_fill_label_holes_respects_radius_depth_guard():
    seg = np.zeros((16, 16), dtype=np.uint32)
    seg[2:14, 2:14] = 1
    seg[5:11, 5:11] = 0  # 6x6 hole: deepest pixel is 3 px from the wall

    # Too shallow a radius leaves the hole untouched...
    shallow = fill_label_holes(seg.copy(), radius=2)
    assert np.any(shallow[5:11, 5:11] == 0)
    # ...a large enough radius fills it completely.
    deep = fill_label_holes(seg.copy(), radius=3)
    assert np.all(deep[5:11, 5:11] == 1)


def test_fill_label_holes_zero_radius_is_noop():
    seg = np.ones((6, 6), dtype=np.uint32)
    seg[2:4, 2:4] = 0

    result = fill_label_holes(seg, radius=0)

    np.testing.assert_array_equal(result, seg)


def test_fill_label_holes_leaves_open_background_unchanged():
    seg = np.ones((8, 8), dtype=np.uint32)
    seg[2:6, 2:6] = 0
    seg[0:3, 3] = 0

    result = fill_label_holes(seg, radius=999)

    np.testing.assert_array_equal(result, seg)


def test_fix_label_semiholes_repairs_narrow_border_connected_gap():
    seg = np.ones((8, 8), dtype=np.uint32)
    seg[1:7, 1:7] = 2
    seg[2:6, 2:6] = 0
    seg[0, 3] = 0
    seg[1, 3] = 0

    result = fix_label_semiholes(seg, radius=999, max_opening=1)

    assert not np.any(result[2:6, 2:6] == 0)
    assert result[0, 3] != 0


def test_fix_label_semiholes_leaves_wide_border_opening_unchanged():
    seg = np.ones((8, 8), dtype=np.uint32)
    seg[1:7, 1:7] = 2
    seg[2:6, 2:6] = 0
    seg[0, 2:5] = 0
    seg[1, 3] = 0

    result = fix_label_semiholes(seg, radius=999, max_opening=1)

    np.testing.assert_array_equal(result, seg)


def test_fix_label_semiholes_leaves_gap_between_two_cells_unchanged():
    # A narrow border opening, but the gap is bordered by both cell 1 and cell 2
    # -> inter-cellular gap, not a hole within one cell. Must stay background
    # (the old expand-based code wrongly filled it from both sides).
    seg = np.zeros((8, 8), dtype=np.uint32)
    seg[:, 0:3] = 1
    seg[:, 4:8] = 2
    seg[1:7, 3] = 0   # vertical gap between the cells
    seg[0, 3] = 0     # narrow (1 px) opening to the top border
    seg[7, 3] = 2     # closed at the bottom
    before = seg.copy()

    result = fix_label_semiholes(seg, radius=999, max_opening=1)

    np.testing.assert_array_equal(result, before)


def test_fix_label_semiholes_zero_radius_or_opening_is_noop():
    seg = np.ones((8, 8), dtype=np.uint32)
    seg[1:7, 1:7] = 2
    seg[2:6, 2:6] = 0
    seg[0, 3] = 0
    seg[1, 3] = 0

    np.testing.assert_array_equal(
        fix_label_semiholes(seg, radius=0, max_opening=1),
        seg,
    )
    np.testing.assert_array_equal(
        fix_label_semiholes(seg, radius=999, max_opening=0),
        seg,
    )


def test_expand_label_to_foreground_adds_background_foreground_pixels():
    seg = np.zeros((7, 7), dtype=np.uint32)
    seg[3, 3] = 4
    foreground = np.zeros_like(seg, dtype=np.uint8)
    foreground[2:5, 2:5] = 1

    added = expand_label_to_foreground(seg, foreground, 4, max_distance=1)

    assert added == 4
    assert seg[3, 3] == 4
    assert seg[2, 3] == 4
    assert seg[3, 2] == 4
    assert seg[3, 4] == 4
    assert seg[4, 3] == 4
    assert seg[2, 2] == 0


def test_expand_label_to_foreground_respects_max_distance():
    seg = np.zeros((9, 9), dtype=np.uint32)
    seg[4, 4] = 2
    foreground = np.zeros_like(seg, dtype=np.uint8)
    foreground[1:8, 1:8] = 1

    added = expand_label_to_foreground(seg, foreground, 2, max_distance=2)

    assert added == 12
    assert seg[4, 6] == 2
    assert seg[4, 7] == 0


def test_expand_label_to_foreground_does_not_overwrite_neighboring_labels():
    seg = np.zeros((5, 7), dtype=np.uint32)
    seg[2, 2] = 1
    seg[2, 4] = 9
    foreground = np.ones_like(seg, dtype=np.uint8)

    expand_label_to_foreground(seg, foreground, 1, max_distance=3)

    assert seg[2, 4] == 9


def test_expand_label_to_foreground_does_not_cross_foreground_gaps():
    seg = np.zeros((5, 9), dtype=np.uint32)
    seg[2, 1] = 3
    foreground = np.zeros_like(seg, dtype=np.uint8)
    foreground[1:4, 0:3] = 1
    foreground[1:4, 5:8] = 1

    expand_label_to_foreground(seg, foreground, 3, max_distance=0)

    assert np.all(seg[1:4, 0:3][foreground[1:4, 0:3] > 0] == 3)
    assert not np.any(seg[1:4, 5:8] == 3)


def test_expand_label_to_foreground_zero_distance_is_unlimited():
    seg = np.zeros((5, 5), dtype=np.uint32)
    seg[2, 2] = 8
    foreground = np.ones_like(seg, dtype=np.uint8)

    added = expand_label_to_foreground(seg, foreground, 8, max_distance=0)

    assert added == 24
    assert np.all(seg == 8)


def test_expand_label_to_foreground_raises_on_shape_mismatch():
    seg = np.zeros((5, 5), dtype=np.uint32)
    foreground = np.zeros((5, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="same shape"):
        expand_label_to_foreground(seg, foreground, 1, max_distance=1)


def test_expand_label_to_foreground_returns_zero_without_mutation_when_absent_or_disconnected():
    seg = np.zeros((6, 6), dtype=np.uint32)
    seg[1, 1] = 6
    foreground = np.zeros_like(seg, dtype=np.uint8)
    foreground[4:6, 4:6] = 1
    before = seg.copy()

    assert expand_label_to_foreground(seg, foreground, 7, max_distance=0) == 0
    np.testing.assert_array_equal(seg, before)
    assert expand_label_to_foreground(seg, foreground, 6, max_distance=0) == 0
    np.testing.assert_array_equal(seg, before)


def test_add_cell_stamps_disk_on_empty_space_without_image():
    seg = np.zeros((40, 40), dtype=np.uint32)
    assert add_cell(seg, (20, 20), new_label=7, radius=5)
    assert seg[20, 20] == 7
    # disk(5) covers 81 px
    assert int((seg == 7).sum()) == 81


def test_add_cell_rejects_click_on_existing_cell():
    seg = np.zeros((40, 40), dtype=np.uint32)
    seg[10:15, 10:15] = 3
    before = seg.copy()
    assert not add_cell(seg, (12, 12), new_label=9, radius=4)
    np.testing.assert_array_equal(seg, before)


def test_add_cell_never_overwrites_neighbours_or_protected():
    seg = np.zeros((40, 40), dtype=np.uint32)
    seg[18:23, 18:23] = 4
    prot = np.zeros((40, 40), dtype=bool)
    prot[:, :15] = True
    assert add_cell(seg, (25, 25), new_label=8, radius=8, protected_mask=prot)
    assert int((seg == 4).sum()) == 25  # neighbour intact
    assert int(((seg == 8) & (seg == 4)).sum()) == 0
    assert int(((seg == 8) & prot).sum()) == 0


def test_add_cell_snaps_to_nucleus_signal_when_present():
    img = np.zeros((40, 40), dtype=np.float32)
    yy, xx = np.ogrid[:40, :40]
    img[(yy - 20) ** 2 + (xx - 20) ** 2 <= 16] = 100.0  # radius-4 bright blob
    seg = np.zeros((40, 40), dtype=np.uint32)
    assert add_cell(seg, (20, 20), new_label=5, radius=10, image=img)
    # Snapped to the ~49 px blob, not the 317 px disk that radius=10 would stamp.
    assert int((seg == 5).sum()) < 100
    assert seg[20, 20] == 5


def test_add_cell_snaps_to_large_nucleus_by_growing_window():
    # A nucleus far larger than the initial radius*2.5 window: snapping should
    # grow the window until the whole blob is enclosed instead of falling back
    # to a tiny disk.
    img = np.zeros((120, 120), dtype=np.float32)
    yy, xx = np.ogrid[:120, :120]
    big = (yy - 60) ** 2 + (xx - 60) ** 2 <= 30 ** 2  # radius-30 blob
    img[big] = 100.0
    seg = np.zeros((120, 120), dtype=np.uint32)
    assert add_cell(seg, (60, 60), new_label=5, radius=4, image=img)
    snapped = int((seg == 5).sum())
    # Close to the ~2827 px blob, far more than the disk(4) = 49 px fallback.
    assert snapped > 2000
    assert seg[60, 60] == 5


def test_add_cell_does_not_snap_to_bright_background_sheet():
    # A bright region covering almost the whole frame is a sheet, not one
    # nucleus: once the grown window encloses it, the "mostly foreground" guard
    # rejects it and we fall back to the plain disk.
    img = np.full((120, 120), 100.0, dtype=np.float32)
    img[:3, :] = img[-3:, :] = img[:, :3] = img[:, -3:] = 0.0  # thin dark border
    seg = np.zeros((120, 120), dtype=np.uint32)
    assert add_cell(seg, (60, 60), new_label=7, radius=5, image=img)
    assert int((seg == 7).sum()) == 81  # disk(5) fallback, not a huge blob


def test_add_cell_falls_back_to_disk_when_signal_is_flat():
    flat = np.full((40, 40), 50.0, dtype=np.float32)
    seg = np.zeros((40, 40), dtype=np.uint32)
    assert add_cell(seg, (20, 20), new_label=6, radius=5, image=flat)
    assert int((seg == 6).sum()) == 81  # disk fallback


def test_add_cell_yields_single_connected_region_when_stamp_is_fragmented():
    # A neighbouring cell forms a full-height wall that slices the radius-8 disk
    # into a main body (under the click) and a thin sliver on the far side.
    # The spawned cell must keep only the click's contiguous region — the sliver
    # is dropped, never written as a stray disjoint fragment.
    from skimage.measure import label as _cc_label

    seg = np.zeros((40, 40), dtype=np.uint32)
    seg[:, 14:16] = 3  # vertical wall separating cols <14 from the click side
    assert add_cell(seg, (20, 20), new_label=8, radius=8)
    region = seg == 8
    assert region[20, 20]
    # Exactly one 8-connected component, all on the click side of the wall.
    assert _cc_label(region, return_num=True, connectivity=2)[1] == 1
    assert int((region & (np.arange(40) < 14)[None, :]).sum()) == 0
    assert int((seg == 3).sum()) == 40 * 2  # wall untouched


def test_add_cell_fills_interior_hole_of_snapped_ring_nucleus():
    # A ring-shaped bright nucleus (dark hole in the middle). Clicking on the
    # bright ring snaps to the annulus; the enclosed background hole is filled so
    # the spawned cell is a solid, fully connected region — no donut.
    img = np.zeros((60, 60), dtype=np.float32)
    yy, xx = np.ogrid[:60, :60]
    dist2 = (yy - 30) ** 2 + (xx - 30) ** 2
    img[(dist2 >= 6 ** 2) & (dist2 <= 11 ** 2)] = 100.0  # bright annulus
    seg = np.zeros((60, 60), dtype=np.uint32)
    # Click on the bright ring (dist ~8 from centre), not the dark hole.
    assert add_cell(seg, (30, 38), new_label=9, radius=4, image=img)
    region = seg == 9
    assert region[30, 38]
    assert region[30, 30]  # the dark centre hole is filled in
    from skimage.measure import label as _cc_label

    assert _cc_label(region, return_num=True, connectivity=2)[1] == 1
