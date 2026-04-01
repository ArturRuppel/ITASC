"""
Label correction operations on a single (H, W) segmentation frame.

All functions accept a 2-D ``seg`` array and modify it **in-place**.
They return ``True`` on success and ``False`` when the operation is
rejected (e.g. labels don't touch, result too small, background click).

No CellFlow graph is modified — re-run graph extraction after corrections.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.ndimage import binary_dilation, binary_closing, label as nd_label
from scipy.ndimage import distance_transform_edt
from skimage.morphology import disk
from skimage.segmentation import watershed, find_boundaries, expand_labels

MIN_CELL_SIZE: int = 4


# ── bounding-box helpers ──────────────────────────────────────────────────────

def _bbox_of_label(seg: np.ndarray, lab: int) -> tuple[int, int, int, int]:
    """Return (r0, c0, r1, c1) tight around *lab*."""
    rows, cols = np.where(seg == lab)
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def _bbox_of_pts(pts: list) -> tuple[int, int, int, int]:
    """Return (r0, c0, r1, c1) tight around a list of (t,r,c) or (r,c) points."""
    arr = np.array(pts)
    rc = arr[:, -2:]
    return int(rc[:, 0].min()), int(rc[:, 1].min()), int(rc[:, 0].max()) + 1, int(rc[:, 1].max()) + 1


def _bbox_of_two(seg: np.ndarray, la: int, lb: int) -> tuple[int, int, int, int]:
    """Return (r0, c0, r1, c1) tight around both *la* and *lb*."""
    rows, cols = np.where(np.isin(seg, [la, lb]))
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def _extend_bbox(
    bbox: tuple[int, int, int, int],
    factor: float,
    shape: tuple[int, int],
    min_pad: int = 0,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    dr = max(int((r1 - r0) * (factor - 1) / 2), min_pad)
    dc = max(int((c1 - c0) * (factor - 1) / 2), min_pad)
    return (
        max(0, r0 - dr), max(0, c0 - dc),
        min(shape[0], r1 + dr), min(shape[1], c1 + dc),
    )


def _crop(arr: np.ndarray, bbox: tuple) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return arr[r0:r1, c0:c1]


def _to_local(pts: list, bbox: tuple) -> list[tuple[float, float]]:
    """Convert (t,r,c) or (r,c) positions to bbox-local (r,c)."""
    r0, c0 = bbox[0], bbox[1]
    return [(float(p[-2]) - r0, float(p[-1]) - c0) for p in pts]


# ── line drawing ──────────────────────────────────────────────────────────────

def _interpolate(pts: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """Return a dense list of integer (r, c) pixels between consecutive points."""
    out: list[tuple[int, int]] = []
    for i in range(len(pts) - 1):
        r0, c0 = pts[i]
        r1, c1 = pts[i + 1]
        n = max(abs(int(r1) - int(r0)), abs(int(c1) - int(c0)), 1)
        for t in np.linspace(0, 1, n + 1):
            out.append((int(round(r0 + t * (r1 - r0))), int(round(c0 + t * (c1 - c0)))))
    # deduplicate while preserving order
    seen: set = set()
    result = []
    for p in out:
        if p not in seen:
            seen.add(p)
            result.append(p)
    return result


def _draw_line(shape: tuple[int, int], pts: list[tuple[int, int]]) -> np.ndarray:
    line = np.zeros(shape, dtype=np.uint8)
    for r, c in pts:
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            line[r, c] = 1
    return line


# ── misc helpers ──────────────────────────────────────────────────────────────

def _free_label(seg: np.ndarray) -> int:
    return int(seg.max()) + 1



def _touches(seg: np.ndarray, la: int, lb: int) -> bool:
    """Check if labels are adjacent (within ~2 px, handles 0-valued boundaries)."""
    dilated_a = binary_dilation(seg == la, disk(1))
    dilated_b = binary_dilation(seg == lb, disk(1))
    return bool(np.any(dilated_a & dilated_b))


def _label_at(seg: np.ndarray, pos: tuple) -> int:
    r, c = int(round(float(pos[-2]))), int(round(float(pos[-1])))
    r = max(0, min(r, seg.shape[0] - 1))
    c = max(0, min(c, seg.shape[1] - 1))
    return int(seg[r, c])


# ── public operations ─────────────────────────────────────────────────────────

def erase_cell(seg: np.ndarray, pos: tuple | None = None, *, label: int | None = None) -> bool:
    """Set all pixels of the label under *pos* (or *label*) to 0."""
    if label is None:
        if pos is None:
            return False
        label = _label_at(seg, pos)
    if label == 0:
        return False
    seg[seg == label] = 0
    return True


def merge_cells(seg: np.ndarray, pos_start: tuple, pos_end: tuple) -> bool:
    """
    Merge the cell at *pos_start* into the cell at *pos_end*.

    The two labels must be touching; otherwise the operation is rejected
    with a return value of ``False``.
    """
    la = _label_at(seg, pos_start)
    lb = _label_at(seg, pos_end)
    if la == 0 or lb == 0 or la == lb:
        return False
    if not _touches(seg, la, lb):
        return False

    bbox = _bbox_of_two(seg, la, lb)
    bbox = _extend_bbox(bbox, 1.25, seg.shape)
    r0, c0, r1, c1 = bbox
    crop = _crop(seg, bbox)

    combined = np.isin(crop, [la, lb])
    closed = binary_closing(combined, disk(2))
    # prevent the closing from overwriting pixels belonging to other cells
    other_cells = (crop != 0) & ~combined
    closed = closed & ~other_cells
    seg[r0:r1, c0:c1][closed] = lb
    return True


def split_across(
    seg: np.ndarray,
    img: np.ndarray | None,
    pos_start: tuple,
    pos_end: tuple,
) -> bool:
    """
    Watershed-split the cell under *pos_start* using two seeds.

    *pos_start* and *pos_end* must lie on the same cell.
    *img* is the intensity image used by watershed; pass ``None`` to use a
    distance transform instead.
    """
    la = _label_at(seg, pos_start)
    lb = _label_at(seg, pos_end)
    if la == 0 or la != lb:
        return False

    bbox = _bbox_of_label(seg, la)
    bbox = _extend_bbox(bbox, 1.25, seg.shape)
    r0, c0, r1, c1 = bbox
    crop_seg = _crop(seg, bbox)
    mask = (crop_seg == la).astype(np.uint8)
    interior = mask.astype(bool)

    rs = max(0, min(int(round(float(pos_start[-2]))) - r0, mask.shape[0] - 1))
    cs = max(0, min(int(round(float(pos_start[-1]))) - c0, mask.shape[1] - 1))
    re = max(0, min(int(round(float(pos_end[-2]))) - r0, mask.shape[0] - 1))
    ce = max(0, min(int(round(float(pos_end[-1]))) - c0, mask.shape[1] - 1))

    new_lab = _free_label(seg)

    for radius in range(7):
        markers = np.zeros(mask.shape, dtype=np.int32)
        if radius == 0:
            markers[rs, cs] = la
            markers[re, ce] = new_lab
        else:
            d = disk(radius)
            seed_a = np.zeros(mask.shape, dtype=bool)
            seed_a[rs, cs] = True
            seed_b = np.zeros(mask.shape, dtype=bool)
            seed_b[re, ce] = True
            markers[binary_dilation(seed_a, d) & interior] = la
            markers[binary_dilation(seed_b, d) & interior] = new_lab

        if img is not None:
            crop_img = _crop(img, bbox)
            ws = watershed(crop_img, markers=markers, mask=mask)
        else:
            dist = distance_transform_edt(mask)
            ws = watershed(-dist, markers=markers, mask=mask)

        if np.sum(ws == la) >= MIN_CELL_SIZE and np.sum(ws == new_lab) >= MIN_CELL_SIZE:
            seg[r0:r1, c0:c1][ws == new_lab] = new_lab
            return True

    return False


def split_draw(seg: np.ndarray, positions: list, *, curlabel: int | None = None) -> bool:
    """
    Split a cell along a manually drawn line.

    *positions* is a list of (t,r,c) or (r,c) world coordinates collected
    during the mouse drag.

    Pass *curlabel* to target a specific cell (e.g. from a prior selection).
    When omitted the target cell is inferred from the labels under the drawn path.

    Existing labels are used as barriers (the line is only active within the
    target cell's pixels).  The bounding box is kept tight around the drawn
    line — only the target cell's own bounding box is used for the final split,
    so enclosed regions far from the line cannot become spurious new cells.
    """
    if curlabel is None or curlabel == 0 or not np.any(seg == curlabel):
        # identify the target cell from a tight crop around the drawn line
        tight_bbox = _bbox_of_pts(positions)
        tight_bbox = _extend_bbox(tight_bbox, 1.1, seg.shape)
        crop_tight = _crop(seg, tight_bbox)
        local_pts = _to_local(positions, tight_bbox)

        labels_under = [
            int(crop_tight[int(round(r)), int(round(c))])
            for r, c in local_pts
            if 0 <= int(round(r)) < crop_tight.shape[0]
            and 0 <= int(round(c)) < crop_tight.shape[1]
        ]
        if not labels_under:
            return False
        curlabel = max(set(labels_under), key=labels_under.count)
        if curlabel == 0:
            return False

    # re-crop around the cell itself so only that cell's region is in play
    bbox = _bbox_of_label(seg, curlabel)
    bbox = _extend_bbox(bbox, 1.25, seg.shape)
    crop = _crop(seg, bbox).copy()
    local_pts = _to_local(positions, bbox)

    interp = _interpolate(local_pts)
    line = _draw_line(crop.shape, interp)

    return _split_in_crop(seg, crop, line, bbox, curlabel)


def _split_in_crop(
    seg: np.ndarray,
    crop: np.ndarray,
    line: np.ndarray,
    bbox: tuple,
    curlabel: int,
    retry: int = 0,
) -> bool:
    if retry > 6:
        return False

    dilated = binary_dilation(line, disk(retry)) if retry > 0 else line.astype(bool)
    mask = np.zeros(crop.shape, dtype=np.uint8)
    mask[crop == curlabel] = 1
    mask[dilated] = 0

    regions, n = nd_label(mask)
    if (
        n == 2
        and np.sum(regions == 1) >= MIN_CELL_SIZE
        and np.sum(regions == 2) >= MIN_CELL_SIZE
    ):
        # Fill the gap left by the dilated line so labels are contiguous
        expanded = expand_labels(regions, distance=max(retry + 2, 3))
        r0, c0, r1, c1 = bbox
        new_lab = _free_label(seg)
        orig_cell = crop == curlabel
        seg[r0:r1, c0:c1][(expanded == 2) & orig_cell] = new_lab
        return True

    return _split_in_crop(seg, crop, line, bbox, curlabel, retry + 1)


def redraw_junction(seg: np.ndarray, positions: list) -> bool:
    """
    Redraw the boundary between two cells along a manually drawn line.

    The two cells are identified as the most common labels adjacent to the
    drawn path.  Their shared boundary is replaced by the new line.
    """
    tight_bbox = _bbox_of_pts(positions)
    tight_bbox = _extend_bbox(tight_bbox, 1.1, seg.shape, min_pad=4)
    crop_tight = _crop(seg, tight_bbox)
    local_pts = _to_local(positions, tight_bbox)

    # find cells adjacent to the drawn path by sampling labels in a neighbourhood
    labs_near: list[int] = []
    for r, c in local_pts:
        ri, ci = int(round(r)), int(round(c))
        region = crop_tight[max(0, ri - 2):ri + 3, max(0, ci - 2):ci + 3]
        labs_near.extend(int(v) for v in region.flat if v > 0)

    top = [lab for lab, _ in Counter(labs_near).most_common(2)]
    if len(top) < 2:
        return False
    lab_a, lab_b = top[0], top[1]

    bbox = _bbox_of_two(seg, lab_a, lab_b)
    bbox = _extend_bbox(bbox, 1.25, seg.shape)
    r0, c0, r1, c1 = bbox
    init_crop = _crop(seg, bbox).copy()
    local_pts = _to_local(positions, bbox)

    # temporarily merge both cells so the line can split them freely
    merged_bin = np.isin(init_crop, [lab_a, lab_b]).astype(np.uint8)
    merged = binary_closing(merged_bin, disk(2)).astype(np.uint8) * lab_a

    interp = _interpolate(local_pts)
    line = _draw_line(merged.shape, interp)

    return _move_junction(seg, init_crop, merged, line, bbox, lab_a, lab_b)


def _move_junction(
    seg: np.ndarray,
    init_crop: np.ndarray,
    merged: np.ndarray,
    line: np.ndarray,
    bbox: tuple,
    lab_a: int,
    lab_b: int,
    retry: int = 0,
) -> bool:
    if retry > 6:
        return False

    dilated = binary_dilation(line, disk(retry)) if retry > 0 else line.astype(bool)
    mask = (merged > 0).astype(np.uint8)
    mask[dilated] = 0

    regions, n = nd_label(mask)
    if (
        n == 2
        and np.sum(regions == 1) >= MIN_CELL_SIZE
        and np.sum(regions == 2) >= MIN_CELL_SIZE
    ):
        # Fill the gap left by the dilated line
        expanded = expand_labels(regions, distance=max(retry + 2, 3))
        r0, c0, r1, c1 = bbox
        orig_cells = np.isin(init_crop, [lab_a, lab_b])
        for reg_id in (1, 2):
            region_mask = (expanded == reg_id) & orig_cells
            vals, cnts = np.unique(init_crop[region_mask], return_counts=True)
            if len(vals):
                orig_lab = int(vals[np.argmax(cnts)])
                seg[r0:r1, c0:c1][region_mask] = orig_lab
        return True

    return _move_junction(seg, init_crop, merged, line, bbox, lab_a, lab_b, retry + 1)


def swap_labels(seg: np.ndarray, pos_a: tuple, pos_b: tuple) -> bool:
    """Swap the label values at the two click positions across the whole frame."""
    la = _label_at(seg, pos_a)
    lb = _label_at(seg, pos_b)
    if la == 0 or lb == 0 or la == lb:
        return False
    mask_a = seg == la
    mask_b = seg == lb
    seg[mask_a] = lb
    seg[mask_b] = la
    return True
