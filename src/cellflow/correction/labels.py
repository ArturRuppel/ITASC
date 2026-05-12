"""Label correction operations on a single (H, W) segmentation frame.

All functions accept a 2-D ``seg`` array and modify it **in-place**.
They return ``True`` on success and ``False`` when the operation is
rejected (e.g. labels don't touch, result too small, background click).
"""
from __future__ import annotations

import logging
import os

import numpy as np
from scipy.ndimage import binary_dilation, binary_closing, binary_fill_holes, label as nd_label
from scipy.ndimage import distance_transform_edt
from skimage.draw import polygon as draw_polygon
from skimage.morphology import disk
from skimage.segmentation import watershed, expand_labels

log = logging.getLogger("cellflow.correction")
if os.environ.get("CELLFLOW_DEBUG"):
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("[cellflow.correction] %(levelname)s %(message)s"))
        log.addHandler(_h)

MIN_CELL_SIZE: int = 4


# ── bounding-box helpers ──────────────────────────────────────────────────────

def _bbox_of_label(seg: np.ndarray, lab: int) -> tuple[int, int, int, int]:
    rows, cols = np.where(seg == lab)
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def _bbox_of_two(seg: np.ndarray, la: int, lb: int) -> tuple[int, int, int, int]:
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
    r0, c0 = bbox[0], bbox[1]
    return [(float(p[-2]) - r0, float(p[-1]) - c0) for p in pts]


# ── line drawing ──────────────────────────────────────────────────────────────

def _interpolate(pts: list[tuple[float, float]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i in range(len(pts) - 1):
        r0, c0 = pts[i]
        r1, c1 = pts[i + 1]
        n = max(abs(int(r1) - int(r0)), abs(int(c1) - int(c0)), 1)
        for t in np.linspace(0, 1, n + 1):
            out.append((int(round(r0 + t * (r1 - r0))), int(round(c0 + t * (c1 - c0)))))
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
    dilated_a = binary_dilation(seg == la, disk(1))
    dilated_b = binary_dilation(seg == lb, disk(1))
    return bool(np.any(dilated_a & dilated_b))


def _label_at(seg: np.ndarray, pos: tuple) -> int:
    r, c = int(round(float(pos[-2]))), int(round(float(pos[-1])))
    r = max(0, min(r, seg.shape[0] - 1))
    c = max(0, min(c, seg.shape[1] - 1))
    return int(seg[r, c])


def frame_view_2d(arr: np.ndarray, t: int) -> np.ndarray | None:
    """Return a 2D frame view from a time-indexed label stack."""
    if arr.ndim < 3 or t < 0 or t >= arr.shape[0]:
        return None
    view = arr[t]
    while view.ndim > 2:
        if view.shape[0] != 1:
            return None
        view = view[0]
    return view


def best_overlapping_label(
    target_labels: np.ndarray,
    source_labels: np.ndarray,
    t: int,
    source_label: int,
) -> int:
    """Return the non-zero target label with most overlap against source_label."""
    if source_label == 0:
        return 0
    target_frame = frame_view_2d(target_labels, t)
    source_frame = frame_view_2d(source_labels, t)
    if target_frame is None or source_frame is None or target_frame.shape != source_frame.shape:
        return 0
    source_mask = source_frame == int(source_label)
    if not np.any(source_mask):
        return 0
    overlap_values, counts = np.unique(target_frame[source_mask], return_counts=True)
    best_label = 0
    best_count = 0
    for label, count in zip(overlap_values, counts, strict=True):
        label = int(label)
        if label != 0 and int(count) > best_count:
            best_label = label
            best_count = int(count)
    return best_label


# ── public operations ─────────────────────────────────────────────────────────

def expand_label_to_foreground(
    seg: np.ndarray,
    foreground: np.ndarray,
    label: int,
    *,
    max_distance: int,
) -> int:
    """Expand ``label`` into connected foreground background pixels in-place.

    Returns the number of newly labelled pixels. A ``max_distance`` of 0 means
    no distance cap.
    """
    if foreground.shape != seg.shape:
        raise ValueError("foreground and seg must have the same shape")

    label = int(label)
    if label == 0:
        return 0
    seed = seg == label
    if not np.any(seed):
        return 0

    allowed = (foreground > 0) & ((seg == 0) | seed)
    component_labels, _num_components = nd_label(
        allowed,
        structure=np.ones((3, 3), dtype=np.uint8),
    )
    touching_ids = np.unique(component_labels[seed])
    touching_ids = touching_ids[touching_ids != 0]
    if touching_ids.size == 0:
        return 0

    touching_component = np.isin(component_labels, touching_ids)
    if max_distance > 0:
        dist = distance_transform_edt(~seed)
        touching_component &= dist <= int(max_distance)

    added = touching_component & (seg == 0)
    n_added = int(np.count_nonzero(added))
    if n_added:
        seg[added] = label
    return n_added

def erase_cell(seg: np.ndarray, pos: tuple | None = None, *, label: int | None = None) -> bool:
    """Set all pixels of the label under *pos* (or *label*) to 0."""
    if label is None:
        if pos is None:
            return False
        label = _label_at(seg, pos)
    log.debug("erase_cell: label=%s pos=%s", label, pos)
    if label == 0:
        return False
    seg[seg == label] = 0
    return True


def merge_cells(
    seg: np.ndarray,
    pos_start: tuple,
    pos_end: tuple,
    *,
    label_a: int | None = None,
    label_b: int | None = None,
) -> bool:
    """Merge the cell at *pos_start* into the cell at *pos_end*."""
    la = label_a if label_a is not None else _label_at(seg, pos_start)
    lb = label_b if label_b is not None else _label_at(seg, pos_end)
    log.debug("merge_cells: la=%s lb=%s", la, lb)
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
    other_cells = (crop != 0) & ~combined
    closed = closed & ~other_cells
    seg[r0:r1, c0:c1][closed] = lb

    remaining_la = seg == la
    if remaining_la.any():
        seg[remaining_la] = lb

    clean_stranded_pixels(seg)
    return True


def split_across(
    seg: np.ndarray,
    img: np.ndarray | None,
    pos_start: tuple,
    pos_end: tuple,
    *,
    new_label: int | None = None,
) -> bool:
    """Watershed-split the cell under *pos_start* using two seeds."""
    la = _label_at(seg, pos_start)
    lb = _label_at(seg, pos_end)
    log.debug("split_across: la=%s lb=%s", la, lb)
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

    new_lab = int(new_label) if new_label is not None else _free_label(seg)

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

        size_a = int(np.sum(ws == la))
        size_b = int(np.sum(ws == new_lab))
        if size_a >= MIN_CELL_SIZE and size_b >= MIN_CELL_SIZE:
            seg[r0:r1, c0:c1][ws == new_lab] = new_lab
            return True

    return False


def split_draw(
    seg: np.ndarray,
    positions: list,
    *,
    curlabel: int | None = None,
    new_label: int | None = None,
) -> bool:
    """Split a cell along a manually drawn line."""
    log.debug("split_draw: %d raw positions, curlabel=%s", len(positions), curlabel)
    if curlabel is None or curlabel == 0 or not np.any(seg == curlabel):
        return False

    bbox = _bbox_of_label(seg, curlabel)
    bbox = _extend_bbox(bbox, 1.25, seg.shape)
    crop = _crop(seg, bbox).copy()
    local_pts = _to_local(positions, bbox)

    in_cell_indices = [
        i for i, p in enumerate(local_pts)
        if 0 <= int(round(p[0])) < crop.shape[0]
        and 0 <= int(round(p[1])) < crop.shape[1]
        and crop[int(round(p[0])), int(round(p[1]))] == curlabel
    ]
    if len(in_cell_indices) < 2:
        return False

    first_idx, last_idx = in_cell_indices[0], in_cell_indices[-1]
    in_cell = [local_pts[i] for i in in_cell_indices]

    ext_start = local_pts[first_idx - 1] if first_idx > 0 else in_cell[0]
    ext_end   = local_pts[last_idx + 1]  if last_idx < len(local_pts) - 1 else in_cell[-1]

    all_pts = [ext_start] + in_cell + [ext_end]
    interp = _interpolate(all_pts)
    line = _draw_line(crop.shape, interp)

    if int(np.sum(line & (crop == curlabel))) == 0:
        return False

    return _split_in_crop(seg, crop, line, bbox, curlabel, new_label=new_label)


def _split_in_crop(
    seg: np.ndarray,
    crop: np.ndarray,
    line: np.ndarray,
    bbox: tuple,
    curlabel: int,
    retry: int = 0,
    *,
    new_label: int | None = None,
) -> bool:
    if retry > 6:
        return False

    dilated = binary_dilation(line, disk(retry)) if retry > 0 else line.astype(bool)
    mask = np.zeros(crop.shape, dtype=np.uint8)
    mask[crop == curlabel] = 1
    mask[dilated] = 0

    regions, n = nd_label(mask)
    sizes = [int(np.sum(regions == i)) for i in range(1, n + 1)]
    log.debug("_split_in_crop: retry=%d n_regions=%d sizes=%s", retry, n, sizes)

    if n >= 2:
        ids_by_size = sorted(range(1, n + 1), key=lambda i: sizes[i - 1], reverse=True)
        id_a, id_b = ids_by_size[0], ids_by_size[1]
        size_a, size_b = sizes[id_a - 1], sizes[id_b - 1]
        if size_a >= MIN_CELL_SIZE and size_b >= MIN_CELL_SIZE:
            regions_2 = np.zeros_like(regions)
            regions_2[regions == id_a] = 1
            regions_2[regions == id_b] = 2
            expanded = expand_labels(regions_2, distance=max(retry + 2, 3))
            r0, c0, r1, c1 = bbox
            new_lab = int(new_label) if new_label is not None else _free_label(seg)
            orig_cell = crop == curlabel
            seg[r0:r1, c0:c1][(expanded == 2) & orig_cell] = new_lab
            return True

    return _split_in_crop(seg, crop, line, bbox, curlabel, retry + 1, new_label=new_label)


def draw_cell_path(
    seg: np.ndarray,
    positions: list,
    *,
    curlabel: int | None = None,
    new_label: int | None = None,
) -> bool:
    """Draw a closed region from the user's stroke and fill its interior."""
    log.debug("draw_cell_path: %d raw positions, curlabel=%s", len(positions), curlabel)
    if len(positions) < 2:
        return False

    local_pts = [(float(p[-2]), float(p[-1])) for p in positions]

    rows = np.array([p[0] for p in local_pts])
    cols = np.array([p[1] for p in local_pts])
    rr, cc = draw_polygon(rows, cols, seg.shape)
    log.debug("draw_cell_path: polygon fill pixels=%d", len(rr))

    if len(rr) < MIN_CELL_SIZE:
        return False

    extending = bool(curlabel) and curlabel != 0 and np.any(seg == curlabel)
    label = curlabel if extending else (
        int(new_label) if new_label is not None else _free_label(seg)
    )

    fill_mask = np.zeros(seg.shape, dtype=bool)
    fill_mask[rr, cc] = True
    if extending:
        existing_mask = seg == label
        connected_regions, _ = nd_label(existing_mask | fill_mask)
        connected_ids = np.unique(connected_regions[existing_mask])
        fill_mask &= np.isin(connected_regions, connected_ids)
    else:
        fill_mask &= (seg == 0)

    n_px = int(np.sum(fill_mask))
    if n_px < MIN_CELL_SIZE:
        return False

    seg[fill_mask] = label
    if extending:
        cell_mask = seg == label
        filled_mask = binary_fill_holes(cell_mask)
        seg[filled_mask & ~cell_mask] = label
    return True


def swap_labels(seg: np.ndarray, pos_a: tuple, pos_b: tuple) -> bool:
    """Swap the label values at the two click positions across the whole frame."""
    la = _label_at(seg, pos_a)
    lb = _label_at(seg, pos_b)
    log.debug("swap_labels: la=%s lb=%s", la, lb)
    if la == 0 or lb == 0 or la == lb:
        return False
    mask_a = seg == la
    mask_b = seg == lb
    seg[mask_a] = lb
    seg[mask_b] = la
    return True


def relabel_cell(seg: np.ndarray, pos: tuple, new_label: int) -> bool:
    """Assign *new_label* to the cell at *pos* in *seg* (in-place).

    If *new_label* already exists in the frame, the two cells are swapped so
    no label is lost.  Returns ``False`` when *pos* hits background, already
    has *new_label*, or *new_label* is 0.
    """
    old_label = _label_at(seg, pos)
    if old_label == 0 or new_label == 0 or old_label == new_label:
        return False
    conflict = seg == new_label
    seg[seg == old_label] = new_label
    if np.any(conflict):
        seg[conflict] = old_label
    return True


def fill_label_holes(labels: np.ndarray, radius: int = 5) -> np.ndarray:
    """Fill holes fully enclosed within individual labels.

    Only background pixels completely surrounded by a *single* label are
    filled — label boundaries are never shifted.  ``radius`` limits the
    maximum hole size: holes with area > π·radius² are left open.
    """
    if radius <= 0:
        return labels

    max_area = int(np.pi * radius * radius)
    result = labels.copy()

    for cell_id in np.unique(labels):
        if cell_id == 0:
            continue
        mask = result == cell_id
        filled = binary_fill_holes(mask)
        holes = filled & ~mask
        if not np.any(holes):
            continue
        # never steal from another label
        holes &= result == 0
        if not np.any(holes):
            continue
        # keep only small-enough connected components
        hole_cc, n_cc = nd_label(holes)
        for h_id in range(1, n_cc + 1):
            comp = hole_cc == h_id
            if int(np.count_nonzero(comp)) <= max_area:
                result[comp] = cell_id

    return result


def fix_label_semiholes(
    labels: np.ndarray,
    radius: int = 5,
    max_opening: int = 3,
) -> np.ndarray:
    """Close narrow channels in label boundaries by bridging contour pinch-points.

    For each label, the ordered contour is scanned for pairs of points whose
    Euclidean distance is ≤ *max_opening* but whose arc-length along the
    contour is much larger (indicating a narrow indentation).  Qualifying
    pairs are connected with a 1-px-wide line and the resulting enclosed
    region is filled (up to area π·*radius*²).  Only background pixels are
    ever modified — existing label boundaries are untouched.
    """
    from scipy.spatial import cKDTree
    from skimage.draw import line as draw_line
    from skimage.measure import find_contours

    if max_opening <= 0 or radius <= 0:
        return labels

    max_area = int(np.pi * radius * radius)
    # contour-path distance must exceed this to qualify as a channel
    min_path = max(2 * max_opening, 6)
    result = labels.copy()
    unique_ids = [int(v) for v in np.unique(result) if v != 0]

    for cell_id in unique_ids:
        bbox = _bbox_of_label(result, cell_id)
        bbox = _extend_bbox(bbox, 1.5, result.shape, min_pad=max_opening + 2)
        r0, c0, r1, c1 = bbox
        crop = result[r0:r1, c0:c1]                       # writeable view
        mask = (crop == cell_id).astype(np.uint8)

        contours = find_contours(mask, 0.5)
        if not contours:
            continue

        bridged = False
        for contour in contours:
            pts = np.round(contour).astype(int)
            pts[:, 0] = np.clip(pts[:, 0], 0, crop.shape[0] - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, crop.shape[1] - 1)

            # deduplicate consecutive rounded pixels
            if len(pts) > 1:
                keep = np.ones(len(pts), dtype=bool)
                keep[1:] = np.any(pts[1:] != pts[:-1], axis=1)
                pts = pts[keep]
            n = len(pts)
            if n < min_path:
                continue

            tree = cKDTree(pts.astype(np.float64))
            pairs = tree.query_pairs(r=float(max_opening))

            for i, j in pairs:
                path_dist = min(abs(i - j), n - abs(i - j))
                if path_dist < min_path:
                    continue                               # just neighbours
                rr, cc = draw_line(
                    int(pts[i, 0]), int(pts[i, 1]),
                    int(pts[j, 0]), int(pts[j, 1]),
                )
                valid = (
                    (rr >= 0) & (rr < crop.shape[0])
                    & (cc >= 0) & (cc < crop.shape[1])
                )
                rr, cc = rr[valid], cc[valid]
                bg = crop[rr, cc] == 0
                if np.any(bg):
                    crop[rr[bg], cc[bg]] = cell_id
                    bridged = True

        # fill the pocket that the bridge just enclosed
        if bridged:
            new_mask = crop == cell_id
            filled = binary_fill_holes(new_mask)
            holes = filled & ~new_mask & (crop == 0)
            if np.any(holes):
                hole_cc, n_cc = nd_label(holes)
                for h_id in range(1, n_cc + 1):
                    comp = hole_cc == h_id
                    if int(np.count_nonzero(comp)) <= max_area:
                        crop[comp] = cell_id

    return result


def clean_stranded_pixels(seg: np.ndarray, min_size: int = MIN_CELL_SIZE) -> int:
    """Remove disconnected same-label fragments, keeping each label's largest component."""
    from skimage.measure import label as _cc_label
    cleared = 0

    for cell_id in np.unique(seg):
        if cell_id == 0:
            continue
        mask = seg == cell_id
        labeled, n_comp = _cc_label(mask, return_num=True, connectivity=2)
        if n_comp <= 1:
            continue
        comp_sizes = {cid: int(np.sum(labeled == cid)) for cid in range(1, n_comp + 1)}
        largest = max(comp_sizes, key=comp_sizes.__getitem__)
        for comp_id, n_px in comp_sizes.items():
            if comp_id == largest:
                continue
            comp_mask = labeled == comp_id
            seg[comp_mask] = 0
            filled = expand_labels(seg, distance=n_px + 2)
            seg[comp_mask] = filled[comp_mask]
            cleared += n_px

    return cleared


def cleanup_movie(
    cell_labels: np.ndarray,
    nuc_labels: np.ndarray,
    *,
    progress_cb=None,
) -> dict:
    """Clean and resynchronise a cell label movie against nuclear labels.

    Operations, in order:

    1. **Clean fragments** — for every frame, remove disconnected same-label
       components (keep the largest per label).
    2. **Resync IDs** — relabel each cell region to match the nucleus it
       overlaps most.  Nuclear labels are the source of truth.
    3. **Remove orphans** — erase cell regions that overlap no nucleus at all.

    Parameters
    ----------
    cell_labels : (T, H, W) ndarray
        Modified **in-place**.
    nuc_labels : (T, H, W) ndarray
        Read-only reference (nucleus tracked labels).
    progress_cb : callable, optional
        ``progress_cb(done: int, total: int, message: str)``

    Returns
    -------
    dict
        ``{"fragments_cleared": int, "cells_relabeled": int,
        "orphans_removed": int}``
    """
    if cell_labels.shape != nuc_labels.shape:
        raise ValueError(
            f"Shape mismatch: cell {cell_labels.shape} "
            f"vs nuclear {nuc_labels.shape}"
        )

    T = cell_labels.shape[0]
    total_steps = 2 * T
    stats = {
        "fragments_cleared": 0,
        "cells_relabeled": 0,
        "orphans_removed": 0,
    }

    # ── pass 1: clean fragments ───────────────────────────────────────────
    for t in range(T):
        seg = frame_view_2d(cell_labels, t)
        if seg is None:
            continue
        stats["fragments_cleared"] += clean_stranded_pixels(seg)
        if progress_cb:
            progress_cb(t + 1, total_steps, f"Cleaning fragments: {t + 1}/{T}")

    # ── pass 2: resync with nuclear labels ────────────────────────────────
    for t in range(T):
        cell_frame = frame_view_2d(cell_labels, t)
        nuc_frame = frame_view_2d(nuc_labels, t)
        if cell_frame is None or nuc_frame is None:
            if progress_cb:
                progress_cb(T + t + 1, total_steps, f"Resyncing: {t + 1}/{T}")
            continue

        cell_ids = set(int(v) for v in np.unique(cell_frame)) - {0}
        if not cell_ids:
            if progress_cb:
                progress_cb(T + t + 1, total_steps, f"Resyncing: {t + 1}/{T}")
            continue

        # map each cell → nucleus with largest overlap
        cell_to_nuc: dict[int, int] = {}
        for c in cell_ids:
            c_mask = cell_frame == c
            nuc_vals, counts = np.unique(nuc_frame[c_mask], return_counts=True)
            best_n, best_cnt = 0, 0
            for n, cnt in zip(nuc_vals, counts):
                n_int = int(n)
                if n_int != 0 and int(cnt) > best_cnt:
                    best_n, best_cnt = n_int, int(cnt)
            cell_to_nuc[c] = best_n

        # rebuild the frame in-place
        original = cell_frame.copy()
        cell_frame[:] = 0
        for c, n in cell_to_nuc.items():
            if n == 0:
                stats["orphans_removed"] += 1
            else:
                cell_frame[original == c] = n
                if c != n:
                    stats["cells_relabeled"] += 1

        if progress_cb:
            progress_cb(T + t + 1, total_steps, f"Resyncing: {t + 1}/{T}")

    return stats


from cellflow.segmentation import apply_gamma  # noqa: F401 — re-exported from here
