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

log = logging.getLogger("itasc.correction")
if os.environ.get("ITASC_DEBUG"):
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("[itasc.correction] %(levelname)s %(message)s"))
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
    """Return an unused label id (``max + 1``).

    The result is written back into ``seg``, so it must fit the array's
    dtype; for an integer ``seg`` whose max equals the dtype ceiling the
    increment would wrap to 0 (background) or collide with an existing id.
    Raise instead of corrupting the segmentation.
    """
    new_label = int(seg.max()) + 1
    if np.issubdtype(seg.dtype, np.integer) and new_label > np.iinfo(seg.dtype).max:
        raise OverflowError(
            f"No free label available: max id {new_label - 1} already at the "
            f"{seg.dtype} ceiling. Re-save the stack with a wider integer dtype."
        )
    return new_label


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


def _snap_cell_mask(
    image: np.ndarray,
    r: int,
    c: int,
    radius: int,
) -> np.ndarray | None:
    """Try to carve a single nucleus out of *image* around (r, c).

    Starts from a small window around the click and, whenever the blob spills
    past it, grows the window and retries — so a nucleus much larger than
    *radius* still snaps to its full contour instead of falling back to a tiny
    disk. The same Otsu + connected-component logic runs at every size; only the
    window grows. Returns a full-frame boolean mask, or ``None`` when the local
    signal is too weak/ambiguous to trust (flat window, no bright blob under the
    click, or a blob that never closes within the largest window — i.e.
    background, not one nucleus). The caller falls back to a plain disk then.
    """
    try:
        from skimage.filters import threshold_otsu

        h, w = image.shape
        win = max(int(radius * 2.5), 12)
        # A handful of geometric growth steps. Capped so a bright *background*
        # region — which never closes inside the window — can't snap to a giant
        # blob; it spills at every size and we fall through to ``None``.
        for _ in range(6):
            r0, c0 = max(0, r - win), max(0, c - win)
            r1, c1 = min(h, r + win + 1), min(w, c + win + 1)
            full = r0 == 0 and c0 == 0 and r1 == h and c1 == w
            crop = np.asarray(image[r0:r1, c0:c1], dtype=np.float64)
            if crop.size == 0:
                return None

            lo, hi = float(crop.min()), float(crop.max())
            # Flat / no contrast: the window may sit entirely inside the
            # nucleus — grow to reach its edge. If it already spans the whole
            # image there's nothing more to see, so give up.
            if hi - lo < 1e-6:
                if full:
                    return None
                win = int(win * 1.6) + 1
                continue

            thr = float(threshold_otsu(crop))
            fg = crop > thr
            lr, lc = r - r0, c - c0
            # The click itself must land on bright signal.
            if not fg[lr, lc]:
                return None

            labelled, _ = nd_label(fg)
            comp = labelled[lr, lc]
            if comp == 0:
                return None
            blob = labelled == comp

            # Does the blob run off an *interior* crop edge — i.e. spill into
            # image we didn't search? Edges that coincide with the image border
            # don't count; the nucleus genuinely ends there. While it spills,
            # grow and retry; once the window covers the whole image it can't
            # spill any further, so we accept what we have.
            spills = (
                (r0 > 0 and blob[0, :].any())
                or (r1 < h and blob[-1, :].any())
                or (c0 > 0 and blob[:, 0].any())
                or (c1 < w and blob[:, -1].any())
            )
            if spills:
                win = int(win * 1.6) + 1
                continue

            if blob.sum() < MIN_CELL_SIZE:
                return None
            # An enclosed blob filling almost the entire window is a bright
            # sheet, not one nucleus (a real nucleus leaves background margin
            # on the sides it doesn't touch).
            if fg.mean() > 0.85:
                return None

            mask = np.zeros(image.shape, dtype=bool)
            mask[r0:r1, c0:c1] = blob
            return mask
        return None
    except Exception:
        return None


def _keep_connected_region(
    mask: np.ndarray, r: int, c: int
) -> np.ndarray | None:
    """Reduce *mask* to a single, hole-free connected region.

    Picks the 8-connected component containing the click ``(r, c)`` — falling
    back to the largest component when the click pixel itself was masked out
    (e.g. it sits under a protected pixel) — then fills any interior holes.
    Returns ``None`` when *mask* is empty.
    """
    if not mask.any():
        return None
    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
    labelled, n_comp = nd_label(mask, structure=structure)
    if n_comp == 0:
        return None
    comp = int(labelled[r, c]) if mask[r, c] else 0
    if comp == 0:
        # Click pixel isn't part of the mask — keep the largest component.
        counts = np.bincount(labelled.ravel())
        counts[0] = 0
        comp = int(counts.argmax())
    region = labelled == comp
    return binary_fill_holes(region)


def add_cell(
    seg: np.ndarray,
    pos: tuple,
    *,
    new_label: int,
    radius: int = 6,
    image: np.ndarray | None = None,
    protected_mask: np.ndarray | None = None,
) -> bool:
    """Spawn a new cell at *pos* in empty space.

    When *image* (a nucleus/intensity frame the same shape as *seg*) carries a
    clear local signal the new cell snaps to it; otherwise a disk of *radius* is
    stamped. Only background pixels are written — existing cells and any
    *protected_mask* pixels are never overwritten — so the result can't clobber
    neighbours. Returns ``False`` when *pos* is on an existing cell or nothing
    paintable remains.
    """
    r, c = int(round(float(pos[-2]))), int(round(float(pos[-1])))
    if not (0 <= r < seg.shape[0] and 0 <= c < seg.shape[1]):
        return False
    if seg[r, c] != 0:
        return False
    if new_label <= 0:
        return False

    mask = None
    if image is not None and np.asarray(image).shape == seg.shape:
        mask = _snap_cell_mask(np.asarray(image), r, c, max(1, radius))
    if mask is None:
        rad = max(1, radius)
        d = disk(rad)
        size = 2 * rad + 1
        rr0, cc0 = r - rad, c - rad
        sr0, sc0 = max(0, rr0), max(0, cc0)
        sr1 = min(seg.shape[0], rr0 + size)
        sc1 = min(seg.shape[1], cc0 + size)
        mask = np.zeros(seg.shape, dtype=bool)
        mask[sr0:sr1, sc0:sc1] = d[
            sr0 - rr0 : sr0 - rr0 + (sr1 - sr0),
            sc0 - cc0 : sc0 - cc0 + (sc1 - sc0),
        ].astype(bool)

    # Never overwrite existing cells or protected pixels.
    allowed = seg == 0
    if protected_mask is not None:
        allowed &= ~protected_mask.astype(bool)
    mask &= allowed
    if mask.sum() < MIN_CELL_SIZE:
        return False

    # Intersecting with `allowed` can fracture the stamped blob into several
    # disconnected pieces or shed stray pixels (e.g. an existing cell or a
    # protected region cuts through it). Keep only the single connected
    # component under the click so the spawned cell is always one contiguous
    # region, never scattered fragments.
    mask = _keep_connected_region(mask, r, c)
    if mask is None:
        return False
    # Re-apply `allowed`: hole-filling in the helper may have spanned an
    # enclosed existing/protected pixel, which must still not be overwritten.
    # Removing those interior pixels leaves the surrounding region connected.
    mask &= allowed
    if mask.sum() < MIN_CELL_SIZE:
        return False

    log.debug("add_cell: label=%s pos=(%d,%d) px=%d", new_label, r, c, int(mask.sum()))
    seg[mask] = new_label
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
        # Cells don't touch: just give them the same id without painting any
        # new pixels. The shared label is left with disconnected components,
        # which is intentional — merge joins ids, it does not tidy geometry.
        seg[seg == la] = lb
        return True

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


def _mask_touches(a: np.ndarray, b: np.ndarray) -> bool:
    """True if boolean masks *a* and *b* are adjacent (8-connectivity)."""
    return bool(np.any(binary_dilation(a, disk(1)) & b))


def _fragments_along_line(
    seg: np.ndarray,
    lab: int,
    line: np.ndarray,
) -> list[np.ndarray]:
    """Split cell *lab* along *line* into its two largest fragments.

    Returns two full-frame boolean masks that together tile the whole cell (the
    drawn line pixels are handed to the nearest fragment, exactly like
    ``_split_in_crop``). Returns ``[]`` when the line does not run *all the way*
    through the cell — i.e. removing it leaves the cell in one piece, or one
    side is below ``MIN_CELL_SIZE``. ``seg`` is never modified.
    """
    bbox = _extend_bbox(_bbox_of_label(seg, lab), 1.0, seg.shape, min_pad=2)
    r0, c0, r1, c1 = bbox
    cell = _crop(seg, bbox) == lab
    crop_line = _crop(line, bbox).astype(bool)

    # Only a hair of dilation, to bridge a 1-px diagonal gap left by the drawn
    # line. Dilating further would force a split even where the line merely
    # clips the cell, so a stroke that doesn't run clean through stays a no-op.
    for retry in range(2):
        cut = binary_dilation(crop_line, disk(retry)) if retry else crop_line
        regions, n = nd_label(cell & ~cut)
        if n < 2:
            continue
        sizes = [int(np.sum(regions == i)) for i in range(1, n + 1)]
        id_a, id_b = sorted(range(1, n + 1), key=lambda i: sizes[i - 1], reverse=True)[:2]
        if sizes[id_a - 1] < MIN_CELL_SIZE or sizes[id_b - 1] < MIN_CELL_SIZE:
            continue
        two = np.zeros_like(regions)
        two[regions == id_a] = 1
        two[regions == id_b] = 2
        expanded = expand_labels(two, distance=max(retry + 2, 3))
        frags = []
        for v in (1, 2):
            full = np.zeros(seg.shape, dtype=bool)
            full[r0:r1, c0:c1] = (expanded == v) & cell
            frags.append(full)
        return frags
    return []


def carve_into_selected(
    seg: np.ndarray,
    positions: list,
    *,
    selected_label: int,
) -> bool:
    """Cut neighbouring cells along the drawn line and annex the near pieces.

    The user draws a line *through* one or more neighbouring cells, starting from
    (or towards) the selected cell. Every *other* cell the line crosses end to
    end is split along that line — exactly like the Shift+Right split gesture of
    the nucleus widget — into two fragments. The fragment that touches the
    selected cell is then merged into it; if both fragments touch, the smaller
    one is taken so the neighbour keeps its bulk. A cell the line only clips
    (without dividing it in two) is left alone.

    Cells are merged iteratively: a fragment that is not adjacent to the
    selection at first can become adjacent once a cell between them has been
    annexed, so each pass re-checks adjacency against the grown selection until
    no further fragment touches it. Only one fragment per neighbour is ever
    annexed, so the cell label set stays in lock-step with the nuclei — no label
    is created, deleted or renumbered.

    Cutting clean through the *selected* cell takes priority and is exclusive: if
    the line divides the selection in two, its smaller piece is removed entirely
    (set to background), the larger piece stays as the selected cell, and no
    neighbour is annexed on that stroke.

    A line that divides nothing (e.g. a straight swipe over background, or a
    stroke that never fully crosses a cell) is a no-op. Returns ``True`` if any
    pixel changed.
    """
    if not selected_label or not np.any(seg == selected_label):
        return False
    if len(positions) < 2:
        return False

    pts = [(float(p[-2]), float(p[-1])) for p in positions]
    line = _draw_line(seg.shape, _interpolate(pts))
    line_mask = line.astype(bool)
    if not line_mask.any():
        return False

    sel_mask = seg == selected_label

    # A line cutting clean through the selection trims it and nothing else: drop
    # the smaller piece to background, keep the larger one, and stop here.
    if (line_mask & sel_mask).any():
        sel_frags = _fragments_along_line(seg, int(selected_label), line)
        if len(sel_frags) >= 2:
            drop = min(sel_frags, key=lambda f: int(f.sum()))
            seg[drop] = 0
            return True

    crossed = set(int(v) for v in np.unique(seg[line_mask]))
    crossed.discard(0)
    crossed.discard(int(selected_label))

    # Pre-split every neighbour the line fully traverses. Each entry maps the
    # cell label to the two fragments it was divided into.
    pending: dict[int, list[np.ndarray]] = {}
    for lab in crossed:
        frags = _fragments_along_line(seg, lab, line)
        if len(frags) >= 2:
            pending[lab] = frags

    changed = False
    progress = True
    while pending and progress:
        progress = False
        for lab in list(pending):
            touching = [f for f in pending[lab] if _mask_touches(f, sel_mask)]
            if not touching:
                continue  # may become reachable after a neighbour is annexed
            annex = min(touching, key=lambda f: int(f.sum()))
            seg[annex] = selected_label
            sel_mask = sel_mask | annex
            del pending[lab]
            changed = True
            progress = True

    if not changed:
        return False

    # Annexing a slice can leave a thin background gap enclosed in the selection;
    # close only those background holes so the boundary stays clean.
    holes = binary_fill_holes(sel_mask) & ~sel_mask & (seg == 0)
    if holes.any():
        seg[holes] = selected_label
    return True


def draw_cell_path(
    seg: np.ndarray,
    positions: list,
    *,
    curlabel: int | None = None,
    new_label: int | None = None,
    protected_mask: np.ndarray | None = None,
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
    if protected_mask is not None:
        protected_mask = np.asarray(protected_mask, dtype=bool)
        if protected_mask.shape != seg.shape:
            raise ValueError(
                f"protected_mask shape {protected_mask.shape} does not match "
                f"segmentation shape {seg.shape}"
            )
        fill_mask &= ~protected_mask
    if extending:
        existing_mask = seg == label
        if protected_mask is not None:
            existing_mask &= ~protected_mask
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
        hole_fill = filled_mask & ~cell_mask
        if protected_mask is not None:
            hole_fill &= ~protected_mask
        seg[hole_fill] = label
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
    """Fill background holes enclosed within a single cell.

    A "hole" is a background component that does not touch the image border and
    is surrounded by exactly one label; it is filled with that label. Background
    bordered by two or more cells is an inter-cellular gap, not a hole, and is
    left untouched (so cells are never expanded into the space between them).
    ``radius`` caps the hole depth: a hole is only filled when its deepest pixel
    lies within ``radius`` of the surrounding cell. ``radius <= 0`` is a no-op.
    """
    if radius <= 0:
        return labels

    result = labels.copy()
    background = labels == 0
    bg_cc, n_cc = nd_label(background)
    if n_cc == 0:
        return result
    depth = distance_transform_edt(background)

    for comp_id in range(1, n_cc + 1):
        comp = bg_cc == comp_id
        if (
            np.any(comp[0, :])
            or np.any(comp[-1, :])
            or np.any(comp[:, 0])
            or np.any(comp[:, -1])
        ):
            continue
        ring = binary_dilation(comp) & ~comp
        neighbours = np.unique(labels[ring])
        neighbours = neighbours[neighbours != 0]
        if neighbours.size != 1:
            continue  # gap between cells, not a hole within one cell
        if depth[comp].max() > radius:
            continue
        result[comp] = neighbours[0]

    return result


def fix_label_semiholes(
    labels: np.ndarray,
    radius: int = 5,
    max_opening: int = 3,
) -> np.ndarray:
    """Fill a hole enclosed by a single cell except for a narrow image-edge opening.

    Like :func:`fill_label_holes`, but for background that escapes to the image
    border through a thin gap (up to ``max_opening`` pixels of edge contact).
    The opening's edge flanks are ignored; the hole is filled only when its
    interior is enclosed by exactly one label, so cells are never expanded into
    inter-cellular gaps. ``radius`` caps the hole depth as in
    :func:`fill_label_holes`. ``radius <= 0`` or ``max_opening <= 0`` is a no-op.
    """
    if max_opening <= 0 or radius <= 0:
        return labels

    result = labels.copy()
    background = labels == 0
    bg_cc, n_cc = nd_label(background)
    if n_cc == 0:
        return result
    depth = distance_transform_edt(background)
    edge = np.zeros(labels.shape, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True

    for comp_id in range(1, n_cc + 1):
        comp = bg_cc == comp_id
        border_contact = (
            int(np.count_nonzero(comp[0, :]))
            + int(np.count_nonzero(comp[-1, :]))
            + int(np.count_nonzero(comp[1:-1, 0]))
            + int(np.count_nonzero(comp[1:-1, -1]))
        )
        if border_contact == 0 or border_contact > max_opening:
            continue
        # Ignore ring pixels on the image edge: those flank the opening, not the
        # interior wall, and may belong to a different (outside) label.
        ring = binary_dilation(comp) & ~comp & ~edge
        neighbours = np.unique(labels[ring])
        neighbours = neighbours[neighbours != 0]
        if neighbours.size != 1:
            continue  # gap between cells, not a hole within one cell
        if depth[comp].max() > radius:
            continue
        result[comp] = neighbours[0]

    return result


def clean_stranded_pixels(
    seg: np.ndarray,
    min_size: int = MIN_CELL_SIZE,
    *,
    exclude: set[int] | None = None,
) -> int:
    """Remove disconnected same-label fragments, keeping each label's largest component.

    Fragments are cleared and then ``expand_labels`` is used to propose a new
    label.  The proposal is only accepted (written back) when the reassigned
    fragment pixels are 8-connected to an existing component of that same
    label — this prevents ``expand_labels`` from recreating a disconnected
    fragment.

    Labels in *exclude* are left untouched. This is used by ``merge_cells`` to
    preserve a merge target whose components are intentionally disconnected
    (e.g. after merging a fragmented nucleus).
    """
    from skimage.measure import label as _cc_label
    cleared = 0

    for cell_id in np.unique(seg):
        if cell_id == 0:
            continue
        if exclude is not None and int(cell_id) in exclude:
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

            # expand_labels and the keep-check only read pixels within the
            # expansion distance of the fragment, so crop to the fragment bbox
            # padded by that distance instead of scanning the whole frame per
            # fragment. Any label that can reach a fragment pixel within
            # ``distance`` lies inside this padded box, so the result is identical.
            distance = n_px + 2
            ys, xs = np.nonzero(comp_mask)
            y0 = max(int(ys.min()) - distance, 0)
            y1 = min(int(ys.max()) + distance + 1, seg.shape[0])
            x0 = max(int(xs.min()) - distance, 0)
            x1 = min(int(xs.max()) + distance + 1, seg.shape[1])
            sub = seg[y0:y1, x0:x1]
            sub_comp = comp_mask[y0:y1, x0:x1]

            filled = expand_labels(sub, distance=distance)
            new_labels = filled[sub_comp]
            keep = np.zeros(new_labels.shape, dtype=bool)

            for lbl in np.unique(new_labels):
                if lbl == 0:
                    continue
                assigned_here = new_labels == lbl
                assign_mask = np.zeros_like(sub, dtype=bool)
                assign_mask[sub_comp] = assigned_here
                dilated = binary_dilation(assign_mask, structure=np.ones((3, 3), dtype=bool))
                if np.any((sub == lbl) & dilated):
                    keep[assigned_here] = True

            sub[sub_comp] = np.where(keep, new_labels, 0).astype(seg.dtype)
            cleared += n_px

    return cleared

