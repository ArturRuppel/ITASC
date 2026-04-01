from __future__ import annotations

"""Graph extraction from segmentation labels.

Uses contour-based boundary detection inspired by ForSys's Skeleton approach:
boundary pixels between adjacent cells are detected directly from the label
image, and triple junction points (where 3+ cells meet) are computed so that
every junction extends fully to its endpoints.
"""
import cv2
import logging
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, FrozenSet
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
from skimage.measure import regionprops, label as cc_label
from skimage.segmentation import expand_labels

from ..utils.structures import CellData, JunctionData

logger = logging.getLogger(__name__)


def find_border_cells(label_frame: np.ndarray) -> set:
    """Return the set of cell IDs that touch the image border or background."""
    border_ids = set()
    # Top and bottom rows
    border_ids.update(np.unique(label_frame[0, :]))
    border_ids.update(np.unique(label_frame[-1, :]))
    # Left and right columns
    border_ids.update(np.unique(label_frame[:, 0]))
    border_ids.update(np.unique(label_frame[:, -1]))

    # Also include cells that touch background (label 0) anywhere
    kernel = np.ones((3, 3), np.uint8)
    bg_mask = (label_frame == 0).astype(np.uint8)
    bg_dilated = cv2.dilate(bg_mask, kernel)
    # Any cell label that overlaps with dilated background touches background
    touching_bg = np.unique(label_frame[bg_dilated > 0])
    border_ids.update(touching_bg)

    border_ids.discard(0)  # remove background
    return border_ids


def find_border_boundary(
    frame: np.ndarray,
    cell_id: int,
    min_edge_length: float = 0.0,
) -> list:
    """Find the boundary segments of a cell with the image border or background.

    A cell may have multiple disconnected border segments (e.g. a corner cell
    touching the top and right image edges).  Each segment is returned
    separately so that lengths are computed correctly.

    Returns a list of ``(ordered_coordinates, length)`` tuples.  The list is
    empty when the cell has no qualifying border boundary.
    """
    mask = (frame == cell_id).astype(np.uint8)

    # Find contour of the cell
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)
    # contour shape is (N, 1, 2) with (x, y) order
    contour_pts = contour.squeeze(1)  # (N, 2) in (x, y)
    if len(contour_pts) < 2:
        return []

    # Classify each contour point as border or not.
    # A border point sits on the image edge OR has a background (0) neighbour.
    h, w = frame.shape
    is_border_mask = np.zeros(len(contour_pts), dtype=bool)
    for idx, (x, y) in enumerate(contour_pts):
        if x == 0 or x == w - 1 or y == 0 or y == h - 1:
            is_border_mask[idx] = True
        else:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx_ = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx_ < w and frame[ny, nx_] == 0:
                        is_border_mask[idx] = True
                        break
                if is_border_mask[idx]:
                    break

    if not np.any(is_border_mask):
        return []

    # Split into contiguous segments.  The contour is a closed loop, so we
    # need to handle wrap-around: if border points span the start/end of the
    # contour array they belong to the same segment.
    segments: list = []
    current: list = []
    for idx in range(len(contour_pts)):
        if is_border_mask[idx]:
            x, y = contour_pts[idx]
            current.append([int(y), int(x)])  # store as (y, x)
        else:
            if current:
                segments.append(current)
                current = []
    if current:
        # If the very first point was also a border point the last and first
        # segments are actually one contiguous run around the loop.
        if segments and is_border_mask[0]:
            segments[0] = current + segments[0]
        else:
            segments.append(current)

    results = []
    for seg in segments:
        if len(seg) < 2:
            continue
        coords = np.array(seg)
        length = calculate_edge_length(coords)
        if min_edge_length > 0 and length < min_edge_length:
            continue
        results.append((coords, length))

    return results


def _find_boundary_pixels(
    label_frame: np.ndarray,
) -> Dict[FrozenSet[int], List]:
    """Detect all boundary pixels between adjacent cells.

    For each pair of horizontally or vertically adjacent pixels with different
    non-zero labels, a boundary point is recorded at their sub-pixel midpoint.
    This produces boundary coordinates that lie exactly on the cell–cell
    interface.

    Returns a dict mapping ``frozenset(cell_a, cell_b)`` to a list of
    ``[y, x]`` boundary positions (floats).
    """
    # Horizontal boundaries at (y, x + 0.5)
    h_left = label_frame[:, :-1]
    h_right = label_frame[:, 1:]
    h_mask = (h_left != h_right) & (h_left > 0) & (h_right > 0)
    h_ys, h_xs = np.where(h_mask)

    # Vertical boundaries at (y + 0.5, x)
    v_top = label_frame[:-1, :]
    v_bot = label_frame[1:, :]
    v_mask = (v_top != v_bot) & (v_top > 0) & (v_bot > 0)
    v_ys, v_xs = np.where(v_mask)

    junction_pixels: Dict[FrozenSet[int], List] = defaultdict(list)

    for i in range(len(h_ys)):
        pair = frozenset((int(h_left[h_ys[i], h_xs[i]]),
                          int(h_right[h_ys[i], h_xs[i]])))
        junction_pixels[pair].append([float(h_ys[i]), h_xs[i] + 0.5])

    for i in range(len(v_ys)):
        pair = frozenset((int(v_top[v_ys[i], v_xs[i]]),
                          int(v_bot[v_ys[i], v_xs[i]])))
        junction_pixels[pair].append([v_ys[i] + 0.5, float(v_xs[i])])

    return dict(junction_pixels)


def _find_triple_junctions(
    label_frame: np.ndarray,
) -> List[Tuple[float, float, FrozenSet[int]]]:
    """Find triple (or higher) junction positions from a label image.

    A triple junction is the meeting point of 3+ cells. It is detected by
    scanning every 2×2 pixel block: when 3+ distinct non-zero labels appear
    in the block, the centre ``(y + 0.5, x + 0.5)`` is a junction point.

    Returns a list of ``(y, x, frozenset_of_cell_ids)`` tuples.
    """
    tl = label_frame[:-1, :-1]
    tr = label_frame[:-1, 1:]
    bl = label_frame[1:, :-1]
    br = label_frame[1:, 1:]

    # Quick reject: blocks where all 4 pixels are the same
    candidate = ~((tl == tr) & (tr == bl) & (bl == br))
    tj_ys, tj_xs = np.where(candidate)

    results: List[Tuple[float, float, FrozenSet[int]]] = []
    for idx in range(len(tj_ys)):
        y, x = int(tj_ys[idx]), int(tj_xs[idx])
        block = {int(tl[y, x]), int(tr[y, x]),
                 int(bl[y, x]), int(br[y, x])}
        block.discard(0)
        if len(block) >= 3:
            results.append((y + 0.5, x + 0.5, frozenset(block)))
    return results


def _order_point_set(points: np.ndarray) -> np.ndarray:
    """Order 2-D points into a path by nearest-neighbour walk.

    Starts from one of the two most distant points (an endpoint) and
    greedily visits the nearest unvisited point.  After ordering, any
    large gap (where the walk jumps back) is detected and the longest
    contiguous segment is kept.
    """
    n = len(points)
    if n <= 2:
        return points

    D = cdist(points, points)
    # Start from one endpoint (point farthest from the centroid)
    start, _ = np.unravel_index(np.argmax(D), D.shape)

    ordered = np.empty_like(points)
    visited = np.zeros(n, dtype=bool)
    current = start
    for step in range(n):
        ordered[step] = points[current]
        visited[current] = True
        if step < n - 1:
            dists = D[current].copy()
            dists[visited] = np.inf
            current = int(np.argmin(dists))

    # Detect large gaps (spurious jumps) and re-stitch segments in
    # spatial order so that all points are preserved.
    if n > 3:
        diffs = np.diff(ordered, axis=0)
        step_dists = np.sqrt(np.sum(diffs**2, axis=1))
        median_step = np.median(step_dists)
        if median_step > 0:
            gap_threshold = max(8.0 * median_step, 5.0)
            gap_indices = np.where(step_dists > gap_threshold)[0]
            if len(gap_indices) > 0:
                # Split at gaps into segments
                boundaries = np.concatenate([[0], gap_indices + 1, [n]])
                segments = [
                    ordered[boundaries[i]:boundaries[i + 1]]
                    for i in range(len(boundaries) - 1)
                    if boundaries[i + 1] > boundaries[i]
                ]
                if len(segments) > 1:
                    # Stitch segments by connecting nearest endpoints
                    result = [segments.pop(0)]
                    while segments:
                        tail = result[-1][-1]
                        head = result[0][0]
                        best_i = 0
                        best_d = np.inf
                        best_flip = False
                        best_prepend = False
                        for i, seg in enumerate(segments):
                            for d, flip, pre in [
                                (np.linalg.norm(tail - seg[0]), False, False),
                                (np.linalg.norm(tail - seg[-1]), True, False),
                                (np.linalg.norm(head - seg[-1]), False, True),
                                (np.linalg.norm(head - seg[0]), True, True),
                            ]:
                                if d < best_d:
                                    best_d, best_i = d, i
                                    best_flip, best_prepend = flip, pre
                        seg = segments.pop(best_i)
                        if best_flip:
                            seg = seg[::-1]
                        if best_prepend:
                            result.insert(0, seg)
                        else:
                            result.append(seg)
                    ordered = np.concatenate(result)

    return ordered


def calculate_edge_length(coords: np.ndarray) -> float:
    """Calculate total Euclidean path length along ordered coordinates."""
    if len(coords) < 2:
        return 0.0
    diffs = np.diff(coords, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))


def labels_to_graph(
    label_frame: np.ndarray,
    min_edge_length: float = 0.0,
    filter_isolated: bool = True,
    min_border_edge_length: float = 5.0,
    min_bg_hole_size: int = 500,
    **kwargs,
) -> Tuple[Dict[int, CellData], Dict[FrozenSet[int], JunctionData], nx.Graph]:
    """Build cell data, junction data, and graph from a single label frame.

    Uses contour-based boundary detection: for each pair of adjacent pixels
    with different labels, a boundary point is placed at their sub-pixel
    midpoint.  Triple junction points (where 3+ cells meet) are detected
    from 2×2 pixel blocks and added as junction endpoints.  This ensures
    that junctions extend fully to where cells meet, matching the topology
    produced by ForSys's Skeleton approach.

    Args:
        label_frame: 2-D integer array, 0 = background.  Cells should be
            touching (no gaps) for best results; use
            ``skimage.segmentation.expand_labels`` to fill gaps first.
        min_edge_length: Minimum junction length (px) to keep.
        filter_isolated: If True, detect border cells and add tagged
            ``"border"`` junctions for cell–background boundaries.
        min_border_edge_length: Minimum length (px) for a border boundary
            segment to count.
        min_bg_hole_size: Background regions smaller than this many pixels
            are treated as segmentation artifacts and ignored when detecting
            cell-background borders.  Set to 0 to disable filtering.

    Returns:
        (cells, junctions, graph)
    """
    H, W = label_frame.shape
    cell_ids = np.unique(label_frame)
    cell_ids = cell_ids[cell_ids != 0]

    if len(cell_ids) == 0:
        return {}, {}, nx.Graph()

    props = regionprops(label_frame)
    props_by_label = {p.label: p for p in props}

    # ---- Step 0: fill artifact background gaps ----
    # smooth_labels (sigma≈4, thresh≈0.4) leaves thin (1–3 px) background
    # strips at every cell junction.  These strips prevent direct-adjacency
    # detection between neighbouring cells and make peripheral cells appear
    # isolated (no edges).  We build a clean frame by:
    #   - keeping real outer tissue background (connected region ≥ 1 % of image)
    #   - filling all smaller background components with the nearest cell label
    # The clean frame is used for both junction detection and border detection.
    bg_mask = label_frame == 0
    bg_labeled = cc_label(bg_mask)
    if bg_labeled.max() > 0:
        bg_sizes = np.bincount(bg_labeled.ravel())
        bg_sizes[0] = 0
        size_threshold = max(1, int(0.01 * label_frame.size))
        real_bg_ids = np.where(bg_sizes >= size_threshold)[0]
        if real_bg_ids.size == 0:
            real_bg_ids = np.array([int(bg_sizes.argmax())])
        outer_bg_mask = np.isin(bg_labeled, real_bg_ids)
    else:
        outer_bg_mask = bg_mask  # no background at all

    artifact_mask = bg_mask & ~outer_bg_mask
    if artifact_mask.any():
        filled = expand_labels(label_frame, distance=label_frame.shape[0])
        clean_frame = label_frame.copy()
        clean_frame[artifact_mask] = filled[artifact_mask]
    else:
        clean_frame = label_frame

    # ---- Step 1: boundary pixels between adjacent cells ----
    junction_pixels = _find_boundary_pixels(clean_frame)

    # ---- Step 2: triple junction detection ----
    triple_junctions = _find_triple_junctions(clean_frame)

    # Map each cell pair to its triple junction endpoints
    pair_to_tjs: Dict[FrozenSet[int], List] = defaultdict(list)
    for tj_y, tj_x, tj_cells in triple_junctions:
        cells_list = sorted(tj_cells)
        for i in range(len(cells_list)):
            for j in range(i + 1, len(cells_list)):
                pair_to_tjs[frozenset((cells_list[i], cells_list[j]))].append(
                    [tj_y, tj_x]
                )

    # ---- Step 3: build junctions ----
    graph = nx.Graph()
    junctions: Dict[FrozenSet[int], JunctionData] = {}

    for pair, pixels in junction_pixels.items():
        coords = np.array(pixels)

        # Append triple junction endpoints so the junction reaches them
        tjs = pair_to_tjs.get(pair, [])
        if tjs:
            coords = np.vstack([coords, np.array(tjs)])

        if len(coords) < 2:
            continue

        ordered = _order_point_set(coords)
        length = calculate_edge_length(ordered)

        if min_edge_length > 0 and length < min_edge_length:
            continue

        sorted_pair = tuple(sorted(pair))
        midpoint = ordered[len(ordered) // 2].astype(float)

        junctions[pair] = JunctionData(
            cell_pair=sorted_pair,
            length=length,
            coordinates=ordered,
            midpoint=midpoint,
        )
        graph.add_edge(sorted_pair[0], sorted_pair[1], length=length)

    # ---- Step 4: detect border cells and add cell-background edges ----
    # Binarise label_frame → fill holes → trace the outer tissue contour →
    # assign each contour pixel to the cell it belongs to (via clean_frame) →
    # any cell whose total contour arc is ≥ min_border_edge_length is a border cell.
    # For each border cell, create a JunctionData representing the cell-background
    # interface (cell_pair=(0, cell_id)), tagged as "border".
    border_cell_ids: set = set()
    if filter_isolated:
        from scipy.ndimage import binary_fill_holes

        binary = (label_frame > 0).astype(np.uint8)
        filled_binary = binary_fill_holes(binary).astype(np.uint8)

        contours_cv, _ = cv2.findContours(
            filled_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
        )
        if contours_cv:
            tissue_contour = max(contours_cv, key=cv2.contourArea).squeeze(1)
            # tissue_contour: (N, 2) in (x, y) order

            # Accumulate border arc-length per cell.  Only sum distances between
            # consecutive contour pixels that belong to the same cell so that
            # short isolated touches don't inflate a cell's border length.
            border_length: Dict[int, float] = defaultdict(float)
            prev_cell: int = -1
            prev_xy: Optional[Tuple[int, int]] = None
            for (x, y) in tissue_contour:
                cell_id = int(clean_frame[y, x])
                if cell_id <= 0:
                    prev_cell = -1
                    prev_xy = None
                    continue
                if cell_id == prev_cell and prev_xy is not None:
                    dx = x - prev_xy[0]
                    dy = y - prev_xy[1]
                    border_length[cell_id] += np.sqrt(dx * dx + dy * dy)
                prev_cell = cell_id
                prev_xy = (x, y)

            for cell_id, length in border_length.items():
                if length >= min_border_edge_length:
                    border_cell_ids.add(cell_id)

        # Build a frame for border detection where small background holes
        # (segmentation artifacts between cells) are filled in so they don't
        # create spurious border junctions.
        if min_bg_hole_size > 0:
            bg_mask_bd = label_frame == 0
            bg_labeled_bd = cc_label(bg_mask_bd)
            if bg_labeled_bd.max() > 0:
                bg_sizes_bd = np.bincount(bg_labeled_bd.ravel())
                bg_sizes_bd[0] = 0
                small_ids = np.where(
                    (bg_sizes_bd > 0) & (bg_sizes_bd < min_bg_hole_size)
                )[0]
                if small_ids.size > 0:
                    border_frame = label_frame.copy()
                    small_mask = np.isin(bg_labeled_bd, small_ids)
                    filled_bd = expand_labels(
                        label_frame, distance=label_frame.shape[0]
                    )
                    border_frame[small_mask] = filled_bd[small_mask]
                else:
                    border_frame = label_frame
            else:
                border_frame = label_frame
        else:
            border_frame = label_frame

        # For each border cell, create a JunctionData for the cell-background
        # interface using the actual contour of the cell against background.
        for border_cid in border_cell_ids:
            segments = find_border_boundary(border_frame, border_cid, min_edge_length=0.0)
            if not segments:
                continue
            all_coords = np.vstack([seg[0] for seg in segments])
            total_length = sum(seg[1] for seg in segments)
            if total_length < min_border_edge_length:
                continue
            midpoint = all_coords[len(all_coords) // 2].astype(float)
            pair_key = frozenset({0, border_cid})
            junctions[pair_key] = JunctionData(
                cell_pair=(0, border_cid),
                length=total_length,
                coordinates=all_coords,
                midpoint=midpoint,
                tags={"border"},
            )

    # ---- Step 5: build cell data ----
    cells_in_graph = set()
    for pair in junctions:
        cells_in_graph.update(pair)

    cells: Dict[int, CellData] = {}
    for cid in cells_in_graph:
        p = props_by_label.get(cid)
        if p is None:
            continue
        area = float(p.area)
        perimeter = float(p.perimeter)
        shape_index = perimeter / np.sqrt(area) if area > 0 else 0.0
        num_neighbors = graph.degree(cid) if graph.has_node(cid) else 0

        # Extract cell contour for polygon vertices
        mask = (label_frame == cid).astype(np.uint8)
        contours_cv, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
        )
        vertices = None
        if contours_cv:
            contour = max(contours_cv, key=cv2.contourArea).squeeze(1)
            if contour.ndim == 2 and len(contour) >= 3:
                vertices = contour[:, ::-1].astype(float)  # OpenCV (x,y) → (y,x)

        cells[cid] = CellData(
            cell_id=cid,
            position=np.array(p.centroid),
            area=area,
            perimeter=perimeter,
            shape_index=shape_index,
            num_neighbors=num_neighbors,
            vertices=vertices,
            is_border=(cid in border_cell_ids),
        )

    return cells, junctions, graph
"""
Label correction operations on a single (H, W) segmentation frame.

All functions accept a 2-D ``seg`` array and modify it **in-place**.
They return ``True`` on success and ``False`` when the operation is
rejected (e.g. labels don't touch, result too small, background click).

No CellFlow graph is modified — re-run graph extraction after corrections.
"""


import logging
import os
from collections import Counter

import numpy as np
from scipy.ndimage import binary_dilation, binary_closing, label as nd_label
from scipy.ndimage import distance_transform_edt
from skimage.morphology import disk
from skimage.segmentation import watershed, find_boundaries, expand_labels

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


def _extend_endpoints(
    pts: list[tuple[float, float]], distance: float
) -> list[tuple[float, float]]:
    """Prepend and append one extra point beyond each endpoint along the endpoint tangent.

    This ensures the drawn line reaches (and crosses) the cell boundary even
    when the user's drag started or ended slightly inside the cell.
    """
    if len(pts) < 2:
        return pts

    def _extend(pt: tuple, neighbor: tuple, dist: float) -> tuple[float, float]:
        dr = pt[0] - neighbor[0]
        dc = pt[1] - neighbor[1]
        mag = (dr ** 2 + dc ** 2) ** 0.5
        if mag < 1e-6:
            return pt
        return (pt[0] + dr / mag * dist, pt[1] + dc / mag * dist)

    new_start = _extend(pts[0],  pts[1],       distance)
    new_end   = _extend(pts[-1], pts[-2],       distance)
    return [new_start] + list(pts) + [new_end]


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
    log.debug("erase_cell: label=%s pos=%s", label, pos)
    if label == 0:
        log.debug("erase_cell: rejected — label is background")
        return False
    count = int(np.sum(seg == label))
    seg[seg == label] = 0
    log.debug("erase_cell: erased %d pixels", count)
    return True


def merge_cells(seg: np.ndarray, pos_start: tuple, pos_end: tuple) -> bool:
    """
    Merge the cell at *pos_start* into the cell at *pos_end*.

    The two labels must be touching; otherwise the operation is rejected
    with a return value of ``False``.
    """
    la = _label_at(seg, pos_start)
    lb = _label_at(seg, pos_end)
    log.debug("merge_cells: la=%s pos_start=%s  lb=%s pos_end=%s", la, pos_start, lb, pos_end)
    if la == 0 or lb == 0 or la == lb:
        log.debug("merge_cells: rejected — background or same label (la=%s lb=%s)", la, lb)
        return False
    touching = _touches(seg, la, lb)
    log.debug("merge_cells: touching=%s", touching)
    if not touching:
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
    log.debug("split_across: la=%s lb=%s pos_start=%s pos_end=%s", la, lb, pos_start, pos_end)
    if la == 0 or la != lb:
        log.debug("split_across: rejected — background or different labels (la=%s lb=%s)", la, lb)
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
    log.debug(
        "split_across: label=%s cell_px=%d bbox=%s seed_A_local=(%d,%d) seed_B_local=(%d,%d) img=%s",
        la, int(mask.sum()), bbox, rs, cs, re, ce, "yes" if img is not None else "no",
    )

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

        size_a = int(np.sum(ws == la))
        size_b = int(np.sum(ws == new_lab))
        log.debug("split_across: radius=%d  ws_A=%d ws_B=%d (min=%d)", radius, size_a, size_b, MIN_CELL_SIZE)
        if size_a >= MIN_CELL_SIZE and size_b >= MIN_CELL_SIZE:
            seg[r0:r1, c0:c1][ws == new_lab] = new_lab
            log.debug("split_across: success new_lab=%s", new_lab)
            return True

    log.debug("split_across: failed after all radii")
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
    log.debug("split_draw: %d raw positions, curlabel=%s", len(positions), curlabel)
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
        log.debug("split_draw: labels_under counter=%s", Counter(labels_under).most_common(4))
        if not labels_under:
            log.debug("split_draw: rejected — no labels under drawn path")
            return False
        curlabel = max(set(labels_under), key=labels_under.count)
        if curlabel == 0:
            log.debug("split_draw: rejected — most common label is background")
            return False
    log.debug("split_draw: target label=%s", curlabel)

    # re-crop around the cell itself so only that cell's region is in play
    bbox = _bbox_of_label(seg, curlabel)
    bbox = _extend_bbox(bbox, 1.25, seg.shape)
    crop = _crop(seg, bbox).copy()
    local_pts = _to_local(positions, bbox)

    # Extend endpoints so the line is guaranteed to cross the cell boundary
    h = bbox[2] - bbox[0]
    w = bbox[3] - bbox[1]
    extend_dist = max(int(np.sqrt(h ** 2 + w ** 2) / 2), 10)
    local_pts = _extend_endpoints(local_pts, extend_dist)

    interp = _interpolate(local_pts)
    line = _draw_line(crop.shape, interp)
    cell_px = int(np.sum(crop == curlabel))
    line_on_cell = int(np.sum(line & (crop == curlabel)))
    log.debug(
        "split_draw: bbox=%s cell_px=%d  interp_pts=%d line_on_cell=%d extend_dist=%d",
        bbox, cell_px, len(interp), line_on_cell, extend_dist,
    )

    if line_on_cell == 0:
        # The drawn stroke missed the target cell entirely.  Fall back to
        # inferring the target from the labels actually under the path.
        log.debug("split_draw: line_on_cell=0 — falling back to path inference")
        tight_bbox = _bbox_of_pts(positions)
        tight_bbox = _extend_bbox(tight_bbox, 1.1, seg.shape)
        crop_tight = _crop(seg, tight_bbox)
        local_pts_tight = _to_local(positions, tight_bbox)
        labels_under = [
            int(crop_tight[int(round(r)), int(round(c))])
            for r, c in local_pts_tight
            if 0 <= int(round(r)) < crop_tight.shape[0]
            and 0 <= int(round(c)) < crop_tight.shape[1]
        ]
        log.debug("split_draw fallback: labels_under=%s", Counter(labels_under).most_common(4))
        fallback_label = max(
            (lab for lab in set(labels_under) if lab != 0),
            key=labels_under.count,
            default=0,
        )
        if fallback_label == 0 or fallback_label == curlabel:
            log.debug("split_draw fallback: no usable label found — giving up")
            return False
        log.debug("split_draw fallback: switching target to label=%s", fallback_label)
        curlabel = fallback_label
        bbox = _bbox_of_label(seg, curlabel)
        bbox = _extend_bbox(bbox, 1.25, seg.shape)
        crop = _crop(seg, bbox).copy()
        local_pts = _to_local(positions, bbox)
        h = bbox[2] - bbox[0]
        w = bbox[3] - bbox[1]
        extend_dist = max(int(np.sqrt(h ** 2 + w ** 2) / 2), 10)
        local_pts = _extend_endpoints(local_pts, extend_dist)
        interp = _interpolate(local_pts)
        line = _draw_line(crop.shape, interp)
        line_on_cell = int(np.sum(line & (crop == curlabel)))
        log.debug(
            "split_draw fallback: bbox=%s cell_px=%d interp_pts=%d line_on_cell=%d",
            bbox, int(np.sum(crop == curlabel)), len(interp), line_on_cell,
        )
        if line_on_cell == 0:
            log.debug("split_draw fallback: line still misses cell — giving up")
            return False

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
        log.debug("_split_in_crop: failed after 6 retries")
        return False

    dilated = binary_dilation(line, disk(retry)) if retry > 0 else line.astype(bool)
    mask = np.zeros(crop.shape, dtype=np.uint8)
    mask[crop == curlabel] = 1
    mask[dilated] = 0

    regions, n = nd_label(mask)
    sizes = [int(np.sum(regions == i)) for i in range(1, n + 1)]
    log.debug("_split_in_crop: retry=%d n_regions=%d sizes=%s", retry, n, sizes)
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
        log.debug("_split_in_crop: success new_lab=%s at retry=%d", new_lab, retry)
        return True

    return _split_in_crop(seg, crop, line, bbox, curlabel, retry + 1)


def draw_cell_path(
    seg: np.ndarray,
    positions: list,
    *,
    curlabel: int | None = None,
) -> bool:
    """
    Draw a thin 1-px barrier line and flood-fill the target cell into the
    enclosed background region.

    If *curlabel* is set and present in *seg*, background pixels that are
    connected to the cell's existing pixels — but cut off from the rest of the
    image by the drawn line and other cell labels — are absorbed into the cell.

    If no cell is selected, a new label is created from the enclosed background
    region adjacent to the drawn line.  If the path encloses nothing, the thin
    line pixels themselves become the new cell (fallback).

    Returns True on success, False if the stroke is too short or the fill area
    is too small.
    """
    log.debug("draw_cell_path: %d raw positions, curlabel=%s", len(positions), curlabel)
    if len(positions) < 2:
        return False

    local_pts = [(float(p[-2]), float(p[-1])) for p in positions]
    interp = _interpolate(local_pts)
    line_mask = _draw_line(seg.shape, interp).astype(bool)

    if not np.any(line_mask):
        log.debug("draw_cell_path: rejected — empty line")
        return False

    has_cell = bool(curlabel) and curlabel != 0 and np.any(seg == curlabel)

    if has_cell:
        # Flood-fill from the cell's pixels into background, blocked by the
        # drawn line and every other non-zero label.
        traversable = (seg == curlabel) | ((seg == 0) & ~line_mask)
        labeled_t, _ = nd_label(traversable)
        cell_comp_ids = set(np.unique(labeled_t[seg == curlabel])) - {0}
        fill_mask = (seg == 0) & np.isin(labeled_t, list(cell_comp_ids))
        n_px = int(np.sum(fill_mask))
        log.debug("draw_cell_path: flood-fill px=%d for label=%s", n_px, curlabel)
        if n_px < MIN_CELL_SIZE:
            log.debug("draw_cell_path: rejected — fill area too small")
            return False
        seg[fill_mask] = curlabel
        return True

    # No cell selected: find background regions enclosed by the drawn line.
    background_open = (seg == 0) & ~line_mask
    labeled_bg, n_comp = nd_label(background_open)
    border_ids: set = set()
    for edge in (labeled_bg[0, :], labeled_bg[-1, :],
                 labeled_bg[:, 0], labeled_bg[:, -1]):
        border_ids.update(np.unique(edge))
    border_ids.discard(0)
    enclosed_ids = [i for i in range(1, n_comp + 1) if i not in border_ids]

    if enclosed_ids:
        fill_mask = np.isin(labeled_bg, enclosed_ids)
        n_px = int(np.sum(fill_mask))
        log.debug("draw_cell_path: enclosed fill px=%d", n_px)
        if n_px >= MIN_CELL_SIZE:
            new_label = _free_label(seg)
            seg[fill_mask] = new_label
            log.debug("draw_cell_path: new label=%s from enclosed region", new_label)
            return True

    # Fallback: assign the thin line pixels themselves as a new cell.
    stroke_mask = line_mask & (seg == 0)
    n_px = int(np.sum(stroke_mask))
    log.debug("draw_cell_path: fallback thin stroke px=%d", n_px)
    if n_px < MIN_CELL_SIZE:
        log.debug("draw_cell_path: rejected — stroke too small")
        return False
    new_label = _free_label(seg)
    seg[stroke_mask] = new_label
    log.debug("draw_cell_path: new label=%s from thin stroke", new_label)
    return True


def swap_labels(seg: np.ndarray, pos_a: tuple, pos_b: tuple) -> bool:
    """Swap the label values at the two click positions across the whole frame."""
    la = _label_at(seg, pos_a)
    lb = _label_at(seg, pos_b)
    log.debug("swap_labels: la=%s pos_a=%s  lb=%s pos_b=%s", la, pos_a, lb, pos_b)
    if la == 0 or lb == 0 or la == lb:
        log.debug("swap_labels: rejected — background or same label (la=%s lb=%s)", la, lb)
        return False
    count_a = int(np.sum(seg == la))
    count_b = int(np.sum(seg == lb))
    mask_a = seg == la
    mask_b = seg == lb
    seg[mask_a] = lb
    seg[mask_b] = la
    log.debug("swap_labels: swapped la=%s(%dpx) ↔ lb=%s(%dpx)", la, count_a, lb, count_b)
    return True
