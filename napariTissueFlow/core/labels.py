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

from ..structures import CellData, JunctionData

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
