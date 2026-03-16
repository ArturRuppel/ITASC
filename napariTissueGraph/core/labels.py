"""Graph extraction from segmentation labels.

Ported from napariCellFlow's edge_analysis.py, refactored into standalone functions.
"""
import cv2
import logging
import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional, FrozenSet
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
from skimage.measure import regionprops

from ..structures import CellData, JunctionData

logger = logging.getLogger(__name__)


def find_shared_boundary(
    frame: np.ndarray,
    cell1_id: int,
    cell2_id: int,
    dilation_radius: int = 1,
    min_overlap_pixels: int = 5,
    min_edge_length: float = 0.0,
) -> Optional[Tuple[np.ndarray, float]]:
    """Find the shared boundary between two cells in a label frame.

    Returns (ordered_coordinates, length) or None if no valid boundary.
    """
    mask1 = (frame == cell1_id).astype(np.uint8)
    mask2 = (frame == cell2_id).astype(np.uint8)

    # Dilate to find overlap
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * dilation_radius + 1, 2 * dilation_radius + 1),
    )
    dilated1 = cv2.dilate(mask1, kernel)
    dilated2 = cv2.dilate(mask2, kernel)
    overlap = dilated1 & dilated2

    if not np.any(overlap):
        return None

    # Thin dilation for precise boundary
    thin_kernel = np.ones((3, 3), np.uint8)
    thin_dilated1 = cv2.dilate(mask1, thin_kernel)
    thin_dilated2 = cv2.dilate(mask2, thin_kernel)
    boundary_region = thin_dilated1 & thin_dilated2

    if np.sum(boundary_region) < min_overlap_pixels:
        return None

    skeleton = skeletonize(boundary_region)
    if not np.any(skeleton):
        return None

    ordered_coords = order_boundary_pixels(skeleton)
    if len(ordered_coords) < 2:
        return None

    length = calculate_edge_length(ordered_coords)
    if min_edge_length > 0 and length < min_edge_length:
        return None

    return ordered_coords, length


def order_boundary_pixels(skeleton: np.ndarray) -> np.ndarray:
    """Order pixels along a skeletonized boundary from endpoint to endpoint."""
    points = np.column_stack(np.where(skeleton))
    if len(points) <= 2:
        return points

    # Find endpoints (pixels with exactly 1 neighbor)
    endpoint_indices = []
    for i, point in enumerate(points):
        y, x = point
        neighbors = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < skeleton.shape[0] and 0 <= nx_ < skeleton.shape[1]:
                    if skeleton[ny, nx_]:
                        neighbors += 1
        if neighbors == 1:
            endpoint_indices.append(i)

    if len(endpoint_indices) != 2:
        distances = cdist(points, points)
        i, j = np.unravel_index(np.argmax(distances), distances.shape)
        endpoint_indices = [i, j]

    # Walk from one endpoint to the other
    ordered = [points[endpoint_indices[0]]]
    remaining = list(range(len(points)))
    remaining.remove(endpoint_indices[0])

    while remaining:
        current = ordered[-1]
        min_dist = float("inf")
        next_idx = None
        for idx in remaining:
            dist = np.sum((points[idx] - current) ** 2)
            if dist < min_dist:
                min_dist = dist
                next_idx = idx
        if next_idx is None:
            break
        ordered.append(points[next_idx])
        remaining.remove(next_idx)

    return np.array(ordered)


def calculate_edge_length(coords: np.ndarray) -> float:
    """Calculate total Euclidean path length along ordered coordinates."""
    if len(coords) < 2:
        return 0.0
    diffs = np.diff(coords, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))


def labels_to_graph(
    label_frame: np.ndarray,
    dilation_radius: int = 1,
    min_overlap_pixels: int = 5,
    min_edge_length: float = 0.0,
    filter_isolated: bool = True,
) -> Tuple[Dict[int, CellData], Dict[FrozenSet[int], JunctionData], nx.Graph]:
    """Build cell data, junction data, and graph from a single label frame.

    Args:
        label_frame: 2D integer array, 0 = background.
        dilation_radius: Radius for dilation when detecting adjacency.
        min_overlap_pixels: Minimum boundary pixels to count as adjacent.
        min_edge_length: Minimum junction length to keep.
        filter_isolated: Remove edges where either cell has only one neighbor.

    Returns:
        (cells, junctions, graph)
    """
    cell_ids = np.unique(label_frame)
    cell_ids = cell_ids[cell_ids != 0]

    # Get cell properties from regionprops
    props = regionprops(label_frame)
    props_by_label = {p.label: p for p in props}

    # Detect all junctions
    raw_junctions = {}
    for i, cell1 in enumerate(cell_ids):
        for cell2 in cell_ids[i + 1:]:
            result = find_shared_boundary(
                label_frame, int(cell1), int(cell2),
                dilation_radius=dilation_radius,
                min_overlap_pixels=min_overlap_pixels,
                min_edge_length=min_edge_length,
            )
            if result is not None:
                coords, length = result
                pair = frozenset((int(cell1), int(cell2)))
                raw_junctions[pair] = (coords, length)

    # Filter isolated edges
    if filter_isolated:
        from collections import defaultdict
        connections = defaultdict(set)
        for pair in raw_junctions:
            c1, c2 = pair
            connections[c1].add(c2)
            connections[c2].add(c1)
        raw_junctions = {
            pair: val
            for pair, val in raw_junctions.items()
            if all(len(connections[c]) > 1 for c in pair)
        }

    # Build graph
    graph = nx.Graph()
    junctions: Dict[FrozenSet[int], JunctionData] = {}

    for pair, (coords, length) in raw_junctions.items():
        sorted_pair = tuple(sorted(pair))
        midpoint = coords[len(coords) // 2].astype(float)
        jd = JunctionData(
            cell_pair=sorted_pair,
            length=length,
            coordinates=coords,
            midpoint=midpoint,
        )
        junctions[pair] = jd
        graph.add_edge(sorted_pair[0], sorted_pair[1], length=length)

    # Build cell data — only for cells that appear in junctions
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
        cells[cid] = CellData(
            cell_id=cid,
            position=np.array(p.centroid),
            area=area,
            perimeter=perimeter,
            shape_index=shape_index,
            num_neighbors=num_neighbors,
        )

    return cells, junctions, graph
