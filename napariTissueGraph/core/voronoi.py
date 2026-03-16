"""Voronoi tessellation from nuclear positions."""
import logging
import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional, FrozenSet
from scipy.spatial import Voronoi

from ..structures import CellData, JunctionData

logger = logging.getLogger(__name__)


def compute_voronoi(
    positions: np.ndarray,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Voronoi:
    """Compute Voronoi tessellation from 2D positions.

    Args:
        positions: Nx2 array of (y, x) positions.
        image_shape: Optional (H, W) to add mirror points for bounded tessellation.

    Returns:
        scipy.spatial.Voronoi object.
    """
    if image_shape is not None:
        # Add mirror points beyond each boundary to close edge regions
        H, W = image_shape
        mirrored = [positions]
        mirrored.append(np.column_stack((-positions[:, 0], positions[:, 1])))  # top
        mirrored.append(np.column_stack((2 * H - positions[:, 0], positions[:, 1])))  # bottom
        mirrored.append(np.column_stack((positions[:, 0], -positions[:, 1])))  # left
        mirrored.append(np.column_stack((positions[:, 0], 2 * W - positions[:, 1])))  # right
        all_points = np.vstack(mirrored)
    else:
        all_points = positions

    return Voronoi(all_points)


def voronoi_to_graph(
    vor: Voronoi,
    positions: np.ndarray,
    n_real: int,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[int, CellData], Dict[FrozenSet[int], JunctionData], nx.Graph]:
    """Extract cell data, junction data, and graph from a Voronoi tessellation.

    Args:
        vor: Voronoi object (possibly with mirror points).
        positions: Original Nx2 array of (y, x) positions (real cells only).
        n_real: Number of real cells (first n_real points in vor are real).
        image_shape: Optional (H, W) for clipping.

    Returns:
        (cells, junctions, graph)
    """
    graph = nx.Graph()
    junctions: Dict[FrozenSet[int], JunctionData] = {}
    cells: Dict[int, CellData] = {}

    # Process ridges (edges between Voronoi regions)
    for (p1, p2), ridge_verts in zip(vor.ridge_points, vor.ridge_vertices):
        # Only consider ridges involving real cells
        if p1 >= n_real and p2 >= n_real:
            continue
        # Skip if either vertex is at infinity
        if -1 in ridge_verts:
            continue

        # Map mirror points back to real cell indices
        real_p1 = p1 if p1 < n_real else p1 % n_real
        real_p2 = p2 if p2 < n_real else p2 % n_real

        if real_p1 == real_p2:
            continue

        verts = vor.vertices[ridge_verts]

        # Clip to image bounds if provided
        if image_shape is not None:
            H, W = image_shape
            if np.any(verts[:, 0] < 0) or np.any(verts[:, 0] > H):
                continue
            if np.any(verts[:, 1] < 0) or np.any(verts[:, 1] > W):
                continue

        pair = frozenset((real_p1, real_p2))
        if pair in junctions:
            continue

        sorted_pair = tuple(sorted(pair))
        length = float(np.linalg.norm(verts[0] - verts[-1]))
        midpoint = np.mean(verts, axis=0)

        junctions[pair] = JunctionData(
            cell_pair=sorted_pair,
            length=length,
            coordinates=verts,
            midpoint=midpoint,
        )
        graph.add_edge(sorted_pair[0], sorted_pair[1], length=length)

    # Build cell data
    for i in range(n_real):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        if -1 in region or len(region) == 0:
            # Boundary cell with infinite region — still include with estimated properties
            num_neighbors = graph.degree(i) if graph.has_node(i) else 0
            cells[i] = CellData(
                cell_id=i,
                position=positions[i],
                area=0.0,
                perimeter=0.0,
                shape_index=0.0,
                num_neighbors=num_neighbors,
            )
            continue

        verts = vor.vertices[region]
        area = _polygon_area(verts)
        perimeter = _polygon_perimeter(verts)
        shape_index = perimeter / np.sqrt(area) if area > 0 else 0.0
        num_neighbors = graph.degree(i) if graph.has_node(i) else 0

        cells[i] = CellData(
            cell_id=i,
            position=positions[i],
            area=area,
            perimeter=perimeter,
            shape_index=shape_index,
            num_neighbors=num_neighbors,
        )

    return cells, junctions, graph


def _polygon_area(vertices: np.ndarray) -> float:
    """Shoelace formula for polygon area."""
    n = len(vertices)
    if n < 3:
        return 0.0
    y = vertices[:, 0]
    x = vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _polygon_perimeter(vertices: np.ndarray) -> float:
    """Sum of edge lengths of a polygon."""
    if len(vertices) < 2:
        return 0.0
    closed = np.vstack([vertices, vertices[0:1]])
    diffs = np.diff(closed, axis=0)
    return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))
