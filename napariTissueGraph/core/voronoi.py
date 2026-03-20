"""Voronoi tessellation from nuclear positions."""
import logging
import numpy as np
import networkx as nx
from typing import Dict, Tuple, Optional, FrozenSet
from scipy.spatial import Voronoi, cKDTree

from ..structures import CellData, JunctionData, VoronoiMethod

logger = logging.getLogger(__name__)


def compute_voronoi(
    positions: np.ndarray,
    image_shape: Optional[Tuple[int, int]] = None,
    method: VoronoiMethod = VoronoiMethod.STANDARD,
    lloyd_iterations: int = 10,
    lloyd_tol: float = 0.1,
) -> Tuple[Voronoi, np.ndarray]:
    """Compute Voronoi tessellation from 2D positions.

    Args:
        positions: Nx2 array of (y, x) positions.
        image_shape: Optional (H, W) to add mirror points for bounded tessellation.
        method: VoronoiMethod.STANDARD or VoronoiMethod.LLOYD.
        lloyd_iterations: Max iterations for Lloyd's relaxation.
        lloyd_tol: Convergence tolerance (max displacement) for Lloyd's.

    Returns:
        (Voronoi object, final positions after any relaxation).
    """
    if method == VoronoiMethod.LLOYD and image_shape is not None:
        positions = lloyd_relaxation(
            positions, image_shape, lloyd_iterations, lloyd_tol
        )

    vor = _raw_voronoi(positions, image_shape)
    return vor, positions


def _raw_voronoi(
    positions: np.ndarray,
    image_shape: Optional[Tuple[int, int]] = None,
) -> Voronoi:
    """Compute raw Voronoi tessellation with optional mirror points."""
    if image_shape is not None:
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


def lloyd_relaxation(
    positions: np.ndarray,
    image_shape: Tuple[int, int],
    n_iterations: int = 10,
    tol: float = 0.1,
) -> np.ndarray:
    """Lloyd's algorithm (centroidal Voronoi tessellation).

    Each iteration: compute Voronoi -> move seeds to polygon centroids -> repeat.
    After convergence, position == shape centroid.

    Args:
        positions: Nx2 array of (y, x) seed positions.
        image_shape: (H, W) bounding box.
        n_iterations: Maximum iterations.
        tol: Stop when max displacement < tol.

    Returns:
        Final Nx2 positions after relaxation.
    """
    H, W = image_shape
    pts = positions.copy()

    for iteration in range(n_iterations):
        vor = _raw_voronoi(pts, image_shape)
        n_real = len(pts)
        new_pts = pts.copy()

        for i in range(n_real):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]

            if -1 in region or len(region) < 3:
                continue

            verts = vor.vertices[region]
            # Clip vertices to image bounds
            verts = np.clip(verts, [0, 0], [H, W])

            centroid = _polygon_centroid(verts)
            if centroid is not None:
                new_pts[i] = centroid

        max_disp = np.max(np.linalg.norm(new_pts - pts, axis=1))
        pts = new_pts

        if max_disp < tol:
            logger.debug(f"Lloyd's converged after {iteration + 1} iterations (max_disp={max_disp:.4f})")
            break

    return pts


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
        # Clip vertices to image bounds if provided
        if image_shape is not None:
            H, W = image_shape
            verts = np.clip(verts, [0, 0], [H, W])

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
            vertices=verts,
        )

    return cells, junctions, graph


def voronoi_to_labels(
    positions: np.ndarray,
    image_shape: Tuple[int, int],
    method: VoronoiMethod = VoronoiMethod.STANDARD,
    lloyd_iterations: int = 10,
    lloyd_tol: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize a Voronoi tessellation into an integer label array.

    Each pixel is assigned to the nearest cell (1-indexed labels).
    Uses a KD-tree for efficient nearest-neighbor assignment.

    Args:
        positions: Nx2 array of (y, x) cell positions.
        image_shape: (H, W) of the output label array.
        method: VoronoiMethod.STANDARD or VoronoiMethod.LLOYD.
        lloyd_iterations: Max iterations for Lloyd's relaxation.
        lloyd_tol: Convergence tolerance for Lloyd's.

    Returns:
        (labels, final_positions) where labels is an (H, W) int32 array
        with cell IDs 1..N, and final_positions are the (possibly relaxed)
        seed positions.
    """
    _, final_positions = compute_voronoi(
        positions, image_shape=image_shape,
        method=method, lloyd_iterations=lloyd_iterations, lloyd_tol=lloyd_tol,
    )

    H, W = image_shape
    tree = cKDTree(final_positions)

    # Query all pixel coordinates at once
    yy, xx = np.mgrid[0:H, 0:W]
    pixel_coords = np.column_stack([yy.ravel(), xx.ravel()])
    _, nearest = tree.query(pixel_coords)

    # Reshape and convert to 1-indexed labels
    labels = (nearest.reshape(H, W) + 1).astype(np.int32)

    return labels, final_positions


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


def _polygon_centroid(vertices: np.ndarray) -> Optional[np.ndarray]:
    """Centroid of a simple polygon using the shoelace-derived formula.

    Returns (y, x) centroid, or None if degenerate.
    """
    n = len(vertices)
    if n < 3:
        return None
    y = vertices[:, 0]
    x = vertices[:, 1]
    # Signed area (2x)
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    signed_area = np.sum(cross)
    if abs(signed_area) < 1e-12:
        return np.mean(vertices, axis=0)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (3 * signed_area)
    cx = np.sum((x + np.roll(x, -1)) * cross) / (3 * signed_area)
    return np.array([cy, cx])
