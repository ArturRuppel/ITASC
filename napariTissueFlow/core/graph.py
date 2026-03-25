"""High-level API for building TissueGraphTimeSeries and TissueGraphDataset."""
import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Union

from ..structures import (
    InputType,
    TissueGraphDataset,
    TissueGraphFrame,
    TissueGraphTimeSeries,
    VoronoiMethod,
)
from .voronoi import compute_voronoi, voronoi_to_graph
from .labels import labels_to_graph
from .label_tracking import assign_track_ids, label_to_vertices
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Stage 1: Graph extraction (no tracking)
# ------------------------------------------------------------------

def extract_graphs_from_labels(
    label_stack: np.ndarray,
    pixel_size: Optional[float] = None,
    time_interval: Optional[float] = None,
    dilation_radius: int = 1,
    min_overlap_pixels: int = 5,
    min_edge_length: float = 0.0,
    filter_isolated: bool = True,
    min_border_edge_length: float = 5.0,
    min_bg_hole_size: int = 500,
) -> TissueGraphTimeSeries:
    """Extract per-frame graphs from segmentation labels (no tracking).

    All cell.track_id remain None. Call assign_tracking_labels() afterwards
    to add tracking as a separate stage.
    """
    frames_dict = {}

    for frame_idx in range(len(label_stack)):
        cells, junctions, graph = labels_to_graph(
            label_stack[frame_idx],
            dilation_radius=dilation_radius,
            min_overlap_pixels=min_overlap_pixels,
            min_edge_length=min_edge_length,
            filter_isolated=filter_isolated,
            min_border_edge_length=min_border_edge_length,
            min_bg_hole_size=min_bg_hole_size,
        )

        # Extract vertices for each cell (no track assignment)
        for cell_id, cell in cells.items():
            verts = label_to_vertices(label_stack[frame_idx], cell_id)
            if verts is not None:
                cell.vertices = verts

        frames_dict[frame_idx] = TissueGraphFrame(
            frame=frame_idx,
            graph=graph,
            cells=cells,
            junctions=junctions,
            input_type=InputType.SEGMENTATION,
        )

    return TissueGraphTimeSeries(
        frames=frames_dict,
        pixel_size=pixel_size,
        time_interval=time_interval,
        input_type=InputType.SEGMENTATION,
    )


def extract_graphs_from_tracks(
    positions: np.ndarray,
    pixel_size: Optional[float] = None,
    time_interval: Optional[float] = None,
    image_shape: Optional[tuple] = None,
    method: VoronoiMethod = VoronoiMethod.STANDARD,
    lloyd_iterations: int = 10,
    lloyd_tol: float = 0.1,
) -> TissueGraphTimeSeries:
    """Extract per-frame graphs from nuclear positions (no tracking).

    All cell.track_id remain None. Use apply_track_map() afterwards
    to add tracking as a separate stage.
    """
    frames_dict = {}
    frame_indices = np.unique(positions[:, 0].astype(int))

    for frame_idx in frame_indices:
        mask = positions[:, 0].astype(int) == frame_idx
        pts = positions[mask, 1:3]

        n_real = len(pts)
        vor, pts = compute_voronoi(
            pts, image_shape=image_shape,
            method=method, lloyd_iterations=lloyd_iterations, lloyd_tol=lloyd_tol,
        )
        cells, junctions, graph = voronoi_to_graph(
            vor, pts, n_real, image_shape=image_shape
        )

        frames_dict[int(frame_idx)] = TissueGraphFrame(
            frame=int(frame_idx),
            graph=graph,
            cells=cells,
            junctions=junctions,
            input_type=InputType.VORONOI,
        )

    return TissueGraphTimeSeries(
        frames=frames_dict,
        pixel_size=pixel_size,
        time_interval=time_interval,
        input_type=InputType.VORONOI,
    )


# ------------------------------------------------------------------
# Stage 2: Cell tracking
# ------------------------------------------------------------------

def assign_tracking_labels(
    series: TissueGraphTimeSeries,
    label_stack: np.ndarray,
    min_iou: float = 0.3,
    max_area_change: float = float('inf'),
) -> None:
    """Assign track IDs to cells via IoU matching on the label stack.

    Mutates series in place: sets cell.track_id for all matched cells.
    """
    track_assignments = assign_track_ids(
        label_stack, min_iou=min_iou, max_area_change=max_area_change,
    )

    for frame_idx, frame in series.frames.items():
        frame_tracks = track_assignments.get(frame_idx, {})
        for cell_id, cell in frame.cells.items():
            if cell_id in frame_tracks:
                cell.track_id = frame_tracks[cell_id]


def apply_track_map(
    series: TissueGraphTimeSeries,
    track_map: Dict[int, Dict[int, int]],
) -> None:
    """Apply a pre-computed track map to a series.

    Sets cell.track_id = track_map[frame_idx][cell_id] for each cell
    where a mapping exists. Mutates series in place.

    Args:
        series: The time series to update.
        track_map: Dict of {frame_idx: {cell_id: track_id}}.
    """
    for frame_idx, frame in series.frames.items():
        frame_tracks = track_map.get(frame_idx, {})
        for cell_id, cell in frame.cells.items():
            if cell_id in frame_tracks:
                cell.track_id = frame_tracks[cell_id]


def has_tracking(series: TissueGraphTimeSeries) -> bool:
    """Return True if any cell in any frame has a track_id assigned."""
    for frame in series.frames.values():
        for cell in frame.cells.values():
            if cell.track_id is not None:
                return True
    return False


# ------------------------------------------------------------------
# Monolithic build functions (Stage 1 + Stage 2 combined)
# ------------------------------------------------------------------

def build_from_tracks(
    positions: np.ndarray,
    pixel_size: Optional[float] = None,
    time_interval: Optional[float] = None,
    image_shape: Optional[tuple] = None,
    track_ids: Optional[Dict[int, Dict[int, int]]] = None,
    method: VoronoiMethod = VoronoiMethod.STANDARD,
    lloyd_iterations: int = 10,
    lloyd_tol: float = 0.1,
) -> TissueGraphTimeSeries:
    """Build tissue graph time series from nuclear tracking data.

    Args:
        positions: Shape (N, 3) array with columns (frame, y, x).
        pixel_size: Optional µm/pixel calibration.
        time_interval: Optional seconds between frames.
        image_shape: Optional (H, W) for bounding the Voronoi tessellation.
        track_ids: Optional dict frame -> {cell_index -> track_id}.
        method: Voronoi tessellation method.
        lloyd_iterations: Max iterations for Lloyd's relaxation.
        lloyd_tol: Convergence tolerance for Lloyd's.
    """
    series = extract_graphs_from_tracks(
        positions,
        pixel_size=pixel_size,
        time_interval=time_interval,
        image_shape=image_shape,
        method=method,
        lloyd_iterations=lloyd_iterations,
        lloyd_tol=lloyd_tol,
    )

    # Assign track IDs if provided
    if track_ids is not None:
        for frame_idx, frame in series.frames.items():
            if frame_idx in track_ids:
                frame_tracks = track_ids[frame_idx]
                for cell_id, cell in frame.cells.items():
                    if cell_id in frame_tracks:
                        cell.track_id = frame_tracks[cell_id]

    return series


def build_from_labels(
    label_stack: np.ndarray,
    pixel_size: Optional[float] = None,
    time_interval: Optional[float] = None,
    dilation_radius: int = 1,
    min_overlap_pixels: int = 5,
    min_edge_length: float = 0.0,
    filter_isolated: bool = True,
    min_border_edge_length: float = 5.0,
    min_bg_hole_size: int = 500,
    min_iou: float = 0.3,
    max_area_change: float = float('inf'),
) -> TissueGraphTimeSeries:
    """Build tissue graph time series from segmentation labels.

    Args:
        label_stack: Shape (T, H, W) integer labels, 0 = background.
        pixel_size: Optional µm/pixel calibration.
        time_interval: Optional seconds between frames.
        dilation_radius: Radius for dilation when detecting adjacency.
        min_overlap_pixels: Minimum boundary pixels for adjacency.
        min_edge_length: Minimum junction length to keep.
        filter_isolated: Remove edges where either cell has only one neighbor.
        min_border_edge_length: Minimum length for border boundary segments.
        min_iou: Minimum IoU threshold for label tracking.
        max_area_change: Max area ratio for label matching (inf = no limit).
    """
    series = extract_graphs_from_labels(
        label_stack,
        pixel_size=pixel_size,
        time_interval=time_interval,
        dilation_radius=dilation_radius,
        min_overlap_pixels=min_overlap_pixels,
        min_edge_length=min_edge_length,
        filter_isolated=filter_isolated,
        min_border_edge_length=min_border_edge_length,
        min_bg_hole_size=min_bg_hole_size,
    )
    assign_tracking_labels(
        series, label_stack, min_iou=min_iou, max_area_change=max_area_change,
    )
    return series


def build_from_labels_4d(
    label_stacks: Union[np.ndarray, List[np.ndarray]],
    pixel_size: Optional[float] = None,
    time_interval: Optional[float] = None,
    condition: str = "",
    progress_callback: Optional[Callable] = None,
    **kwargs,
) -> TissueGraphDataset:
    """Build dataset from multiple label stacks (multiple tissues).

    Args:
        label_stacks: Either a 4D array (N_tissues, T, H, W) or a list of
            3D arrays, each with shape (T_i, H, W). Using a list allows
            tissues with different numbers of frames.
        pixel_size: Physical pixel size in µm.
        time_interval: Time between frames in seconds.
        condition: Experimental condition label.
        progress_callback: Optional callback(progress_fraction, message).
        **kwargs: Passed to build_from_labels (dilation_radius, min_iou, etc.).
    """
    if isinstance(label_stacks, np.ndarray):
        if label_stacks.ndim != 4:
            raise ValueError(
                f"Expected 4D array (N_tissues, T, H, W), got {label_stacks.ndim}D"
            )
        stacks = [label_stacks[i] for i in range(label_stacks.shape[0])]
    else:
        stacks = list(label_stacks)
        for i, s in enumerate(stacks):
            if s.ndim != 3:
                raise ValueError(
                    f"Each label stack must be 3D (T, H, W), but stack {i} is {s.ndim}D"
                )

    dataset = TissueGraphDataset(
        tissues={},
        condition=condition,
        pixel_size=pixel_size,
        time_interval=time_interval,
        input_type=InputType.SEGMENTATION,
    )
    n_tissues = len(stacks)
    for i, stack in enumerate(stacks):
        if progress_callback:
            progress_callback(i / n_tissues, f"Processing tissue {i + 1}/{n_tissues}")
        series = build_from_labels(
            stack,
            pixel_size=pixel_size,
            time_interval=time_interval,
            **kwargs,
        )
        dataset.add_tissue(series)

    logger.info(f"Built dataset with {n_tissues} tissues from labels")
    return dataset


def build_from_tracks_4d(
    positions: np.ndarray,
    pixel_size: Optional[float] = None,
    time_interval: Optional[float] = None,
    condition: str = "",
    image_shape: Optional[tuple] = None,
    progress_callback: Optional[Callable] = None,
    method: VoronoiMethod = VoronoiMethod.STANDARD,
    lloyd_iterations: int = 10,
    lloyd_tol: float = 0.1,
) -> TissueGraphDataset:
    """Build dataset from tracked nuclear positions across multiple tissues.

    Args:
        positions: Nx4 array with columns (tissue_id, frame, y, x).
        pixel_size: Physical pixel size in µm.
        time_interval: Time between frames in seconds.
        condition: Experimental condition label.
        image_shape: Optional (H, W) for bounding the Voronoi tessellation.
        progress_callback: Optional callback(progress_fraction, message).
        method: Voronoi tessellation method.
        lloyd_iterations: Max iterations for Lloyd's relaxation.
        lloyd_tol: Convergence tolerance for Lloyd's.
    """
    dataset = TissueGraphDataset(
        tissues={},
        condition=condition,
        pixel_size=pixel_size,
        time_interval=time_interval,
        input_type=InputType.VORONOI,
    )
    tissue_ids = np.unique(positions[:, 0].astype(int))
    n_tissues = len(tissue_ids)
    for idx, tid in enumerate(tissue_ids):
        if progress_callback:
            progress_callback(idx / n_tissues, f"Processing tissue {idx + 1}/{n_tissues}")
        mask = positions[:, 0].astype(int) == tid
        pos_i = positions[mask, 1:]  # (frame, y, x)
        series = build_from_tracks(
            pos_i, pixel_size, time_interval, image_shape,
            method=method, lloyd_iterations=lloyd_iterations, lloyd_tol=lloyd_tol,
        )
        dataset.add_tissue(series)

    logger.info(f"Built dataset with {n_tissues} tissues from tracks")
    return dataset


