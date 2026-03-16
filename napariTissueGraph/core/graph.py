"""High-level API for building TissueGraphTimeSeries."""
import logging
import numpy as np
from typing import Optional

from ..structures import (
    InputType,
    TissueGraphFrame,
    TissueGraphTimeSeries,
)
from .voronoi import compute_voronoi, voronoi_to_graph
from .labels import labels_to_graph

logger = logging.getLogger(__name__)


def build_from_tracks(
    positions: np.ndarray,
    pixel_size: Optional[float] = None,
    time_interval: Optional[float] = None,
    image_shape: Optional[tuple] = None,
) -> TissueGraphTimeSeries:
    """Build tissue graph time series from nuclear tracking data.

    Args:
        positions: Shape (N, 3) array with columns (frame, y, x).
        pixel_size: Optional µm/pixel calibration.
        time_interval: Optional seconds between frames.
        image_shape: Optional (H, W) for bounding the Voronoi tessellation.
    """
    frames_dict = {}
    frame_indices = np.unique(positions[:, 0].astype(int))

    for frame_idx in frame_indices:
        mask = positions[:, 0].astype(int) == frame_idx
        pts = positions[mask, 1:3]  # (y, x)

        n_real = len(pts)
        vor = compute_voronoi(pts, image_shape=image_shape)
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


def build_from_labels(
    label_stack: np.ndarray,
    pixel_size: Optional[float] = None,
    time_interval: Optional[float] = None,
    dilation_radius: int = 1,
    min_overlap_pixels: int = 5,
    min_edge_length: float = 0.0,
    filter_isolated: bool = True,
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
    """
    frames_dict = {}

    for frame_idx in range(len(label_stack)):
        cells, junctions, graph = labels_to_graph(
            label_stack[frame_idx],
            dilation_radius=dilation_radius,
            min_overlap_pixels=min_overlap_pixels,
            min_edge_length=min_edge_length,
            filter_isolated=filter_isolated,
        )

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
