"""Force inference API using ForSys.

Provides high-level functions that run ForSys on a TissueGraphTimeSeries
and write inferred tensions and pressures back into the data structures.
"""
import logging
from typing import Optional

import numpy as np

from .structures import TissueGraphTimeSeries
from ..backend.forsys import (
    forsys_available,
    tissue_frame_to_forsys,
    forsys_results_to_tissue,
)

logger = logging.getLogger(__name__)


def _require_forsys():
    if not forsys_available():
        raise ImportError(
            "forsys is not installed. Install it with: "
            "pip install cellflow[forces]"
        )


def infer_forces(
    series: TissueGraphTimeSeries,
    method: str = "static",
    vertices_per_edge: int = 6,
    endpoint_cluster_tol: float = 3.0,
    allow_negatives: bool = False,
) -> None:
    """Run force inference on a time series, writing results in place.

    Parameters
    ----------
    series : TissueGraphTimeSeries
        The tissue time series. Must have junctions with coordinates.
    method : str
        ``"static"`` runs the ForSys solver independently per frame.
        ``"dynamic"`` is reserved for future velocity-based inference.
    vertices_per_edge : int
        Target number of vertices per BigEdge in the ForSys mesh.
    endpoint_cluster_tol : float
        Distance tolerance (pixels) for clustering junction endpoints
        into triple junctions.
    allow_negatives : bool
        If False (default), ForSys constrains tensions to be non-negative.

    Raises
    ------
    ImportError
        If forsys is not installed.
    ValueError
        If method is not recognized.
    """
    _require_forsys()
    import forsys as fsys

    if method not in ("static", "dynamic"):
        raise ValueError(f"Unknown method: {method!r}. Use 'static' or 'dynamic'.")

    if method == "dynamic":
        raise NotImplementedError(
            "Dynamic (velocity-based) inference is not yet implemented. "
            "Use method='static'."
        )

    logger.info(f"Running {method} force inference on {series.num_frames} frames")

    for frame_idx in series.frame_indices:
        tissue_frame = series.frames[frame_idx]

        try:
            fs_frame = tissue_frame_to_forsys(
                tissue_frame,
                vertices_per_edge=vertices_per_edge,
                endpoint_cluster_tol=endpoint_cluster_tol,
            )
        except ValueError as e:
            logger.warning(f"Frame {frame_idx}: skipping — {e}")
            continue

        # Build and solve the ForSys system.
        # np.errstate suppresses ForSys's unclipped arccos (fmatrix.py:213)
        # which can produce NaN from dot products slightly outside [-1, 1].
        fs_obj = fsys.ForSys(frames={0: fs_frame})

        try:
            with np.errstate(invalid="ignore"):
                fs_obj.build_force_matrix(when=0)
                fs_obj.solve_stress(when=0, allow_negatives=allow_negatives)
        except Exception as e:
            logger.warning(f"Frame {frame_idx}: tension inference failed — {e}")
            continue

        try:
            with np.errstate(invalid="ignore"):
                fs_obj.build_pressure_matrix(when=0)
                fs_obj.solve_pressure(when=0, method="lagrange_pressure")
        except Exception as e:
            logger.warning(
                f"Frame {frame_idx}: pressure inference failed — {e}. "
                f"Tensions were still written."
            )
            # Still write back tensions even if pressure fails
            forsys_results_to_tissue(fs_frame, tissue_frame)
            continue

        forsys_results_to_tissue(fs_frame, tissue_frame)

    logger.info("Force inference complete")
