"""Track-dynamics compute core: motion read off tracked label stacks.

The headless backend for the dynamics quantifiers. A tracked label stack becomes
per-track centroid trajectories (:mod:`.trajectories`), then instantaneous
velocities + per-track motility summaries (:mod:`.kinematics`), an ensemble
mean-square-displacement curve with a power-law fit (:mod:`.msd`), and
tissue-scale collective metrics — alignment, velocity correlation, length scale
(:mod:`.collective`). :func:`build_track_dynamics` runs all of it and persists a
multi-table ``.h5``; :func:`read_track_dynamics` loads it back. No Qt / napari
import, so scripts and the standalone wheel can use it.
"""

from .collective import COLLECTIVE_COLUMNS, CORR_CURVE_COLUMNS, pooled_corr_length
from .kinematics import DAC_COLUMNS, INSTANTANEOUS_COLUMNS, TRACK_COLUMNS
from .msd import MSD_COLUMNS, MSD_TRACK_COLUMNS, per_track_msd_fit
from .store import (
    DEFAULT_PARAMS,
    TrackDynamics,
    build_track_dynamics,
    read_instantaneous_table,
    read_track_dynamics,
)
from .trajectories import Trajectory, extract_trajectories, trajectories_from_stack

__all__ = [
    "COLLECTIVE_COLUMNS",
    "CORR_CURVE_COLUMNS",
    "DAC_COLUMNS",
    "DEFAULT_PARAMS",
    "INSTANTANEOUS_COLUMNS",
    "MSD_COLUMNS",
    "MSD_TRACK_COLUMNS",
    "TRACK_COLUMNS",
    "TrackDynamics",
    "Trajectory",
    "build_track_dynamics",
    "extract_trajectories",
    "per_track_msd_fit",
    "pooled_corr_length",
    "read_instantaneous_table",
    "read_track_dynamics",
    "trajectories_from_stack",
]
