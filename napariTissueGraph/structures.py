from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, FrozenSet
from enum import Enum
import numpy as np
import networkx as nx


class InputType(Enum):
    """How the graph was constructed."""
    VORONOI = "voronoi"
    SEGMENTATION = "segmentation"


@dataclass
class CellData:
    """Per-cell properties at a single timepoint."""
    cell_id: int
    position: np.ndarray        # (y, x) centroid
    area: float
    perimeter: float
    shape_index: float          # p0 = perimeter / sqrt(area)
    num_neighbors: int
    velocity: Optional[np.ndarray] = None
    instantaneous_speed: Optional[float] = None


@dataclass
class JunctionData:
    """Per-junction properties at a single timepoint."""
    cell_pair: Tuple[int, int]    # Sorted (smaller_id, larger_id)
    length: float
    coordinates: np.ndarray       # Nx2 array of (y,x) points along the junction
    midpoint: np.ndarray          # (y, x) midpoint of junction
    tension: Optional[float] = None
    normal_stress: Optional[float] = None


@dataclass
class T1Event:
    """A T1 transition (intercalation / neighbor exchange)."""
    frame: int
    losing_pair: Tuple[int, int]
    gaining_pair: Tuple[int, int]
    location: np.ndarray          # (y, x) approximate location
    all_cells: Set[int] = field(default_factory=set)


@dataclass
class EdgeTrajectory:
    """A junction tracked through time, potentially through T1 transitions.

    Sign convention: junction length is positive before a T1 (collapsing edge)
    and negative after (new edge growing). Each T1 flips the sign.
    """
    trajectory_id: int
    frames: List[int]
    cell_pairs: List[Tuple[int, int]]
    signed_lengths: List[float]
    coordinates: List[np.ndarray]
    t1_events: List[T1Event] = field(default_factory=list)


@dataclass
class TissueGraphFrame:
    """The tissue graph at a single timepoint."""
    frame: int
    graph: nx.Graph
    cells: Dict[int, CellData]
    junctions: Dict[FrozenSet[int], JunctionData]
    input_type: InputType


@dataclass
class TissueGraphTimeSeries:
    """The complete tissue graph over time."""
    frames: Dict[int, TissueGraphFrame]
    edge_trajectories: Dict[int, EdgeTrajectory] = field(default_factory=dict)
    t1_events: List[T1Event] = field(default_factory=list)
    pixel_size: Optional[float] = None
    time_interval: Optional[float] = None
    input_type: Optional[InputType] = None

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def frame_indices(self) -> List[int]:
        return sorted(self.frames.keys())
