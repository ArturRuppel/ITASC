from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional, Set, FrozenSet
from enum import Enum
import numpy as np
import networkx as nx


class InputType(Enum):
    """How the graph was constructed."""
    VORONOI = "voronoi"
    SEGMENTATION = "segmentation"


class VoronoiMethod(Enum):
    """Voronoi tessellation method."""
    STANDARD = "standard"
    LLOYD = "lloyd"


@dataclass
class CellData:
    """Per-cell properties at a single timepoint."""
    cell_id: int
    position: np.ndarray        # (y, x) centroid
    area: float
    perimeter: float
    shape_index: float          # p0 = perimeter / sqrt(area)
    num_neighbors: int
    track_id: Optional[int] = None
    vertices: Optional[np.ndarray] = None  # Nx2 ordered polygon boundary
    velocity: Optional[np.ndarray] = None
    instantaneous_speed: Optional[float] = None
    pressure: Optional[float] = None


@dataclass
class JunctionData:
    """Per-junction properties at a single timepoint."""
    cell_pair: Tuple[int, int]    # Sorted (smaller_id, larger_id)
    length: float
    coordinates: np.ndarray       # Nx2 array of (y,x) points along the junction
    midpoint: np.ndarray          # (y, x) midpoint of junction
    tension: Optional[float] = None
    normal_stress: Optional[float] = None
    tags: Set[str] = field(default_factory=set)


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
    tags: Set[str] = field(default_factory=set)
    name: Optional[str] = None


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


@dataclass
class TissueGraphDataset:
    """Collection of tissue graph time series — the primary output of the plugin.

    Represents multiple tissues from one experimental condition.
    This is the object that gets saved to disk and loaded by analysis scripts.
    """
    tissues: Dict[int, TissueGraphTimeSeries] = field(default_factory=dict)

    condition: str = ""
    pixel_size: Optional[float] = None
    time_interval: Optional[float] = None
    input_type: Optional[InputType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_tissue(self, series: TissueGraphTimeSeries) -> int:
        """Add a tissue, returns assigned tissue_id."""
        tissue_id = max(self.tissues.keys(), default=-1) + 1
        self.tissues[tissue_id] = series
        return tissue_id

    def remove_tissue(self, tissue_id: int) -> None:
        """Remove a tissue (e.g., after QC inspection reveals bad data)."""
        del self.tissues[tissue_id]

    @property
    def n_tissues(self) -> int:
        return len(self.tissues)

    @property
    def tissue_ids(self) -> List[int]:
        return sorted(self.tissues.keys())

    def get_all_t1_events(self) -> List[Tuple[int, T1Event]]:
        """Get all T1 events across all tissues, tagged with tissue_id."""
        events = []
        for tid, series in self.tissues.items():
            for event in series.t1_events:
                events.append((tid, event))
        return events

    def get_all_edge_trajectories(self) -> List[Tuple[int, EdgeTrajectory]]:
        """Get all edge trajectories across all tissues, tagged with tissue_id."""
        trajectories = []
        for tid, series in self.tissues.items():
            for traj in series.edge_trajectories.values():
                trajectories.append((tid, traj))
        return trajectories
