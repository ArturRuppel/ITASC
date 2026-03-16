# napariTissueGraph — Implementation Plan for Claude Code

## Project Overview

**napariTissueGraph** is a napari plugin for quantifying tissue topology, rearrangement dynamics, and junction mechanics from microscopy data. It replaces and extends napariCellFlow.

The tool takes cell positions over time (from nuclear tracking or segmentation labels) and builds a dynamic graph representation of the tissue. It tracks junction lengths, detects T1 transitions (cell intercalation events), computes cell-level statistics, and optionally maps traction force data onto junction edges.

### Key Design Principles

1. **Layered architecture**: Each layer builds on the previous. Users get analysis depth proportional to their input data.
2. **External tools for segmentation and tracking**: This tool does NOT do segmentation or tracking. It accepts their outputs.
3. **TFM/MSM is optional**: Force mapping enriches the analysis but is never required.
4. **Two input paths, one internal representation**: Segmentation labels and nuclear tracks both produce the same `TissueGraph` object.
5. **Separation of core logic and GUI**: All analysis code must work without napari. The napari widget is a thin visualization/interaction layer on top.

---

## Architecture

### Layer Model

```
Layer 1: Cell positions over time
  Input: nuclear tracks (N×3 array: frame, y, x) OR segmentation labels (T×H×W array)
  Output: cell positions, velocities, MSD, pairwise distances
  → Available from both input types

Layer 2: Cell graph (Voronoi or segmentation-derived)
  Input: Layer 1 positions OR segmentation labels
  Output: TissueGraph — nodes (cells) with area/shape index, edges (junctions) with lengths/coordinates
  → Available from both input types, but segmentation gives real boundaries

Layer 3: Topology dynamics  
  Input: TissueGraph time series from Layer 2
  Output: T1 events, neighbor exchange tracking, edge trajectories with sign convention
  → Requires Layer 2

Layer 4 (optional): Mechanics
  Input: Layer 2 graph + external TFM/MSM force data
  Output: Junction tensions, force-topology correlations
  → Requires Layer 2 + external force data (e.g., from napariTFM)
```

### Directory Structure

```
napariTissueGraph/
├── napariTissueGraph/
│   ├── __init__.py
│   ├── napari.yaml                # napari manifest
│   ├── core/
│   │   ├── __init__.py
│   │   ├── graph.py               # TissueGraph class — central data structure
│   │   ├── voronoi.py             # Nuclear positions → Voronoi tessellation → graph
│   │   ├── labels.py              # Segmentation labels → graph (adapted from napariCellFlow)
│   │   ├── tracks.py              # Cell track analysis (Layer 1: MSD, velocities, etc.)
│   │   ├── topology.py            # T1 detection, neighbor exchange tracking (Layer 3)
│   │   ├── mechanics.py           # Optional TFM/MSM force mapping (Layer 4)
│   │   └── io.py                  # Import/export, napariTFM interop
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── trajectories.py        # Edge trajectory construction with sign convention
│   │   ├── statistics.py          # Shape index (p0), T1 rates, spatial correlations
│   │   └── events.py              # Event-triggered averaging around T1s
│   ├── napari/
│   │   ├── __init__.py
│   │   ├── widget.py              # Main napari dock widget
│   │   └── visualization.py       # Napari layer rendering (graph overlay, T1 markers, etc.)
│   └── structures.py              # All dataclasses and type definitions
├── tests/
│   ├── __init__.py
│   ├── test_graph.py
│   ├── test_voronoi.py
│   ├── test_labels.py
│   ├── test_topology.py
│   ├── test_tracks.py
│   └── conftest.py                # Shared fixtures (synthetic data generators)
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Data Structures (`structures.py`)

Define all dataclasses here. These are the contracts between modules.

```python
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, FrozenSet
from enum import Enum
import numpy as np
import networkx as nx


class InputType(Enum):
    """How the graph was constructed."""
    VORONOI = "voronoi"          # From nuclear positions
    SEGMENTATION = "segmentation"  # From label masks


@dataclass
class CellData:
    """Per-cell properties at a single timepoint."""
    cell_id: int
    position: np.ndarray        # (y, x) centroid
    area: float                 # Cell area (Voronoi polygon or segmentation region)
    perimeter: float            # Cell perimeter
    shape_index: float          # p0 = perimeter / sqrt(area)
    num_neighbors: int          # Number of adjacent cells
    # Optional attributes populated by different layers
    velocity: Optional[np.ndarray] = None          # (vy, vx), from Layer 1
    instantaneous_speed: Optional[float] = None     # |v|, from Layer 1


@dataclass  
class JunctionData:
    """Per-junction properties at a single timepoint."""
    cell_pair: Tuple[int, int]    # Sorted (smaller_id, larger_id)
    length: float                 # Junction length in pixels (or physical units if calibrated)
    coordinates: np.ndarray       # Nx2 array of (y,x) points along the junction
    midpoint: np.ndarray          # (y, x) midpoint of junction
    # Optional: from Layer 4
    tension: Optional[float] = None
    normal_stress: Optional[float] = None


@dataclass
class T1Event:
    """A T1 transition (intercalation / neighbor exchange)."""
    frame: int                            # Frame where transition occurs
    losing_pair: Tuple[int, int]          # Cell pair that loses contact
    gaining_pair: Tuple[int, int]         # Cell pair that gains contact
    location: np.ndarray                  # (y, x) approximate location of event
    all_cells: Set[int] = field(default_factory=set)  # The 4 cells involved


@dataclass
class EdgeTrajectory:
    """A junction tracked through time, potentially through T1 transitions.
    
    Sign convention: junction length is positive before a T1 (collapsing edge)
    and negative after (new edge growing). Each T1 flips the sign.
    This gives a continuous signed length trajectory across topology changes.
    """
    trajectory_id: int
    frames: List[int]                          
    cell_pairs: List[Tuple[int, int]]          # May change across T1s
    signed_lengths: List[float]                # Positive → negative at T1
    coordinates: List[np.ndarray]
    t1_events: List[T1Event] = field(default_factory=list)


@dataclass
class TissueGraphFrame:
    """The tissue graph at a single timepoint."""
    frame: int
    graph: nx.Graph                  # Nodes = cell_ids, edges = junctions
    cells: Dict[int, CellData]       # cell_id → CellData
    junctions: Dict[FrozenSet[int], JunctionData]  # frozenset(cell_pair) → JunctionData
    input_type: InputType


@dataclass
class TissueGraphTimeSeries:
    """The complete tissue graph over time. This is the central data object."""
    frames: Dict[int, TissueGraphFrame]     # frame_index → TissueGraphFrame
    edge_trajectories: Dict[int, EdgeTrajectory] = field(default_factory=dict)
    t1_events: List[T1Event] = field(default_factory=list)
    # Metadata
    pixel_size: Optional[float] = None       # µm/pixel
    time_interval: Optional[float] = None    # seconds between frames
    input_type: Optional[InputType] = None
    
    @property
    def num_frames(self) -> int:
        return len(self.frames)
    
    @property  
    def frame_indices(self) -> List[int]:
        return sorted(self.frames.keys())
```

---

## Module Specifications

### `core/graph.py` — TissueGraph construction coordinator

This module provides the high-level API for building a `TissueGraphTimeSeries`. It delegates to `voronoi.py` or `labels.py` depending on input type.

```python
def build_from_tracks(
    positions: np.ndarray,          # Shape (N, 3): columns are (frame, y, x)
    pixel_size: Optional[float] = None,
    time_interval: Optional[float] = None,
    boundary_mask: Optional[np.ndarray] = None,  # Optional: confine Voronoi to a region
) -> TissueGraphTimeSeries:
    """Build tissue graph time series from nuclear tracking data.
    
    For each frame:
    1. Extract positions for that frame
    2. Compute Voronoi tessellation (voronoi.py)
    3. Build graph with cell and junction properties
    4. Optionally clip Voronoi to boundary_mask (e.g., circular micropattern)
    """

def build_from_labels(
    label_stack: np.ndarray,        # Shape (T, H, W): integer labels, 0 = background
    pixel_size: Optional[float] = None,
    time_interval: Optional[float] = None,
) -> TissueGraphTimeSeries:
    """Build tissue graph time series from segmentation labels.
    
    For each frame:
    1. Extract cell regions from labels (labels.py)
    2. Build graph with cell and junction properties
    3. Compute cell centroids for Layer 1 compatibility
    """
```

### `core/voronoi.py` — Voronoi tessellation from positions

Converts a set of 2D positions into a Voronoi tessellation and extracts the graph structure.

Key functions:
- `compute_voronoi(positions: np.ndarray, boundary_mask: Optional[np.ndarray]) -> VoronoiResult`
  - Uses `scipy.spatial.Voronoi`
  - Clips infinite regions to boundary (image bounds or mask)
  - Returns vertices, regions, ridge_points, ridge_vertices
- `voronoi_to_graph(voronoi_result, positions) -> Tuple[Dict[int, CellData], Dict[FrozenSet, JunctionData]]`
  - Extracts cell areas, perimeters, shape indices from Voronoi polygons
  - Extracts junction lengths and coordinates from Voronoi ridges
  - Builds the cell-junction mappings

**Important**: Handle boundary cells carefully. Voronoi cells at the edge of the pattern will have infinite vertices. Either clip to a provided boundary mask or exclude boundary cells from analysis (flag them).

Use `scipy.spatial.Voronoi` for the tessellation. For clipping to a boundary, use `shapely` for polygon intersection if available, otherwise implement a simple convex hull clip.

### `core/labels.py` — Graph extraction from segmentation labels

Adapted from napariCellFlow's `edge_analysis.py` `_detect_edges` and `_find_shared_boundary` methods.

Key functions:
- `labels_to_graph(label_frame: np.ndarray, dilation_radius: int = 1, min_overlap_pixels: int = 5) -> Tuple[Dict[int, CellData], Dict[FrozenSet, JunctionData]]`
  - For each cell: compute centroid, area, perimeter from regionprops
  - For each cell pair: detect shared boundary using dilation + overlap (as in napariCellFlow)
  - Skeletonize boundary, order pixels, compute length
  - Return cell and junction data

**Reuse logic from**: `napariCellFlow/edge_analysis.py` methods `_detect_edges`, `_find_shared_boundary`, `_order_boundary_pixels`, `_calculate_edge_length`. These are well-tested and work correctly. Refactor them into standalone functions (remove class dependency).

### `core/tracks.py` — Cell-level track analysis (Layer 1)

Operates on cell positions over time regardless of how they were obtained.

Key functions:
- `compute_velocities(positions_by_cell: Dict[int, np.ndarray], dt: float) -> Dict[int, np.ndarray]`
  - Finite differences of position time series
- `compute_msd(positions_by_cell: Dict[int, np.ndarray], max_lag: int) -> np.ndarray`
  - Mean squared displacement vs. lag time
- `compute_velocity_correlation(graph_series: TissueGraphTimeSeries) -> np.ndarray`
  - Spatial velocity-velocity correlation function
- `extract_positions_from_series(series: TissueGraphTimeSeries) -> Dict[int, np.ndarray]`
  - Helper to pull position time series from the graph object

### `core/topology.py` — T1 detection (Layer 3)

Adapted from napariCellFlow's T1 detection logic but operating on `TissueGraphFrame` objects instead of raw label arrays.

Key functions:
- `detect_t1_events(series: TissueGraphTimeSeries) -> List[T1Event]`
  - Compare graph topology between consecutive frames
  - For each lost edge + gained edge pair, validate as T1 (4 unique cells, connecting edges persist)
  - Return validated T1 events
- `build_edge_trajectories(series: TissueGraphTimeSeries, t1_events: List[T1Event]) -> Dict[int, EdgeTrajectory]`
  - Track junction identity through time
  - Link junctions across T1 events with sign-flipping convention
  - Adapted from napariCellFlow's `_create_edge_trajectories`

**Reuse logic from**: `napariCellFlow/edge_analysis.py` methods `_detect_topology_changes`, `_validate_t1_transition`, `_create_edge_trajectories`. The core logic is correct — refactor from operating on NetworkX graphs of raw boundaries to operating on `TissueGraphFrame.graph` objects.

### `core/mechanics.py` — Force mapping (Layer 4, optional)

Maps external force data onto the tissue graph.

Key functions:
- `map_traction_forces(series: TissueGraphTimeSeries, traction_fields: np.ndarray, pixel_size: float) -> None`
  - For each junction, integrate traction forces perpendicular to the junction
  - Assigns `tension` attribute to `JunctionData`
  - traction_fields shape: (T, 2, H, W) — x and y components
- `map_msm_stress(series: TissueGraphTimeSeries, stress_tensor: np.ndarray) -> None`
  - For each junction, project the stress tensor along the junction normal
  - stress_tensor shape: (T, 3, H, W) — σxx, σyy, σxy components

### `core/io.py` — Import/export

Key functions:
- `save_results(series: TissueGraphTimeSeries, path: Path) -> None`
  - Save to JSON (metadata + edge trajectories + T1 events) + NPY (heavy arrays)
- `load_results(path: Path) -> TissueGraphTimeSeries`
- `import_naparitfm(tfm_path: Path) -> np.ndarray`
  - Load traction field data exported from napariTFM
- `export_for_plotting(series: TissueGraphTimeSeries) -> Dict`
  - Export analysis results as simple dicts/arrays for matplotlib

### `analysis/trajectories.py` — Edge trajectory analysis

Operates on `EdgeTrajectory` objects.

Key functions:
- `get_t1_trajectories(series: TissueGraphTimeSeries) -> List[EdgeTrajectory]`
  - Filter to trajectories that contain T1 events
- `get_stable_trajectories(series: TissueGraphTimeSeries, min_frames: int) -> List[EdgeTrajectory]`
  - Filter to long-lived junctions without T1s
- `compute_length_fluctuations(trajectory: EdgeTrajectory) -> Dict`
  - Mean, std, autocorrelation of junction length

### `analysis/statistics.py` — Tissue-level statistics

Key functions:
- `compute_shape_index_distribution(frame: TissueGraphFrame) -> np.ndarray`
  - Distribution of p0 = perimeter / sqrt(area) for all cells
- `compute_t1_rate(series: TissueGraphTimeSeries) -> float`
  - T1 events per cell per unit time
- `compute_cell_shape_parameter(frame: TissueGraphFrame) -> float`
  - Mean shape index — relates to jamming transition (p0* ≈ 3.81)
- `compute_neighbor_number_distribution(frame: TissueGraphFrame) -> np.ndarray`
  - Distribution of coordination number

### `analysis/events.py` — Event-triggered analysis

Key functions:
- `average_around_t1(series: TissueGraphTimeSeries, quantity: str, window: int) -> np.ndarray`
  - Event-triggered average of a quantity (junction length, tension, etc.) around T1 events
  - Aligns all T1 events at t=0 and averages
- `spatial_average_around_t1(series: TissueGraphTimeSeries, quantity: str, window: int, radius: float) -> np.ndarray`
  - Same but for spatially resolved quantities (e.g., how does traction force change near a T1?)

---

## Napari Widget (`napari/widget.py`)

Simple tabbed widget with minimal UI. NO preprocessing, NO segmentation, NO tracking.

### Tab 1: Data Input
- Dropdown to select input type: "Nuclear Tracks" or "Segmentation Labels"
- For nuclear tracks: select a napari Points layer (tracked points with frame column)
- For segmentation labels: select a napari Labels layer
- Optional: load TFM data from file (button to browse for napariTFM output)
- Optional: set pixel size (µm/px) and time interval (s/frame)
- Optional: load or draw boundary mask (for Voronoi clipping)
- "Build Graph" button → runs `build_from_tracks` or `build_from_labels`

### Tab 2: Visualization
- Toggle overlays: Voronoi/junction edges, cell IDs, T1 event markers
- Color junctions by: length, tension (if available), trajectory ID
- Color cells by: area, shape index, number of neighbors
- Frame slider synced with napari dimension slider
- Junction network rendered as a napari Shapes layer (lines)
- T1 events rendered as a napari Points layer (markers at T1 locations)

### Tab 3: Analysis
- Button: "Detect T1 Events" → runs topology detection
- Button: "Build Edge Trajectories" → runs trajectory construction
- Button: "Compute Statistics" → runs all statistics, shows summary
- Display: T1 count, mean T1 rate, mean shape index
- Button: "Export Results" → saves to JSON + NPY

### Tab 4: Plots (optional, can be simple)
- Junction length time series (select trajectory by clicking)
- Shape index distribution histogram
- T1 rate over time
- Event-triggered average of junction length around T1s
- Use matplotlib embedded in Qt (FigureCanvasQTAgg)

---

## Napari Visualization (`napari/visualization.py`)

Key functions:
- `draw_junction_network(viewer, frame: TissueGraphFrame, color_by: str) -> napari.layers.Shapes`
  - Draw all junctions as line segments on a Shapes layer
  - Color by junction property (length, tension, etc.)
- `draw_cell_overlay(viewer, frame: TissueGraphFrame, color_by: str) -> napari.layers.Labels`
  - Color cells by property (shape index, area, etc.)
- `mark_t1_events(viewer, events: List[T1Event], current_frame: int) -> napari.layers.Points`
  - Show T1 event locations as markers
- `draw_voronoi_tessellation(viewer, frame: TissueGraphFrame) -> napari.layers.Shapes`
  - Draw full Voronoi tessellation as polygon outlines

---

## What to Reuse from napariCellFlow

### Reuse directly (refactor into functions):
1. **`edge_analysis.py` → `_find_shared_boundary`**: Boundary detection between labeled cells via dilation + skeletonization. Move to `core/labels.py`.
2. **`edge_analysis.py` → `_order_boundary_pixels`**: Pixel ordering along skeleton. Move to `core/labels.py`.
3. **`edge_analysis.py` → `_calculate_edge_length`**: Euclidean path length. Move to `core/labels.py`.
4. **`edge_analysis.py` → `_validate_t1_transition`**: T1 validation logic (4 unique cells, connecting edges). Move to `core/topology.py`.
5. **`edge_analysis.py` → `_detect_topology_changes`**: Frame-to-frame graph comparison for T1 detection. Move to `core/topology.py`.
6. **`edge_analysis.py` → `_create_edge_trajectories`**: Trajectory construction with sign convention. Move to `analysis/trajectories.py`.
7. **`structure.py` → dataclass patterns**: `CellBoundary`, `IntercalationEvent`, `EdgeData` patterns inform the new `structures.py`.

### Do NOT reuse:
1. **`preprocessing.py` and `preprocessing_widget.py`** — Out of scope.
2. **`segmentation.py` and `segmentation_widget.py`** — Cellpose wrapper, out of scope.
3. **`cell_tracking.py` and `cell_tracking_widget.py`** — Tracking is external.
4. **`cell_correction_widget.py`** — Manual correction UI, out of scope.
5. **`data_manager.py`** — Over-engineered for new architecture. Replace with simpler I/O.
6. **`visualization_manager.py`** — Too tightly coupled to old widget. Rewrite for new visualization needs.
7. **`error_handling.py`** — Custom error framework, unnecessary complexity.

---

## Dependencies

```toml
[project]
name = "napariTissueGraph"
version = "0.1.0"
description = "A napari plugin for quantifying tissue topology, rearrangement dynamics, and junction mechanics"
requires-python = ">=3.9"
license = {text = "GPL-3.0"}
authors = [
    {name = "Artur Ruppel"}
]

dependencies = [
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "scikit-image>=0.20.0",
    "networkx>=3.0",
    "napari>=0.4.18",
    "qtpy>=2.3.0",
    "matplotlib>=3.7.0",
    "opencv-python>=4.7.0",
    "tifffile>=2023.1.0",
]

[project.optional-dependencies]
shapely = ["shapely>=2.0.0"]  # For Voronoi clipping to arbitrary boundaries
dev = [
    "pytest>=7.0.0",
    "pytest-qt>=4.2.0",
]

[project.entry-points."napari.manifest"]
napariTissueGraph = "napariTissueGraph:napari.yaml"
```

Note: no cellpose dependency. No tqdm. Minimal footprint.

---

## Implementation Order

### Phase 1: Core data structures and graph building
1. `structures.py` — all dataclasses
2. `core/voronoi.py` — Voronoi tessellation from positions
3. `core/labels.py` — Graph from segmentation labels (port from napariCellFlow)
4. `core/graph.py` — High-level `build_from_tracks` and `build_from_labels`
5. Tests for all of the above with synthetic data

### Phase 2: Topology analysis
6. `core/topology.py` — T1 detection (port from napariCellFlow)
7. `analysis/trajectories.py` — Edge trajectory construction (port from napariCellFlow)
8. Tests with synthetic T1 scenarios

### Phase 3: Cell-level analysis
9. `core/tracks.py` — MSD, velocities from positions
10. `analysis/statistics.py` — Shape index, T1 rates, distributions
11. `analysis/events.py` — Event-triggered averaging

### Phase 4: Napari integration
12. `napari/visualization.py` — Layer rendering functions
13. `napari/widget.py` — Dock widget with tabs
14. `napari.yaml` — Plugin manifest

### Phase 5: Force mapping and I/O
15. `core/mechanics.py` — TFM/MSM mapping
16. `core/io.py` — Save/load, napariTFM interop

---

## Testing Strategy

### Synthetic data generators (in `conftest.py`):
- `make_grid_positions(nx, ny, noise)` — Regular grid of points with optional jitter, for Voronoi testing
- `make_label_frame(n_cells, image_size)` — Random Voronoi-based label mask
- `make_t1_sequence(n_frames)` — Synthetic label stack where a known T1 event occurs at a specific frame
- `make_circular_monolayer(n_cells, radius)` — Points arranged in a circular boundary

### Key test cases:
- Voronoi from regular grid → all cells hexagonal, all junctions equal length
- Voronoi from grid with one displaced cell → verify junction changes
- Labels from known geometry → verify cell areas, junction lengths match expected
- T1 detection on synthetic 4-cell rosette → verify event detected at correct frame
- Edge trajectory through T1 → verify sign flip
- Round-trip: save → load → compare

---

## Style and Code Quality

- Type hints on all public functions
- Docstrings (NumPy style) on all public functions and classes
- No GUI imports in `core/` or `analysis/` modules
- `core/` and `analysis/` must be fully testable without napari running
- Use logging (not print) for debug output
- Keep functions short and focused — prefer composition over inheritance
