# CellFlow — Implementation Plan

## Project Overview

**CellFlow** is a napari plugin for quantifying tissue topology, rearrangement dynamics, and junction mechanics from microscopy data. It replaces and extends napariCellFlow.

The tool takes cell positions over time (from nuclear tracking or segmentation labels) and builds a dynamic graph representation of the tissue. It tracks junction lengths, detects T1 transitions (cell intercalation events), computes cell-level statistics, and optionally maps traction force data onto junction edges.

### Key Design Principles

1. **Layered architecture**: Each layer builds on the previous. Users get analysis depth proportional to their input data.
2. **External tools for segmentation**: This tool does NOT do segmentation. It accepts segmentation labels as input.
3. **Force inference is optional**: ForSys-based tension/pressure inference enriches the analysis but is never required.
4. **Separation of core logic and GUI**: All analysis code must work without napari. The napari widget is a thin visualization/interaction layer on top.
5. **Multi-tissue first**: The primary output is `TissueGraphDataset` (multiple tissues from one condition). Single-tissue is just a dataset with one entry.

---

## Architecture

### Layer Model

```
Layer 1: Cell positions over time
  Input: segmentation labels (T×H×W array)
  Output: cell positions, velocities, MSD, pairwise distances

Layer 2: Cell graph (segmentation-derived)
  Input: segmentation labels
  Output: TissueGraph — nodes (cells) with area/shape index, edges (junctions) with lengths/coordinates

Layer 3: Topology dynamics
  Input: TissueGraph time series from Layer 2
  Output: T1 events, neighbor exchange tracking, edge trajectories with sign convention
  → Requires Layer 2

Layer 4 (optional): Mechanics
  Input: Layer 2 graph (cell boundary geometry)
  Output: Inferred junction tensions, cell pressures, force-topology correlations
  → Uses ForSys (optional dep) for non-invasive force inference from geometry alone
  → Static mode: single-frame inference; Dynamic mode: uses vertex velocities across frames
```

### Directory Structure

```
CellFlow/
├── CellFlow/
│   ├── __init__.py
│   ├── napari.yaml                # napari manifest
│   ├── core/
│   │   ├── __init__.py
│   │   ├── graph.py               # High-level build API (single + batch)
│   │   ├── labels.py              # Segmentation labels → graph
│   │   ├── label_tracking.py      # IoU-based label tracking + contour extraction
│   │   ├── topology.py            # T1 detection, batch T1 detection
│   │   ├── tracks.py              # Cell track analysis (Layer 1: MSD, velocities)
│   │   ├── forsys_adapter.py       # TissueGraphFrame ↔ ForSys Frame conversion
│   │   ├── mechanics.py           # Force inference API (wraps ForSys solver)
│   │   └── io.py                  # Import/export
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── trajectories.py        # Edge trajectory construction with sign convention
│   │   ├── statistics.py          # Shape index (p0), T1 rates, distributions
│   │   └── events.py              # Event-triggered averaging around T1s
│   ├── napari/
│   │   ├── __init__.py
│   │   ├── widget.py              # Multi-tissue dock widget
│   │   └── visualization.py       # Napari layer rendering
│   └── structures.py              # All dataclasses and type definitions
├── tests/
│   ├── conftest.py                # Shared fixtures (synthetic data generators)
│   ├── test_graph.py
│   ├── test_labels.py
│   ├── test_label_tracking.py
│   ├── test_topology.py
│   └── test_dataset.py
├── pyproject.toml
└── TODO.md
```

---

## Data Structures (`structures.py`)

- `CellData` — Per-cell properties at a single timepoint (position, area, perimeter, shape_index, num_neighbors, track_id, vertices)
- `JunctionData` — Per-junction properties (cell_pair, length, coordinates, midpoint; optional tension/stress)
- `T1Event` — T1 transition (frame, losing_pair, gaining_pair, location, all 4 cells)
- `EdgeTrajectory` — Junction tracked through time with sign convention (positive before T1, negative after)
- `TissueGraphFrame` — Graph at one timepoint (networkx graph + cells dict + junctions dict)
- `TissueGraphTimeSeries` — One tissue over time (frames dict + trajectories + T1 events + metadata)
- `TissueGraphDataset` — **Primary output**: collection of TissueGraphTimeSeries (add/remove tissues, condition label, metadata)

---

## Implemented Modules

### `core/graph.py` — Build API

Single-tissue:
- `build_from_labels(label_stack, ..., min_iou) → TissueGraphTimeSeries` — assigns track IDs via IoU matching and extracts cell vertices
- `extract_graphs_from_labels(label_stack, ...) → TissueGraphTimeSeries` — Stage 1: graph extraction without tracking
- `assign_tracking_labels(series, label_stack, ...) → None` — Stage 2: IoU-based tracking, mutates in place
- `apply_track_map(series, track_map) → None` — apply pre-computed track map dict
- `has_tracking(series) → bool` — check if any cell has tracking

Multi-tissue (batch):
- `build_from_labels_4d(label_stacks, ...) → TissueGraphDataset` — accepts `Union[np.ndarray, List[np.ndarray]]` for variable-length movies

### `core/labels.py` — Graph from segmentation labels

- `labels_to_graph(label_frame, ...)` — regionprops + dilation/overlap adjacency detection + skeletonized boundaries
- `find_shared_boundary()`, `order_boundary_pixels()`, `calculate_edge_length()` — helpers

### `core/label_tracking.py` — Label-based cell tracking

- `match_labels(frame_t, frame_t1, min_iou)` — IoU-based frame-to-frame label matching
- `assign_track_ids(label_stack, min_iou)` — consistent track ID assignment across all frames
- `label_to_vertices(label_frame, cell_id)` — ordered boundary vertices via cv2 contour extraction

### `core/topology.py` — T1 detection

- `detect_t1_events(series)` — frame-to-frame graph comparison, validates 4-cell rosette pattern
- `detect_all_t1_events(dataset)` — batch: runs T1 detection + trajectory building on all tissues

### `analysis/trajectories.py` — Edge trajectories

- `build_edge_trajectories(series, t1_events)` — two-pass: link T1 edges, then fill junction data with sign convention
- `get_t1_trajectories(series)`, `get_stable_trajectories(series, min_frames)`

### `napari/visualization.py` — Layer rendering

- `build_all_junction_lines(series)` — pre-builds all junction lines with (frame, y, x) coordinates, colors by length
- `build_all_centroids(series)` — all centroids as Nx3 array
- `build_t1_markers(t1_events)` — T1 positions as Nx3 array

### `core/io.py` — Save/Load Dataset

- `save_dataset(dataset, path)` — saves to directory with `metadata.json` + one `tissue_NNN.npz` per tissue
- `load_dataset(path) → TissueGraphDataset` — reconstructs from NPZ files
- `load_multiple_datasets(paths) → Dict[str, TissueGraphDataset]` — loads multiple, keyed by condition
- NPZ format: edge lists (no pickle), ragged arrays via flat+offsets, NaN sentinels for None
- Private helpers: `_serialize_tissue`, `_deserialize_tissue`, `_serialize_ragged`, `_deserialize_ragged`

### `napari/widget.py` — Dock widget

- `CellFlowWidget`:
  - Segmentation labels input from napari Labels layer
  - Staged pipeline: Track → Extract → Analyze
  - Parameter inputs (pixel size, time interval, condition)
  - Preview layers, tagging UI, dataset management
  - **Dataset section**: summary, tissue spinner, Show/Remove tissue, Save/Load/New buttons

---

## Completed: Input Path Refactor (TODO.md)

- IoU-based label tracking for consistent track IDs
- Cell vertex extraction from label contours
- Build APIs for segmentation labels

*Note: Voronoi/TrackMate/nuclear tracks input paths were removed (2026-03-31). Only segmentation labels input remains.*

## Completed: Save/Load and Widget Redesign

- `core/io.py` — `save_dataset()`, `load_dataset()`, `load_multiple_datasets()` with NPZ + metadata.json format
- Widget redesigned with staged pipeline workflow
- Preview layers prefixed `[Preview]`, separate from dataset inspection layers
- Save/Load/New dataset buttons, scrollable layout

## Completed: Staged Pipeline with Visual QC

- Split monolithic `build_from_labels` into Stage 1 (graph extraction) and Stage 2 (tracking assignment)
  - `extract_graphs_from_labels()` — produces graphs with `track_id=None`
  - `assign_tracking_labels()` — mutates series in place
  - `apply_track_map()` — apply pre-computed track map dict to a series
  - `has_tracking()` — helper to check if any cell has tracking
  - Original `build_from_labels` wrapper preserved (calls stage-1 then stage-2)
- New QC visualization functions in `napari/visualization.py`:
  - `build_tracked_centroids()` — centroids colored by track_id (gray for untracked)
  - `build_tracked_labels()` — label array with track IDs as pixel values for napari Labels layer QC
  - `build_track_breaks()` — marks where tracks start/end mid-series (births/deaths)
  - `build_trajectory_lines()` — junction lines colored by trajectory_id

## Completed: UI/UX Redesign (TODO §10)

- **Napari-native label loading**: replaced file-picker with Labels layer dropdown; auto-syncs with `viewer.layers.events`; auto-selects active layer
- **Pipeline reorder for segmentation mode**: Track → Extract → Analyze (verify tracking before expensive graph extraction)
  - `CellTrackingWorker` — runs `assign_track_ids()` directly, returns track_map dict
  - Stage 1 shows tracked labels as napari Labels layer for QC
  - Stage 2 extracts graphs then applies `apply_track_map()`, keeps tracked labels visible underneath
- **Symmetric parameter layout**: each stage group contains its own parameters inline
  - Stage 1: tracking params (seg) or Voronoi/extraction params (non-seg)
  - Stage 2: extraction params (seg) or tracking params (non-seg)
  - Stage 3: all T1/trajectory params inline (no hidden Advanced panel)
- **Graph extraction params exposed**: dilation_radius, min_overlap_pixels, min_edge_length, filter_isolated — passed through to `GraphExtractWorker` and `extract_graphs_from_labels()`
- **Stage re-runnability**: Stage 1 stays enabled after completion for parameter tweaking
- **Batch mode hidden for segmentation** (single tissue from viewer)
- `PipelineStage` enum: IDLE → STAGE1_DONE → STAGE2_DONE → STAGE3_DONE
- 183 tests passing

---

## Remaining Work

### UI/UX improvements (TODO §11)

- Move Add/Discard controls below tagging group (tag before committing)
- Accept Image layers for tracking and graph building (auto-convert to labels)
- Include border edges in graph extraction, auto-tag with `"edge_border"`

### Phase 3: Cell-level analysis

- `core/tracks.py` — `compute_velocities()`, `compute_msd()`, `compute_velocity_correlation()`
- `analysis/statistics.py` — Shape index distributions, T1 rates (single-tissue + dataset-level), neighbor number distributions
- `analysis/events.py` — Event-triggered averaging of junction length around T1s, pooled across tissues

### Phase 4: Force inference via ForSys integration

Membrane tension and cell pressure inference using [ForSys](https://github.com/borgesaugusto/forsys) (Borges et al., iScience 2025). ForSys solves an inverse mechanical problem: given cell boundary geometry, it infers the forces (tensions on edges, pressures in cells) that produce the observed shapes. This replaces the earlier plan of requiring external TFM/MSM data for Layer 4.

**Why ForSys:** ForSys provides non-invasive force inference from segmentation alone — no traction force microscopy needed. It supports both static (single-frame) and dynamic (time-series with vertex velocities) inference. Its data model (Vertex, Edge, Cell, Frame) maps directly onto our TissueGraphFrame.

**Integration strategy:** Use ForSys as an **optional dependency** (`pip install CellFlow[forces]`). Only the solver is used — our graph extraction is more mature and already integrated with napari. A thin adapter converts between data structures.

#### ForSys technical reference (for implementation)

**Package:** `pip install forsys` (v1.1.5, BSD-3-Clause)

**Key dependency constraints:** `numpy < 2.0`, `scikit-image <= 0.21.0` — these are restrictive and may conflict. Plan to pin carefully or contribute upstream fixes.

**ForSys data model:**
- `forsys.vertex.Vertex(id, x, y)` — position in space
- `forsys.edge.Edge(id, v1, v2)` — connects two vertices, carries inferred tension
- `forsys.cell.Cell(id, vertices, edges)` — closed polygon, carries inferred pressure
- `forsys.frames.Frame(id, vertices_dict, edges_dict, cells_dict, time=t)` — one timepoint
- `forsys.ForSys(frames_dict)` — main solver wrapper

**Solver pipeline (per frame):**
```python
solver = forsys.ForSys(frames)
solver.build_force_matrix(when=0)
solver.solve_stress(when=0, allow_negatives=False)
solver.build_pressure_matrix(when=0)
solver.solve_pressure(when=0, method="lagrange_pressure")
```

**Dynamic mode:** With multiple frames and known time intervals, use `b_matrix="velocity"` in `solve_stress()` to incorporate vertex velocities for improved accuracy.

**Meshing:** ForSys subdivides edges into virtual segments (`forsys.virtual_edges.generate_mesh(vertices, edges, cells, ne=6)`) to capture curvature. This should be applied after converting our data.

**Input formats ForSys supports natively:** TIFF skeletons, CellPose `.npy` masks, Surface Evolver files. We bypass all of these — we feed pre-built graph data directly.

#### Implementation plan

- `core/forsys_adapter.py` — convert `TissueGraphFrame` → ForSys `Frame` objects
  - Map `CellData.vertices` → ForSys `Vertex` objects (deduplicate shared vertices between cells)
  - Map `JunctionData.coordinates` → ForSys `Edge` objects
  - Map cells → ForSys `Cell` objects with correct vertex/edge references
  - Handle border cells (ForSys may need special treatment for incomplete polygons)
- `core/mechanics.py` — high-level API
  - `infer_tensions(series, method="static") → TissueGraphTimeSeries` — runs ForSys on each frame, writes tensions back into `JunctionData.tension`
  - `infer_pressures(series) → TissueGraphTimeSeries` — writes pressures into `CellData` (may need new field)
  - Dynamic mode: pass vertex positions across frames for velocity-based inference
  - All ForSys imports guarded with try/except + clear error message
- Widget integration:
  - "Infer Forces" button in Stage 3 (after graph extraction, before/alongside T1 analysis)
  - Static vs Dynamic toggle
  - Visualization: junction lines colored by tension, cells colored by pressure
- `napari/visualization.py` updates:
  - `build_tension_colored_junctions(series)` — color edges by inferred tension
  - `build_pressure_colored_cells(series)` — color cell polygons by inferred pressure

### Testing with real data

- Test staged pipeline with segmentation: Track → inspect tracked labels → Extract → inspect junctions → Analyze → inspect T1s → Add
- Test widget with 2-3 real segmentation movies of different lengths
- Verify batch mode, Save/Load round-trip, tissue removal all work end-to-end

---

## Downstream Analysis (NOT part of the plugin)

Analysis and plotting happen in standalone scripts/notebooks:

```python
from CellFlow.core.io import load_dataset, load_multiple_datasets
from CellFlow.analysis.statistics import compute_t1_rates, pool_shape_index_distributions
from CellFlow.analysis.events import event_triggered_average_dataset

datasets = load_multiple_datasets(["data/wt/", "data/ko/", "data/mix/"])
for condition, ds in datasets.items():
    rates = compute_t1_rates(ds)
    print(f"{condition}: T1 rate = {rates.mean():.3f} ± {rates.std()/np.sqrt(len(rates)):.3f}")
```

---

## Style and Code Quality

- Type hints on all public functions
- Docstrings (NumPy style) on all public functions and classes
- No GUI imports in `core/` or `analysis/` modules
- `core/` and `analysis/` must be fully testable without napari running
- Use logging (not print) for debug output
- Keep functions short and focused — prefer composition over inheritance
