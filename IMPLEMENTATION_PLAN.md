# napariTissueGraph — Implementation Plan

## Project Overview

**napariTissueGraph** is a napari plugin for quantifying tissue topology, rearrangement dynamics, and junction mechanics from microscopy data. It replaces and extends napariCellFlow.

The tool takes cell positions over time (from nuclear tracking or segmentation labels) and builds a dynamic graph representation of the tissue. It tracks junction lengths, detects T1 transitions (cell intercalation events), computes cell-level statistics, and optionally maps traction force data onto junction edges.

### Key Design Principles

1. **Layered architecture**: Each layer builds on the previous. Users get analysis depth proportional to their input data.
2. **External tools for segmentation and tracking**: This tool does NOT do segmentation or tracking. It accepts their outputs.
3. **TFM/MSM is optional**: Force mapping enriches the analysis but is never required.
4. **Two input paths, one internal representation**: Segmentation labels and nuclear tracks both produce the same `TissueGraph` object.
5. **Separation of core logic and GUI**: All analysis code must work without napari. The napari widget is a thin visualization/interaction layer on top.
6. **Multi-tissue first**: The primary output is `TissueGraphDataset` (multiple tissues from one condition). Single-tissue is just a dataset with one entry.

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
│   │   ├── graph.py               # High-level build API (single + batch)
│   │   ├── voronoi.py             # Nuclear positions → Voronoi tessellation → graph
│   │   ├── labels.py              # Segmentation labels → graph
│   │   ├── topology.py            # T1 detection, batch T1 detection
│   │   ├── tracks.py              # Cell track analysis (Layer 1: MSD, velocities)
│   │   ├── mechanics.py           # Optional TFM/MSM force mapping (Layer 4)
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
│   ├── test_voronoi.py
│   ├── test_labels.py
│   ├── test_topology.py
│   └── test_dataset.py
├── pyproject.toml
└── TODO.md
```

---

## Data Structures (`structures.py`)

- `CellData` — Per-cell properties at a single timepoint (position, area, perimeter, shape_index, num_neighbors)
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
- `build_from_labels(label_stack, ...) → TissueGraphTimeSeries`
- `build_from_tracks(positions, ...) → TissueGraphTimeSeries`

Multi-tissue (batch):
- `build_from_labels_4d(label_stacks, ...) → TissueGraphDataset` — accepts `Union[np.ndarray, List[np.ndarray]]` for variable-length movies
- `build_from_tracks_4d(positions, ...) → TissueGraphDataset` — positions as Nx4 array (tissue_id, frame, y, x)

### `core/voronoi.py` — Voronoi from positions

- `compute_voronoi(positions, image_shape)` — scipy Voronoi with mirror-point boundary handling
- `voronoi_to_graph(vor, positions, n_real, image_shape)` — extract cells, junctions, networkx graph

### `core/labels.py` — Graph from segmentation labels

- `labels_to_graph(label_frame, ...)` — regionprops + dilation/overlap adjacency detection + skeletonized boundaries
- `find_shared_boundary()`, `order_boundary_pixels()`, `calculate_edge_length()` — helpers

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

### `napari/widget.py` — Multi-tissue dock widget

- `DatasetBuildWorker` — QThread worker producing `TissueGraphDataset`
- `TissueGraphWidget`:
  - Input type selector (Segmentation Labels / Nuclear Tracks)
  - Multi-file loading via "Load Labels..." button (variable-length .tif files)
  - Layer selector for tracks mode
  - Parameter inputs (pixel size, time interval, condition)
  - Build Graph button with progress bar
  - Tissue inspection: spinner to select tissue, per-tissue layer display, "Remove Tissue" for QC
  - Dataset summary display

---

## Remaining Work

### Phase 3: Cell-level analysis

- `core/tracks.py` — `compute_velocities()`, `compute_msd()`, `compute_velocity_correlation()`
- `analysis/statistics.py` — Shape index distributions, T1 rates (single-tissue + dataset-level), neighbor number distributions
- `analysis/events.py` — Event-triggered averaging of junction length around T1s, pooled across tissues

### Phase 5: Force mapping and I/O

- `core/mechanics.py` — `map_traction_forces()`, `map_msm_stress()` (optional Layer 4)
- `core/io.py`:
  - `save_dataset(dataset, path)` — directory of NPZ files + metadata.json
  - `load_dataset(path) → TissueGraphDataset`
  - `load_multiple_datasets(paths) → Dict[str, TissueGraphDataset]` for cross-condition analysis

### Testing with real data

- Test widget with 2-3 real segmentation movies of different lengths
- Verify frame slider, T1 detection, tissue removal all work end-to-end

---

## Downstream Analysis (NOT part of the plugin)

Analysis and plotting happen in standalone scripts/notebooks:

```python
from napariTissueGraph.core.io import load_dataset, load_multiple_datasets
from napariTissueGraph.analysis.statistics import compute_t1_rates, pool_shape_index_distributions
from napariTissueGraph.analysis.events import event_triggered_average_dataset

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
