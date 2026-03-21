# TODO — napariTissueGraph

## Overview

Three input modes, two computation problems, one shared output structure.

| | **Segmentation** | **Nuclear Tracks** | **Both** |
|---|---|---|---|
| Shapes from | Label boundaries | Voronoi tessellation | Label boundaries |
| Tracking from | Label matching (IoU) | TrackMate (given) | TrackMate (given) |
| Positions | Label centroids | Nuclear positions | Nuclear positions |

---

## 1. Data structure updates (`structures.py`)

### 1a. Add `track_id` to `CellData`
- [x] Add `track_id: Optional[int] = None` field
- [x] For segmentation path: assigned by label matching across frames
- [x] For tracks path: comes directly from TrackMate track ID

### 1b. Add `vertices` to `CellData`
- [x] Add `vertices: Optional[np.ndarray] = None` — ordered polygon boundary (Nx2 array)
- [x] Segmentation path: extract from label contour
- [x] Voronoi path: Voronoi region vertices (clipped to image bounds)
- [x] Needed for uniform downstream analysis (area, shape index from actual polygon)

### 1c. Expand `InputType` enum
- [x] Add `SEGMENTATION_WITH_TRACKS = "segmentation_with_tracks"`

### 1d. Add Voronoi method enum
- [x] `VoronoiMethod` enum: `STANDARD`, `LLOYD` (centroidal Voronoi)

---

## 2. TrackMate XML parser (`core/trackmate.py`)

### 2a. Parse spots
- [x] Extract per-spot: `ID`, `POSITION_X`, `POSITION_Y`, `FRAME`
- [x] Build dict: `frame -> list of (spot_id, y, x)`
- [x] Extract spatial units and calibration from `<Model spatialunits=...>`

### 2b. Parse tracks
- [x] Parse `<AllTracks>` section: track ID, edges (source_id -> target_id)
- [x] Build mapping: `spot_id -> track_id`
- [x] Handle splits (TrackMate allows splitting tracks) — assign child spots to parent track or create new track ID

### 2c. Parse metadata
- [x] Image dimensions from `<ImageData>` section (for Voronoi bounding)
- [x] Detector/tracker settings (informational, store in dataset metadata)
- [x] Time interval (`dt`) and pixel size (`dx`, `dy`) from geometry section

### 2d. Output format
- [x] Return a dataclass or dict with: positions per frame, track assignments, image shape, calibration
- [x] Write tests with the sample XML file

---

## 3. Voronoi enhancements (`core/voronoi.py`)

### 3a. Store cell vertices in `voronoi_to_graph`
- [x] Extract ordered polygon vertices per cell from `vor.regions`
- [x] Clip to image bounds
- [x] Store in `CellData.vertices`

### 3b. Lloyd's algorithm (centroidal Voronoi tessellation)
- [x] Implement `lloyd_relaxation(positions, image_shape, n_iterations, tol)`
- [x] Each iteration: compute Voronoi -> move seeds to polygon centroids -> repeat
- [x] Stop when max displacement < `tol` or `n_iterations` reached
- [x] Return final positions + final Voronoi
- [x] After convergence: `position == shape_centroid` (the nice property)

### 3c. Parameterize `compute_voronoi`
- [x] Add `method: VoronoiMethod = VoronoiMethod.STANDARD` parameter
- [x] Add `lloyd_iterations: int = 10` and `lloyd_tol: float = 0.1` parameters
- [x] Route to standard or Lloyd's path accordingly

### 3d. Tests
- [x] Test Lloyd's converges (centroid displacement decreases)
- [x] Test Lloyd's output matches standard Voronoi when `n_iterations=0`
- [x] Test vertex extraction produces valid closed polygons

---

## 4. Segmentation label tracking (`core/label_tracking.py`)

### 4a. IoU-based frame-to-frame matching
- [x] `match_labels(frame_t, frame_t1, min_iou)` — match labels across two frames by intersection-over-union
- [x] Return mapping: `label_t -> label_t1` (or None if no match)
- [x] Handle appearing/disappearing cells (division, death, entering/leaving FOV)

### 4b. Build track IDs from label matching
- [x] `assign_track_ids(label_stack, min_iou)` — run matching across all frames
- [x] Assign consistent `track_id` to cells tracked across frames
- [x] New track ID for cells that appear without a match

### 4c. Extract label contours as vertices
- [x] `label_to_vertices(label_frame, cell_id)` — ordered boundary points
- [x] Store in `CellData.vertices` during `labels_to_graph`

### 4d. Tests
- [x] Test matching with known overlapping labels
- [x] Test track ID assignment across 3+ frames with a cell disappearing
- [x] Test contour extraction produces valid closed polygons

---

## 5. Build API updates (`core/graph.py`)

### 5a. Update `build_from_tracks`
- [x] Accept TrackMate parsed data (not just raw positions)
- [x] Pass `track_id` into `CellData`
- [x] Pass Voronoi method parameters through
- [x] Store cell vertices from Voronoi

### 5b. Update `build_from_labels`
- [x] Run label tracking to assign `track_id`
- [x] Extract and store cell vertices
- [x] Accept tracking parameters (min_iou, etc.)

### 5c. Add `build_from_both`
- [x] Accept label stack + TrackMate data
- [x] Shapes from labels, track IDs from TrackMate
- [x] Match TrackMate spots to label centroids (nearest neighbor within threshold)
- [x] Requires spatial alignment between tracking and segmentation data

### 5d. Update `build_from_labels_4d` and `build_from_tracks_4d`
- [x] Pass through new parameters
- [x] Add `build_from_both_4d` for combined mode

---

## 6. Widget updates (`napari/widget.py`)

### 6a. TrackMate XML file picker
- [x] Add "Load TrackMate XML..." button (visible in Nuclear Tracks and Both modes)
- [x] Parse XML on load, show summary (N spots, N tracks, N frames, image dims)
- [x] Store parsed data for build step

### 6b. Voronoi parameter panel
- [x] Voronoi method dropdown: Standard / Lloyd's relaxation
- [x] Lloyd's iterations spinbox (default 10)
- [x] Only visible in Nuclear Tracks mode

### 6c. Segmentation tracking parameter panel
- [x] Min IoU threshold (default 0.3)
- [x] Only visible in Segmentation mode

### 6d. "Both" input mode
- [x] Add third option to input type combo
- [x] Show both file pickers (labels + TrackMate XML)
- [x] Matching threshold parameter for spot-to-label assignment

### 6e. Update build pipeline
- [x] Route to correct `build_from_*` based on input mode
- [x] Pass Voronoi/tracking parameters to workers

---

---

## 7. Analysis parameter tunability

Most analysis steps currently have no user-facing parameters. The algorithms are robust and
shouldn't need heavy configuration, but if a QC step fails the user needs some knobs to turn
rather than just discarding the tissue. Keep the core algorithms untouched — expose thresholds
and filtering criteria only.

### 7a. T1 detection parameters
- [x] Minimum junction length threshold — junctions shorter than this are treated as "collapsed" (pre-filter before topology comparison). Default: 0 (current behavior)
- [x] Spatial proximity constraint — max distance between lost/gained edge midpoints to pair them as a T1. Currently all lost/gained pairs are checked. Default: unlimited (current behavior)
- [x] Pass these through `detect_t1_events()` and `detect_all_t1_events()` without restructuring the core algorithm

### 7b. Edge trajectory filtering parameters
- [x] `min_frames` is already exposed in `get_stable_trajectories()` — surface this in the widget
- [x] Min trajectory completeness — fraction of total frames that a trajectory must span to be included in analysis (e.g., 0.5 = must exist for at least half the movie)
- [x] Max gap tolerance — allow trajectories with brief disappearances (edge not detected for N frames) to still be considered continuous

### 7c. Label tracking parameters
- [x] `min_iou` is already exposed — make sure it's adjustable in the Analyze stage (not just at extraction time)
- [x] Max area change ratio — reject matches where cell area changes by more than this factor frame-to-frame (catches segmentation errors)

### 7d. Widget: analysis parameter panel
- [x] Collapsible "Advanced" section in the Analyze stage of the pipeline
- [x] Sensible defaults so users never have to touch it unless QC fails
- [x] Tooltips explaining what each parameter does and when to adjust it

---

## 8. Edge & junction tagging

Users need to name/tag specific junctions so they can find them again in the dataset and
filter downstream analysis to tagged subsets. Primary use case: tagging the "central junction"
in a T1 rosette so later analysis (MSD, event-triggered averaging) can focus on it.

### 8a. Data model changes (`structures.py`)
- [x] Add `tags: Set[str]` field to `EdgeTrajectory` (default: empty set)
- [x] Add `tags: Set[str]` field to `JunctionData` (default: empty set)
- [x] Add `name: Optional[str]` field to `EdgeTrajectory` (user-assigned label, e.g., "central_junction_1")
- [x] Tags propagate through save/load (update `core/io.py`)

### 8b. Programmatic tagging API (`analysis/tagging.py`)
- [x] `tag_trajectory(series, trajectory_id, tag)` / `untag_trajectory()`
- [x] `tag_junction(frame, cell_pair, tag)` — tags a junction in a specific frame
- [x] `get_trajectories_by_tag(series, tag)` — filter trajectories by tag
- [x] `get_junctions_by_tag(frame, tag)` — filter junctions by tag
- [x] Bulk tagging: `tag_trajectories_near(series, location, radius, tag)` — tag all trajectories whose midpoint falls within a radius of a point
- [x] `get_trajectory_by_name(series, name)` — find by user-assigned name
- [x] `name_trajectory(series, trajectory_id, name)` — set/clear name
- [x] `get_all_tags(series)` — collect all unique tags
- [x] `clear_tag(series, tag)` — remove a tag from everything

### 8c. Napari interactive tagging
- [x] Junction Shapes layer carries `features` DataFrame (trajectory_id, cell_pair, tags, name)
- [x] Selection workflow: user switches Shapes layer to select mode (press 4) → clicks junction lines to select → enters tag name → clicks "Tag Selected" → tags written back to EdgeTrajectory/JunctionData
- [x] "Color by tags" toggle — switches between trajectory-colored and tag-colored junctions
- [x] "Show only tagged" toggle — filters the Shapes layer to only tagged junctions
- [x] Tag list widget showing all tags in the current tissue, with counts + "Clear Selected Tag"
- [x] Tagging works both during pipeline (after Stage 3) and when inspecting dataset tissues

### 8d. Persistence
- [x] Tags stored in the NPZ+JSON save format (extend io.py serialization)
- [x] Tags survive the full round-trip: tag in napari → save → load → tags visible again

---

## 9. Phase 3: Cell-level analysis

### 9a. Velocities & MSD (`analysis/cell_dynamics.py`)
- [ ] Cell velocity from tracked centroid displacement (with optional temporal smoothing window)
- [ ] Mean squared displacement per cell and ensemble-averaged MSD
- [ ] Diffusion coefficient extraction (linear fit to MSD)

### 9b. Statistics (`analysis/statistics.py`)
- [ ] Distributions: edge lengths, cell areas, coordination number, shape index
- [ ] Per-frame summary statistics (mean, std, median)
- [ ] Time-averaged statistics across frames

### 9c. Event-triggered averaging (`analysis/events.py`)
- [ ] Average junction length trajectory aligned to T1 events
- [ ] Average cell area/shape index around T1 events
- [ ] Configurable time window around events
- [ ] Support filtering to tagged junctions only (integrate with tagging system, §8)

---

## 10. UI/UX redesign

Rethink the widget workflow to feel native to napari and simplify the pipeline.

### 10a. Load labels from napari viewer
- [x] Replace file-picker label loading with a layer dropdown that lists napari Labels layers
- [x] User selects an existing Labels layer (or loads one via napari's built-in file open)
- [x] Remove batch loading of labels — unnecessary complexity
- [x] Auto-sync dropdowns with viewer layer events (inserted/removed/changed)
- [x] Auto-select active layer

### 10b. Reorder pipeline: track first, then extract
- [x] Stage 1: Cell tracking — run tracking on the label stack, show tracked segmentation in the viewer (color-coded by track ID)
- [x] Stage 2: Graph extraction — extract cell graph from the tracked labels, with user-facing parameters (dilation radius, min overlap pixels, min edge length, filter isolated)
- [x] Stage 3: T1 + edge trajectory analysis (unchanged)
- [x] Non-segmentation modes keep Extract→Track→Analyze order

### 10c. Consolidate tracking parameters
- [x] Move `max_area_change` into the cell tracking panel alongside `min_iou` (not a separate panel)
- [x] Keep tracking parameters grouped logically: IoU threshold, max area change ratio
- [x] Each stage has its own parameter area — symmetric layout across all 3 stages

### 10d. Graph extraction parameters
- [x] Expose in widget: dilation radius, min overlap pixels, min edge length, filter isolated toggle
- [x] Inline in the graph extraction stage (Stage 2 for segmentation, Stage 1 for Both mode)

### 10e. Stage re-runnability
- [x] Stage 1 button stays enabled after completion — user can tweak params and re-run

---

## 11. Next UI/UX improvements

### 11a. Move "Add to Dataset" below tagging
- [x] Move the Add/Discard controls to after the tagging group, so users tag junctions before committing to the dataset

### 11b. Accept Image layers for tracking and graph building
- [x] Tracking (CellTrackingWorker) and graph extraction (GraphExtractWorker) should also accept napari Image layers
- [x] Auto-convert Image layer data to integer labels (e.g. via unique-value casting or thresholding)
- [x] Graph builder should accept the active layer (Labels or Image) directly

### 11c. Include border edges and auto-tag them
- [x] Graph extraction should include edges at the image border (currently filtered by `filter_isolated`)
- [x] Border edges should be automatically tagged with `"edge_border"` so they can be excluded from downstream analysis
- [x] This preserves the full graph while letting users filter border artifacts
- ~~**Known bug:** Border edge detection (`find_border_boundary`) is unreliable~~ — Fixed: segments are now split by contiguity, `min_border_edge_length` filters small holes, only cell-vs-background boundaries get `edge_border` tag.

---

## 12. Tagging UI/UX improvements

The current tagging workflow is functional but clunky. Improve discoverability and
direct manipulation of tags in the viewer.

### 12a. Display tag labels in the viewer
- [x] Show tag names as text annotations next to tagged edges (napari Points layer with text)
- [x] Toggle to show/hide tag labels in the viewer ("Show tag labels" checkbox)
- [x] Tags should be readable at typical zoom levels without overlapping

### 12b. Interactive tag selection and deletion from the viewer
- [x] Clicking a tag label in the viewer should highlight/select the corresponding edge
- [x] Delete key (or a "Remove Tag" button) should remove the tag from the selected edge — both from the viewer display and the internal data (JunctionData.tags / EdgeTrajectory.tags)
- [x] Deleting a tag should immediately update the tag list counts

---

## 13. Force inference via ForSys

Non-invasive inference of membrane tensions and cell pressures from segmentation geometry,
using [ForSys](https://github.com/borgesaugusto/forsys) (Borges et al., iScience 2025) as
an optional dependency. ForSys solves an inverse problem: given cell boundary shapes, it
computes the mechanical forces (edge tensions, cell pressures) that produce those shapes.
Supports static (single frame) and dynamic (multi-frame with vertex velocities) inference.

**Key constraint:** ForSys pins `numpy < 2.0` and `scikit-image <= 0.21.0`. Install as
optional extra: `pip install napariTissueGraph[forces]`.

### 13a. Data model updates (`structures.py`)
- [ ] Add `pressure: Optional[float] = None` to `CellData`
- [ ] `JunctionData.tension` and `JunctionData.normal_stress` already exist — verify they're included in io.py serialization

### 14b. ForSys adapter (`core/forsys_adapter.py`)
- [ ] `tissue_frame_to_forsys(frame: TissueGraphFrame) → forsys.frames.Frame`
  - Deduplicate shared vertices across cells (CellData.vertices share boundary points)
  - Build ForSys Vertex objects from unique vertex positions
  - Build ForSys Edge objects from JunctionData.coordinates
  - Build ForSys Cell objects with correct vertex/edge winding order
  - Handle border cells (incomplete polygons at image boundary)
- [ ] `forsys_results_to_tissue(forsys_frame, tissue_frame)` — write tensions/pressures back
  - Map ForSys edge tensions → JunctionData.tension (match by vertex positions)
  - Map ForSys cell pressures → CellData.pressure
- [ ] Apply ForSys meshing (`virtual_edges.generate_mesh(ne=6)`) for curved boundaries
- [ ] Guard all ForSys imports with try/except, raise clear ImportError

### 14c. Mechanics API (`core/mechanics.py`)
- [ ] `infer_tensions(series, method="static") → TissueGraphTimeSeries`
  - Static: run ForSys solver independently per frame
  - Dynamic: pass vertex positions across consecutive frames for velocity-based inference (`b_matrix="velocity"`)
  - Write results into JunctionData.tension for each frame
- [ ] `infer_pressures(series) → TissueGraphTimeSeries`
  - Run pressure solver (Lagrange multiplier method) per frame
  - Write results into CellData.pressure
- [ ] `infer_forces(series, method="static")` — convenience wrapper: tensions + pressures in one call

### 14d. Visualization (`napari/visualization.py`)
- [ ] `build_tension_colored_junctions(series)` — junction lines colored by inferred tension (continuous colormap)
- [ ] `build_pressure_colored_cells(series)` — cell polygon fills colored by inferred pressure

### 13e. Widget integration
- [ ] "Infer Forces" button (enabled after Stage 2 graph extraction)
- [ ] Static / Dynamic toggle
- [ ] Results shown as overlay layers (tension-colored edges, pressure-colored cells)
- [ ] Graceful handling if forsys not installed (button disabled with tooltip)

### 13f. Tests
- [ ] Test adapter round-trip: TissueGraphFrame → ForSys Frame → solve → write back
- [ ] Test with synthetic regular hexagonal lattice (known analytical tensions)
- [ ] Test dynamic mode with 2-frame series
- [ ] Test graceful failure when forsys not installed

---

## 14. Analysis dashboard

Napari is the right tool for spatial visualization and interactive annotation, but not for
exploratory data analysis. Users need a separate dashboard to query, filter, plot, and compute
statistics over tissue graph datasets. The dashboard should be modular so that analysis
"modules" can be developed independently, shared, and installed by the community.

### 14a. Data API (`core/api.py`)
- [ ] Clean Python API for querying the tissue graph: cells, edges, junctions, trajectories
- [ ] Filtering by tags, timepoint ranges, topological properties (e.g. neighbor count, coordination number)
- [ ] Temporal queries: T1 events, trajectory histories, time-since-last-event
- [ ] Returns pandas DataFrames / standard Python objects for easy downstream use

### 14b. Analysis module interface
- [x] Standard base class / protocol that all analysis modules implement
- [x] Each module declares its parameters (auto-generates UI widgets in the dashboard)
- [x] Each module has a `compute()` method (takes data API + parameters, returns results) and a `visualize()` method (returns figures/tables)
- [x] Modules are discoverable via entry points or a plugin folder — users can pip-install, download, or write their own

### 14c. Dashboard application
- [x] Built with Dash + Plotly — serves as a web app and works inside Jupyter
- [x] Dataset loader: open saved tissue graph datasets
- [x] Module browser: lists installed analysis modules, user selects one to run
- [x] Parameter panel: auto-generated from the module's declared parameters
- [x] Results area: displays interactive Plotly plots, sortable/filterable tables, and summary statistics
- [x] "Open Dashboard" button in napari widget launches dashboard with current dataset
- [x] Theme switcher with 4 themes (Midnight, Ocean, Slate, Light) — instant CSS variable swap
- [ ] Fix theme styling: Dash DataTable and component internals (filter inputs, pagination, tooltips) don't fully pick up CSS custom properties — may need per-render inline styles or a full page reload on theme change
- [ ] Tissue map visualization — interactive Plotly figure showing cell polygons and junction lines, colored by metric
- [ ] "Open in napari" button to launch spatial visualization of the current selection

### 14d. Built-in analysis modules
- [x] Junction length distribution — histogram of junction lengths, filterable by tag and neighbor count
- [x] T1 transition rate — transition rate as a function of time since last transition
- [x] Cell area / shape index distributions — per-frame and time-averaged
- [ ] MSD and diffusion — mean squared displacement per cell, ensemble average, diffusion coefficient
- [x] Event-triggered averaging — average junction length / cell area aligned to T1 events

---

## 15. Separate Voronoi/tracks workflow from main widget

The main TissueGraphWidget should always work with segmentation labels. The Voronoi
tessellation path (creating labels from nuclear tracks) is a preprocessing step that
belongs in its own widget or tool.

### 15a. Refactor main widget architecture
- [x] Extract workers into `napari/workers.py`
- [x] Remove input type mode switching; main widget is segmentation-only
- [x] Use active layer instead of layer dropdown
- [x] Create per-viewer registry (`napari/registry.py`) for shared state between widgets

### 15b. Create Nuclear Tracks widget
- [x] Add `voronoi_to_labels()` in `core/voronoi.py` (rasterize Voronoi → Labels layer)
- [x] Create Nuclear Tracks widget (`napari/tracks_widget.py`) — TrackMate XML loading, Voronoi → Labels layer, track ID assignment
- [x] Register both widgets in `napari.yaml`
- [x] Clean up unused core functions and `InputType` enum values

---

## Implementation order (completed + remaining)

### Done
1. **Structures** (1a-1d)
2. **TrackMate parser** (2a-2d)
3. **Voronoi enhancements** (3a-3d)
4. **Label tracking** (4a-4d)
5. **Build API** (5a-5d)
6. **Widget** (6a-6e)
7. **Edge & junction tagging** (8a-8d)
8. **Analysis parameters** (7a-7d)
9. **UI/UX redesign** (10a-10e)
10. **UI/UX improvements** (11a-11c)
11. **Separate Voronoi/tracks into own widget** (15)
12. **Tagging UI/UX** (12a-12b)
13. **Data API** (14a)
14. **Analysis module interface** (14b)
15. **Dashboard application** (14c) — Dash + Plotly, theme system, napari launch button
16. **Built-in analysis modules** (14d) — junction lengths, T1 rate, cell distributions, event-triggered averaging

### Next
17. **Dashboard polish** — fix theme styling for Dash internals, tissue map visualization
18. **MSD and diffusion** (9a, 14d) — cell dynamics analysis module
19. **Cell-level analysis** (9a-9c) — velocities, statistics, event-triggered cell metrics
20. **Force inference** (13a-13f) — ForSys integration for tension/pressure inference
