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
- [ ] Replace file-picker label loading with a layer dropdown that lists napari Labels layers
- [ ] User selects an existing Labels layer (or loads one via napari's built-in file open)
- [ ] Remove batch loading of labels — unnecessary complexity

### 10b. Reorder pipeline: track first, then extract
- [ ] Stage 1: Cell tracking — run tracking on the label stack, show tracked segmentation in the viewer (color-coded by track ID)
- [ ] Stage 2: Graph extraction — extract cell graph from the tracked labels, with user-facing parameters (dilation radius, min overlap pixels, min edge length, filter isolated)
- [ ] Stage 3: T1 + edge trajectory analysis (unchanged)

### 10c. Consolidate tracking parameters
- [ ] Move `max_area_change` into the cell tracking panel alongside `min_iou` (not a separate panel)
- [ ] Keep tracking parameters grouped logically: IoU threshold, max area change ratio

### 10d. Graph extraction parameters
- [ ] Expose in widget: dilation radius, min overlap pixels, min edge length, filter isolated toggle
- [ ] Collapsible or inline in the graph extraction stage

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

### Next
9. **UI/UX redesign** (10a-10d) — napari-native label loading, reordered pipeline, parameter consolidation
10. **Cell-level analysis** (9a-9c) — new analysis modules
