# TODO — Input Path Refactor & Voronoi Enhancements

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

## Implementation order

1. **Structures** (1a-1d) — foundation, no breaking changes
2. **TrackMate parser** (2a-2d) — independent, testable
3. **Voronoi enhancements** (3a-3d) — independent, testable
4. **Label tracking** (4a-4d) — independent, testable
5. **Build API** (5a-5d) — integrates 2+3+4
6. **Widget** (6a-6e) — thin layer on top of 5
