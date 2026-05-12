# CellFlow — Project Synthesis

**CellFlow** is a hypothesis-driven cell segmentation and tracking pipeline for time-lapse fluorescence microscopy. It ingests multi-channel 3D+time (NDTiff) datasets, runs Cellpose for probability/flow estimation, generates multiple segmentation hypotheses via watershed and flow-following, and solves global ILP tracking via Ultrack — all from a napari GUI.

Version: 0.2.0 · Python ≥3.9 · GPL-3.0

---

## Source Tree

```
src/cellflow/
├── __init__.py                            # Root package (empty)
├── napari.yaml                            # napari manifest entry point
├── core/
│   ├── __init__.py
│   ├── data_prep.py                       # NDTiff → Z-averaged TIFF export
│   ├── paths.py                           # Pipeline directory layout
│   └── logging.py                         # JSON-lines stage logger
├── segmentation/
│   ├── __init__.py                        # Contour watershed, consensus boundary, foreground scoring
│   ├── _array_utils.py                    # DP stack normalisation helpers
│   ├── contour_filtering.py               # Median/Gaussian filters for contour maps
│   ├── flow_following.py                  # Euler-integrated flow + EDT-gravity cell segmentation
│   └── cell_label_icm.py                  # Geodesic-unary ICM solver for cell boundaries
├── tracking/
│   ├── __init__.py                        # Stub (legacy v1 engine removed)
│   └── retracker.py                       # Centroid LAP relabeling after manual correction
├── database/
│   ├── __init__.py                        # Re-exports: tracked + validation I/O
│   ├── tracked.py                         # Tracked label TIFF read/write
│   ├── validation.py                      # Frame-level & track-level validation metadata
│   └── hypotheses.py                      # HDF5 hypothesis pool (T×P×Z×Y×X)
├── tracking_ultrack/
│   ├── __init__.py                        # Lazy-exports: TrackingConfig
│   ├── config.py                          # Pydantic config model for the Ultrack stage
│   ├── ingest.py                          # Hypothesis HDF5 → Ultrack NodeDB + OverlapDB
│   ├── db_build.py                        # Canonical DB construction (segment+inject+score+link)
│   ├── linking.py                         # Default or IoU-weighted frame-to-frame linking
│   ├── solve.py                           # ILP solver wrapper (Gurobi or CBC)
│   ├── export.py                          # Ultrack solution → tracked_labels.tif
│   ├── seed_prior.py                      # Node probability scoring + edge boosting
│   ├── validation_nodes.py                # Inject validated masks as REAL/FAKE nodes
│   ├── anchor.py                          # Anchor-frame constraints (REAL/FAKE annotations)
│   ├── extend.py                          # Per-track forward/backward extension from DB
│   ├── anchor_diagnostics.py              # Anchor quality reports
│   ├── cell_boundary_selection.py         # Boundary-aware cell selection heuristics
│   ├── reseed.py                          # Validate-and-resolve loop orchestration
│   └── metrics.py                         # Track summary & binary IoU metrics
├── analysis/
│   ├── __init__.py                        # Re-exports: build_position_analysis_artifact
│   ├── position_artifact.py               # Per-position HDF5 artifact builder (cells, edges, T1s)
│   ├── artifact_reader.py                 # HDF5 artifact reader for napari visualization
│   └── nls_classification.py              # NLS-mCherry track classification (high/low)
├── correction/
│   ├── __init__.py                        # Re-exports: label correction operations
│   └── labels.py                          # In-place label editing operations
├── meta/
│   ├── __init__.py
│   └── catalog.py                         # Study directory discovery & CSV catalog
└── napari/
    ├── __init__.py                        # napari compat patch + CellFlowWidget re-export
    ├── _napari_compat.py                  # napari layer delegate monkey-patch
    ├── main_widget.py                     # Top-level widget: project, config, 6 workflow sections
    ├── data_prep_widget.py                # Raw data import UI
    ├── data_panel_widget.py               # Project status / file-existence dashboard
    ├── cellpose_widget.py                 # Cellpose execution UI
    ├── hpc_cellpose_widget.py             # HPC-launched Cellpose UI
    ├── nucleus_workflow_widget.py         # Nucleus segmentation + tracking workflow UI
    ├── cell_workflow_widget.py            # Cell segmentation workflow UI
    ├── cell_boundary_workflow_widget.py   # Cell boundary / contour map workflow UI
    ├── correction_widget.py               # Label correction / painting UI
    ├── analysis_widget.py                 # Position analysis artifact generation UI
    ├── nls_classification_widget.py       # NLS classification UI
    ├── meta_widget.py                     # Meta-source browser UI
    ├── artifact_visualization.py          # Edge, T1, centroid rendering on napari layers
    ├── widgets.py                         # Shared UI widgets (CollapsibleSection, etc.)
    ├── utils.py                           # Shared napari utility functions
    └── ui_style.py                        # Styling helpers
```

---

## Module-by-Module Overview

### `cellflow.core` — Foundation

- **`paths.py`**: Defines the on-disk pipeline directory layout. Maps stage names (`raw_import`, `cellpose`, `nucleus`, `cell`, `analysis`) to numbered directories (`0_input` through `4_analysis`) under per-position `posNN/` folders. Provides `stage_dir(root, pos, stage)` and `log_path(root, pos)`.

- **`data_prep.py`**: Ingests raw NDTiff datasets. Reads three channels (nucleus marker CSU642, membrane marker CSU488, NLS-mCherry CSU561), estimates z-drift between frames by fitting double-sigmoid profiles to the membrane channel, applies z-shift correction, and exports Z-averaged 4D stacks and per-Z-slice 4D volumes as compressed TIFFs. Uses a `DatasetConfig` dataclass for parameters (path, positions, downsampling, frame range).

- **`logging.py`**: `StageLogger` — a context manager that appends JSON-lines entries (timestamp, stage name, level, message) to a per-position `pipeline.log`. Used by the napari workflow widgets for progress and error tracking.

### `cellflow.segmentation` — Hypothesis Generation & Cell Segmentation

The segmentation module generates candidate label maps (hypotheses) at multiple parameter settings and provides cell segmentation algorithms. The `__init__.py` serves as a consolidated hub for contour watershed segmentation, consensus boundary construction, foreground scoring, and Cellpose mask generation.

- **`__init__.py`**: Re-exports from `flow_following`, `contour_filtering`, and `cell_label_icm`, and provides the following inline functions and classes:
  - `apply_gamma` — gamma-correct Cellpose logits via sigmoid → power → logit.
  - `foreground_score_stack` / `foreground_mask_stack` — convert probability or flow-DP data into foreground score images or binary masks.
  - `ContourWatershedParams` — parameter dataclass for contour-map watershed hypothesis generation (seed distance, foreground/ridge thresholds, noise perturbation).
  - `build_consensus_boundary` — reduce mask boundaries and occupancy over (threshold × z-slice) combinations, returning (boundary, foreground) images.
  - `build_consensus_boundary_2d` — 2D variant operating on Z-averaged probability and filtered flow.
  - `compute_cellpose_foreground_masks` — generate binary cell foreground masks using Cellpose dynamics across a time stack.
  - `compute_contour_watershed` — seeded watershed on a consensus boundary image using EDT-maxima seeds carved around strong contour ridges.

- **`_array_utils.py`**: DP stack normalisation helpers. `normalize_seeded_watershed_dp_stack` accepts common Cellpose flow layouts (4D or 5D, channels-first or channels-last) and returns a canonical `(T, Z, C, Y, X)` float32 array.

- **`flow_following.py`**: Cell segmentation by Euler-integrating the Cellpose flow field blended with EDT-direction gravity toward tracked nuclei. Two strategies:
  - *Legacy* (`capture_radius > 0`): Pixels are captured when they enter a fixed radius around a nucleus.
  - *Two-phase* (`capture_radius == 0`): Phase 1 integrates every foreground pixel along flow+gravity; Phase 2 grows labels outward in progressive shells through displaced-position topology.
  
  Also provides `compute_flow_following_movie` for per-frame processing and `build_consensus_boundary_flow_following` for consensus contour maps across probability thresholds.

- **`contour_filtering.py`**: Spatial and temporal median/Gaussian filtering for nucleus contour-map stacks. Handles 2D, 3D (TYX), and 4D (TCYX) layouts.

- **`cell_label_icm.py`**: Geodesic-unary Iterated Conditional Modes (ICM) solver for cell boundary optimization. Decomposed into three stages:
  1. `initialize_icm` — computes geodesic distance costs from each nucleus to every foreground pixel, builds spatial (8-connected) and temporal (face-diagonal) Potts pairwise weights.
  2. `refine_icm` — runs Numba-JIT Gauss-Seidel ICM sweeps with anchor constraints (nucleus pixels are frozen).
  3. `commit_labels` — writes the result to TIFF.
  
  Supports parallel geodesic computation via fork-based multiprocessing and HDF5 caching of unary costs.

### `cellflow.database` — Hypothesis & Validation Storage

- **`hypotheses.py`**: HDF5 hypothesis pool with schema `hypotheses/t{t:03d}/p{p:03d}/labels` (shape Z×Y×X, uint32). Each `p` group stores parameter attributes. Provides `read_hypothesis_labels` and `list_hypotheses`.

- **`tracked.py`**: Multipage TIFF storage for tracked nucleus label volumes. Schema: (T, Y, X) uint32. Provides `write_tracked_frame`, `read_tracked_frame`, `tracked_frame_exists`, `tracked_n_frames`.

- **`validation.py`**: Persistent validation metadata. Two stores:
  - `validated_frames.json` — set of frame indices where every cell ID has been individually validated (used by the cell workflow).
  - `validated_cells.json` — per-cell-ID map of validated frame sets (used by the nucleus workflow). Supports `validate_track`, `invalidate_track`, `remap_validated_tracks`.

### `cellflow.tracking` — Legacy Stub

The v1 greedy tracking engine (`propagator.py`, `frame_selector.py`, `consensus_movie.py`) has been removed. Ultrack (`tracking_ultrack/`) is now the sole tracking engine.

- **`retracker.py`**: After manual correction, remaps target frame cell IDs to match a reference frame via centroid-distance Hungarian algorithm. `retrack_frame_constrained` supports locked IDs that must be preserved.

### `cellflow.tracking_ultrack` — Ultrack ILP Tracker

The primary tracking engine, using Ultrack's global ILP solver.

- **`config.py`**: Pydantic `TrackingConfig` model covering all Ultrack parameters: area filters, linking (max distance, neighbors, IoU mode), ILP solver (appear/disappear/division weights, link function, solution gap, time limit), segmentation (min area, foreground threshold, watershed hierarchy), and seed-prior scoring (quality, circularity, seed affinity weights).

- **`ingest.py`**: Converts hypothesis HDF5 into Ultrack's NodeDB + OverlapDB. Each (t, p, label_id) becomes one NodeDB row; cross-partition mask overlaps at the same t become OverlapDB pairs. Deduplicates identical partition structures via canonical BLAKE2b hashing.

- **`db_build.py`**: Orchestrates the canonical Ultrack DB construction pipeline: (1) run `ultrack.segment()` on contour maps + foreground masks to populate NodeDB via hierarchy watershed, (2) optionally inject validated nodes as REAL/FAKE annotations, (3) score node probabilities from nucleus intensity image + seed affinity, (4) run linking, (5) optionally boost edges incident to validated nodes. Returns a `UltrackDatabaseBuildReport`.

- **`linking.py`**: Two linking modes:
  - *default*: Ultrack's built-in linker.
  - *iou*: Custom IoU-weighted linker that aligns node masks at their centroids and blends IoU with distance into a single weight, using KDTree candidate search.

- **`solve.py`**: Thin wrapper around `ultrack.core.solve.processing.solve`. Runs the ILP solver with optional annotation-aware mode.

- **`export.py`**: Exports the solver's selected nodes as `tracked_labels.tif`. Tries three strategies: `tracks_to_zarr`, `to_labels`, and CTC export fallback. Optionally preserves validated cell IDs by pasting them back into the export.

- **`seed_prior.py`**: Computes node probabilities for the ILP solver. Three components: (1) *drop fraction* — fraction of ring pixels around the mask that are dimmer than the mask median (quality signal), (2) *mask circularity*, (3) *seed affinity* — Gaussian-decaying similarity to validated (REAL) nodes in space, time, and area. Also provides `boost_validated_edges` which increases link weights incident to validated nodes.

- **`validation_nodes.py`**: Injects validated tracked label masks as Ultrack NodeDB rows. Matched candidates are updated in-place and marked REAL; conflicting candidates are marked FAKE. Unmatched validated masks get new reserved REAL nodes. Cross-node overlaps are added to OverlapDB.

- **`anchor.py`**: Pins the solver's output at a specific frame to match an anchor labelmap. Matched nodes become REAL; all other nodes at that frame become FAKE. `suppress_anchor_adjacent_fragments` marks obvious fragment alternatives in neighboring frames as FAKE.

- **`extend.py`**: Per-track forward/backward extension using candidates from the Ultrack DB. Supports greedy overwrite planning that resolves label conflicts by finding disjoint assignments for all affected cells.

- **`reseed.py`**: Validate-and-resolve loop orchestration. `resolve_with_canonical_segment` re-ingests hypotheses, injects validated nodes, scores, links, solves, exports, and merges validated IDs back — using canonical `ultrack.segment` instead of hypotheses.h5.

- **`metrics.py`**: `tracked_label_summary` computes track count, average length, and per-track frame-presence lengths. `binary_labelmap_iou` computes foreground IoU between two binarized labelmaps.

### `cellflow.analysis` — Downstream Artifacts

- **`position_artifact.py`**: Builds a per-position HDF5 analysis artifact. Extracts cell columns (frame, cell_id, area, centroid, perimeter, bbox), edge records (cell-cell contacts and border edges with ordered coordinates), and T1 transitions (cell neighbor-swap events). Uses `assign_persistent_edge_ids` to maintain edge identity across frames through T1 events. The artifact schema: `cells/table`, `edges/table`, `edges/coordinates`, `t1_events/table`, plus provenance metadata.

- **`artifact_reader.py`**: Reads the position analysis HDF5 artifact into a `PositionArtifactData` dataclass. Provides helper methods: `edge_lines()` returns per-edge coordinate arrays, `centroid_points()` returns per-cell centroid positions.

- **`nls_classification.py`**: Classifies tracks as NLS-high or NLS-low based on mCherry intensity. Measures per-track median NLS intensity, splits tracks into two clusters via a two-Gaussian EM fit, and patches the position analysis H5 with classification columns (`class_label`, `nls_status`, `nls_track_median_intensity`). Exposed as a CLI entry point: `cellflow-classify-nls`.

### `cellflow.correction` — Interactive Label Editing

- **`labels.py`**: In-place label correction operations on a single 2D segmentation frame. Operations:
  - `erase_cell` — remove a cell.
  - `merge_cells` — merge two touching cells (binary closing + relabel).
  - `split_across` — watershed-split a cell using two click seeds.
  - `split_draw` — split a cell along a manually drawn line.
  - `draw_cell_path` — draw a closed polygon to create or extend a cell.
  - `swap_labels` — swap two cell IDs.
  - `relabel_cell` — assign a new ID to a cell.
  - `expand_label_to_foreground` — grow a label into connected foreground.
  - `fill_label_holes` — fill enclosed background gaps.
  - `fix_label_semiholes` — fill narrow border-connected background gaps.
  - `clean_stranded_pixels` — remove disconnected same-label fragments.
  - `best_overlapping_label` — find the target-frame label with most overlap.

### `cellflow.meta` — Study-Level Discovery

- **`catalog.py`**: Scans a root directory for `condition/experiment/position` trees. Returns sorted records with file-existence checks for the analysis artifact, nucleus tracked labels, and cell tracked labels. Supports CSV catalog I/O (`load_meta_catalog`, `save_meta_catalog`), H5 file discovery, and catalog record merging.

### `cellflow.napari` — GUI Layer

The napari plugin provides a 6-stage collapsible workflow UI:

1. **Data Preparation** — `data_prep_widget.py`: Configure and run `cellflow.core.data_prep.run()`.
2. **Cellpose** — `cellpose_widget.py` + `hpc_cellpose_widget.py`: Run Cellpose locally or launch on HPC.
3. **Nucleus Segmentation & Tracking** — `nucleus_workflow_widget.py`: Hypothesis generation, Ultrack ILP tracking, correction, and validation.
4. **Cell Segmentation** — `cell_workflow_widget.py` + `cell_boundary_workflow_widget.py`: Cell-boundary hypothesis sweeps, consensus movie building, flow-following, ICM refinement, and correction.
5. **Analysis** — `analysis_widget.py` + `nls_classification_widget.py`: Build position analysis artifacts and classify NLS tracks.
6. **Meta Analyzer** — `meta_widget.py`: Browse study directories via the catalog.

Shared components: `main_widget.py` (project config + section container), `data_panel_widget.py` (file-existence dashboard), `correction_widget.py` (label painting/editing), `artifact_visualization.py` (napari layer rendering of edges, T1 events, centroids), `widgets.py` (CollapsibleSection, etc.), `ui_style.py` (styling), `utils.py` (shared helpers).

---

## Pipeline Data Flow

```
NDTiff raw data
  │
  ▼  (core/data_prep.py)
0_input/  ── nucleus_zavg.tif, cell_zavg.tif, NLS_zavg.tif, z_shift.csv
  │
  ▼  (Cellpose, external)
1_cellpose/ ── probability logits + flow fields (Z×Y×X or T×Z×Y×X)
  │
  ├─────────────────────────────────────────────────────┐
  ▼  (segmentation watershed)                            ▼  (segmentation flow_following)
2_nucleus/ hypotheses.h5                                3_cell/ hypotheses.h5
  │  (T × P × Z × Y × X)                                 │  (cell-boundary hypotheses)
  │                                                      │
  └────────────────────────┬─────────────────────────────┘
                           ▼  (tracking_ultrack)
                      data.db (Ultrack NodeDB + LinkDB + solver)
                           │
                           ▼  (tracking_ultrack/export.py)
                     3_cell/ tracked_labels.tif (ILP solution)
                           │
                           ▼  (analysis/position_artifact.py)
                     4_analysis/ position_analysis.h5
                           │  cells/table, edges/table, t1_events/table
                           ▼  (analysis/nls_classification.py)
                           cells/table/class_label (high/low)
```

---

## Key Design Decisions

1. **Hypothesis-driven architecture.** Multiple segmentation parameter sets are generated and stored in HDF5, then the Ultrack ILP solver selects the best combination per cell per frame. This decouples segmentation quality from tracking consistency.

2. **Single tracking engine.** The legacy greedy propagator (`tracking/propagator.py`) and hypothesis-selection pipeline (`frame_selector.py`, `consensus_movie.py`) have been removed. Ultrack's global ILP solver (`tracking_ultrack/`) is the sole tracking engine, providing globally optimal solutions across the full time series.

3. **Validate-and-resolve loop.** Users can manually validate individual cell tracks in the napari viewer. The validated masks are injected as REAL/FAKE annotations into Ultrack's NodeDB, and the ILP is re-solved with these fixed constraints — without losing the validated work.

4. **Anchor-frame gating.** A full annotated frame can be used as a hard constraint: matched nodes become REAL, all others at that frame become FAKE, and adjacent-frame fragment suppression removes obvious alternatives.

5. **Identity invariant.** The pipeline maintains cell_id == nucleus_id identity. Both tracked label stacks share the same ID space, and the analysis artifact builder validates this at construction time.

6. **Parallelism at every level.** Hypothesis generation (multiprocessing pool), geodesic unary computation (fork-based), and DB ingest all use configurable worker counts.

7. **Position-centric directory layout.** All stages for a single microscope position live under `posNN/`, making it easy to archive, share, or reprocess individual positions independently.
