# CellFlow — Design Document

## 1. What CellFlow Is

CellFlow is a napari plugin for segmenting, tracking, and analyzing cell timelapse images.
It is specifically designed for tissue monolayers, where cells are dense, touching, and need to be tracked over time. Cells are imaged with a membrane marker, optionally combined with a nuclear channel.

The end product is a tracked label stack (one integer label per cell, consistent across time) that feeds into a graph/topology pipeline for extracting cell-level features: areas, junction lengths, neighbor relationships, T1 transitions, cell forces (via ForceSys), and custom analysis modules.

---

## 2. Feature Inventory

### Segmentation
- Cellpose-based segmentation (cpsam, nuclei, or custom model)
- Frame-by-frame or full-stack segmentation
- Optional dual-channel mode (membrane + nuclear channel fed to Cellpose as ch1/ch2)
- 3-D nuclear segmentation: Z-projection, stitch-z-slices, or full 3-D volumetric mode
- Export current frame to Cellpose GUI for manual correction; import result back
- Resegment specific frames without disturbing the rest of the stack

### Tracking
- LapTrack LAP tracking from label centroids
- Configurable max link / gap-closing distance and frame count
- Configurable cost metrics (euclidean, sqeuclidean, cityblock, cosine)
- IoU blending option for linking cost
- 3-D nuclear tracking (centroid in z/y/x space with z-scale correction)
- Retrack after corrections

### Manual Correction
- Per-frame editing of Labels layers in napari
- Keyboard-driven: delete, merge, split (watershed), swap, draw-path, line-split
- Undo (Ctrl-Z) with per-frame history
- Fix borders and clean stranded pixels utilities

### Graph / Topology
- Cell contact graph extraction per frame
- T1 transition detection
- Junction length distribution
- Edge trajectory tagging and analysis
- Cell distribution analysis
- Central junction identification
- Event-triggered averaging

### Export / Import
- Import/export label stacks (TIFF, npy, zarr)
- Export to Cellpose GUI via temp directory
- Dataset catalog (add/remove tissue series to a persistent HDF5 dataset)

### Dashboard
- Standalone Qt/web dashboard for visualizing analysis results outside napari

---

## 3. Workflows

### Workflow A — Standard 2D Segmentation and Tracking

**Data:** membrane (or cytoplasm) timelapse, optionally with a nuclear channel. Images are (T, H, W) or (T, Z, H, W); for the standard workflow Z-projection is done before segmentation if needed.

**Steps:**
1. Load image(s) into napari.
2. Open CellFlow. Set membrane channel (and optionally nuclear channel) in Project Panel.
3. **Segment** — run Cellpose on each frame, producing a (T, H, W) Labels layer.
   - Optionally: export a frame to Cellpose GUI, correct manually, import back.
   - Optionally: resegment individual frames with different parameters.
4. **Correct** — activate the Correction tab to fix merge/split/swap errors.
5. **Track** — run LapTrack on the corrected labels, relabelling so each cell keeps its ID across frames.
6. **Correct again** — fix tracking errors (wrong links manifest as ID jumps).
7. **Extract topology** — run graph extraction; inspect junctions and cells.
8. **Analyze** — run built-in or custom analysis modules, tag junctions/trajectories.
9. **Save** — add the tissue series to the dataset catalog.

### Workflow B — Nuclear-Guided Segmentation (primary design target)

**Data:** nuclear channel (z-stack timelapse, shape (T, Z, H, W)) plus membrane channel ((T, H, W) or (T, Z, H, W)).

**Motivation:** when cells are very dense or crawling, Cellpose on the membrane channel alone often merges cells. The nuclei provide unambiguous seeds — one nucleus per cell — that can be used to drive watershed segmentation of the membrane signal.

**Steps:**
1. Load nuclear channel and membrane channel into napari.
2. Open CellFlow.
3. **Nuclear Segmentation** — run Cellpose (nuclei model) on the nuclear z-stacks.
   - Mode options: Z-projection (recommended, gives 2D labels per frame), stitch z-slices, or full 3-D volumetric.
   - Output: Nuclear Labels layer ((T, H, W) for projection, (T, Z, H, W) for 3D modes).
4. **Flatten and post-process** — if 3-D nuclear labels were produced, flatten them to a 2D (T, H, W) layer and clean up (fill holes, remove small fragments, separate touching nuclei). Z-projection output skips this step.
5. **Correct nuclear labels** — use the standard Correction widget to fix any merge/split/erase errors in the nuclear layer.
6. **Track nuclear labels** — run the standard LapTrack tracking widget on the 2D nuclear labels. This assigns consistent nuclear track IDs across frames.
7. **Guided Segmentation** — run Cellpose on the membrane channel to generate flow images (probability maps), then watershed from nuclear seeds.
   - Each tracked nucleus label becomes one watershed seed.
   - The watershed boundary is the Cellpose probability map, so it follows membrane signal.
   - Output: Cell Labels layer with cell IDs equal to the nuclear track IDs — consistent across frames by construction.
8. **Correct cell segmentation** — use the Correction widget on the Cell Labels layer.
9. Optionally **retrack** if cell IDs need further cleanup.
10. Continue to topology extraction and analysis.

---

## 4. Current UI Structure and Its Problems

### Current structure: two separate plugin entries

```
napari Plugins menu
├── CellFlow           → CellFlowWidget (analysis_widget.py)
│     tabs: Edge Analysis | Segmentation | Tracking | Correction | Cell Bodies | Forces | Project
└── CellFlow Guided    → CellFlowGuidedWidget (guided_widget.py)
      tabs: 1. Nuclear Seg | 2. Nuclear Tracking | 3. Guided Seg
```

**Problems:**
- The two widgets are completely separate napari dock widgets; there is no shared state between them.
- Workflow B currently ends by producing a "Guided Segmentation" layer and then instructs the user to "Load this layer in CellFlow for topology and analysis" — a manual hand-off with no UI guidance.
- The Correction widget lives inside CellFlow but not CellFlow Guided, so correcting nuclear labels requires either switching plugins or using napari's built-in tools directly.
- Cellpose parameters (model, diameter, thresholds) are duplicated across NuclearSegTab, GuidedSegTab, and SegmentationTab with nearly identical UIs but no sharing.
- CellFlow Guided has a dedicated "Nuclear Tracking" tab with a 3D LapTrack implementation; this is unnecessary — tracking the flattened 2D nuclear labels with the existing standard tracking widget is equivalent and avoids duplication.
- It is not obvious to a new user that CellFlow Guided is a prerequisite for CellFlow, not an alternative.
- The "flatten/post-process" step that bridges 3D nuclear labels to 2D seeds for guided segmentation does not exist yet.
- There is no concept of two simultaneous segmentation stacks (nuclear labels + cell labels) in the state model.

---

## 5. Proposed Unified UI Design

### Design principle

There is one plugin entry: **CellFlow**. The widget is designed around Workflow B (nuclear-guided). Workflow A (standard) is a natural subset — simply skip the nuclear segmentation steps. Both workflows share the same state model, correction widget, tracking widget, topology, and analysis infrastructure.

### State model

The `CellFlowState` object (shared across all tabs via the registry) gains a second segmentation field:

```
CellFlowState
  .image_layer          → primary membrane/cytoplasm image
  .nuclear_image_layer  → nuclear channel image (optional)
  .cell_labels_layer    → cell segmentation (T, H, W) — was "seg_layer"
  .nuclear_labels_layer → nuclear segmentation (T, H, W) — new field
  .tissue               → TissueData / track metadata
```

Both labels layers are displayed simultaneously in napari. The correction and tracking widgets accept a **layer picker** so the user can explicitly select which labels layer to operate on at any given step.

### Top-level structure

```
CellFlow (single dock widget, tabbed)
│
├── [Project tab]
│     - Image channel pickers: Membrane channel, Nuclear channel (optional)
│     - Pixel size, time interval
│     - Dataset catalog
│
├── [Nuclear Seg tab]         ← new; consolidates NuclearSegTab + flatten
│     Section A — Cellpose Nuclear Segmentation
│       - Nuclear channel picker (auto-filled from Project)
│       - Cellpose parameters (collapsible): model=nuclei, z-mode selector
│       - Segment Frame / Segment Stack
│     Section B — Flatten & Post-process
│       - Source: Nuclear Labels layer picker
│       - Options: projection method (max/mean/sum), hole fill radius,
│                  min object size, split touching (watershed on distance map)
│       - "Flatten" button → writes to Nuclear Labels (T, H, W)
│     (hint: after this, correct the nuclear layer in the Correction tab,
│      then track in the Tracking tab, then run Guided Segmentation below)
│
├── [Cell Seg tab]            ← replaces current Segmentation tab
│     Section A — Standard Cellpose Segmentation
│       - Membrane channel picker
│       - Optional nuclear channel (two-channel mode)
│       - Cellpose parameters (collapsible)
│       - Segment Frame / Segment Stack
│       - Export to Cellpose GUI / Import from Cellpose GUI
│     Section B — Guided Segmentation  (enabled when nuclear labels exist)
│       - Membrane channel picker (auto-filled)
│       - Seed layer picker (defaults to Nuclear Labels)
│       - Cellpose parameters (collapsible)
│       - "Run Guided Segmentation" button
│
├── [Correction tab]          ← unchanged in function; gains layer picker
│     - Labels layer picker: [Cell Labels ▼] (can switch to Nuclear Labels)
│     - Activate / Deactivate
│     - Shortcut reference (collapsible)
│     - Fix borders / Clean stranded pixels
│
├── [Tracking tab]            ← unchanged in function; gains layer picker
│     - Labels layer picker: [Cell Labels ▼] (can switch to Nuclear Labels)
│     - LapTrack parameters (collapsible)
│     - Track / Retrack
│
└── [Analysis tab]            ← unchanged
      - Graph extraction, T1, edge analysis, tagging, analysis modules
```

### Why this is better

- One plugin, one mental model. Nuclear-guided segmentation is a set of additional steps, not a separate tool.
- Two explicit segmentation layers (nuclear + cell) are first-class citizens in the state model, so napari layers are never ambiguous.
- The correction and tracking widgets work on whichever layer the user picks — nuclear labels first, then cell labels, without switching plugins.
- Nuclear tracking uses the existing standard LapTrack tracking widget (on the flattened 2D nuclear labels); no separate 3D tracking implementation.
- The flatten/post-process step is now a first-class UI element.
- Shared Cellpose parameter forms can be a `CellposeParamsWidget` component (eliminates triple duplication).
- `CellFlowGuidedWidget` and its three sub-tabs (`NuclearSegTab`, `NuclearTrackingTab`, `GuidedSegTab`) are removed; logic is redistributed into the new Nuclear Seg tab and the existing Cell Seg tab.

---

## 6. Implementation Plan

**Phase 1 — State model**
- Add `nuclear_labels_layer` field to `CellFlowState` in `registry.py`.
- Update Project Panel to expose a nuclear channel picker and link it to the state.

**Phase 2 — Nuclear Seg tab (new file: `nuclear_seg_widget.py`)**
- Merge `NuclearSegTab` (segmentation logic) with a new flatten/post-process section.
- Section A: Cellpose nuclear segmentation → writes to `nuclear_labels_layer`.
- Section B: Flatten & post-process → reads from a 3D nuclear labels layer, writes 2D result back to `nuclear_labels_layer`.
- Backed by a new `flatten_nuclear_labels()` backend function in `segmentation.py`.

**Phase 3 — Cell Seg tab (update `segmentation_widget.py`)**
- Add a "Guided Segmentation" collapsible section below the standard Cellpose section.
- Guided section: membrane channel picker, seed layer picker (default: nuclear labels), Cellpose params.
- Backed by the existing `run_guided_segmentation()` function; output → `cell_labels_layer`.

**Phase 4 — Correction and Tracking tabs**
- Add an explicit layer picker dropdown to both tabs (defaulting to `cell_labels_layer`).
- Allows the user to switch to `nuclear_labels_layer` without re-opening any dialog.

**Phase 5 — Shared `CellposeParamsWidget`**
- Extract the Cellpose parameter form (model combo, custom path, auto-diameter, diameter, flow threshold, cellprob threshold, min size, GPU checkbox) into a reusable widget in a new `cellpose_params_widget.py`.
- Replace all three existing copies in `NuclearSegTab`, `GuidedSegTab`, `SegmentationTab`.

**Phase 6 — Remove CellFlow Guided**
- Delete `guided_widget.py`, `nuclear_seg_tab.py`, `nuclear_tracking_tab.py`, `guided_seg_tab.py`.
- Remove `CellFlowGuidedWidget` from `__init__.py` and `napari.yaml`.
- Remove `track_nuclei_3d_laptrack()` from `segmentation.py` (replaced by standard tracking).
- Remove `GuidedState` dataclass.

**Backend additions needed (`segmentation.py`)**
- `flatten_nuclear_labels(labels_4d, method, hole_fill_radius, min_size, split_touching)` → `(T, H, W)` label array.
