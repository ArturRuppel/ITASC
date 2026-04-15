# CellFlow Monorepo Refactor Plan

## Vision

CellFlow and ultrack_wrapper merge into a single monorepo. Each pipeline stage
becomes a separate, independently installable Python package. Stages are programmatically
independent (no imports between stage packages), which directly reflects their ability to
run asynchronously. Users install only the stages they need.

A shared `core` package defines the common interfaces, schema, and runner. A single
napari plugin auto-discovers installed stages at runtime via Python entry points. A
pipeline schema file (`pipeline_schema.json`) defines the I/O contracts at every stage
boundary, enabling users to enter the pipeline at any intermediate step — and serving
as the machine-readable "data formats" supplement for publication.

---

## Design Decisions

| # | Decision |
|---|---|
| R1 | One monorepo under the CellFlow name; `ultrack_wrapper` archived after migration |
| R2 | Python namespace packages: `cellflow.core`, `cellflow.cellpose`, `cellflow.ultrack`, `cellflow.analysis`, `cellflow.napari` |
| R3 | Stages registered and discovered at runtime via Python entry points (`cellflow.stages` group) |
| R4 | `uv workspaces` for monorepo management |
| R5 | `pipeline_schema.json` is the central integration artifact — written once by a "New Project" wizard |
| R6 | Pipeline is position-level; project is experiment-level |
| R7 | All widget tabs kept visible; pipeline schema drives per-tab status badges (✓/⚠/✗) |
| R8 | Validation mandatory before each stage runs: file existence + TIFF header shape/dtype check |
| R9 | Async = crash/resume via `pipeline_manifest.json` per position; auto-chain downstream steps is secondary |
| R10 | ultrack_wrapper git history preserved in monorepo via `git subtree add` |

---

## Target Directory Structure

```
CellFlow/                                ← git repo (stays here)
  pyproject.toml                         ← uv workspace root (no code, just workspace def)
  packages/
    core/                                ← cellflow-core
      pyproject.toml
      src/cellflow/core/
        __init__.py
        schema.py          # PipelineSchema, InterfaceSpec Pydantic models
        manifest.py        # PipelineManifest, StageRecord Pydantic models
        runner.py          # run_in_thread() wrapping napari thread_worker + manifest update
        logging.py         # StageLogger context manager
        paths.py           # unified path resolution (merges ultrack_wrapper/_paths.py)
        validation.py      # validate_inputs() — checks files + TIFF headers
        protocol.py        # StageProtocol typing.Protocol definition
    cellpose/                            ← cellflow-cellpose
      pyproject.toml
      src/cellflow/cellpose/
        __init__.py
        stages/
          raw_import.py    # migrated from ultrack_wrapper/stages/s00_raw.py
          nucleus_3d.py    # migrated from s01a_cellpose_nucleus.py
          cell_2d.py       # migrated from s01b_cellpose_cell.py
          flow_watershed.py# migrated from s02_flow_watershed.py
          contours.py      # migrated from s02c_cellpose_contours.py
        processing/
          flow_watershed.py           # migrated from ultrack_wrapper/processing/
          flow_watershed_postproc.py
        config.py          # CellposeConfig, FlowWatershedConfig (unified, Pydantic)
    ultrack/                             ← cellflow-ultrack
      pyproject.toml
      src/cellflow/ultrack/
        __init__.py
        stages/
          tracking.py      # migrated from s03_tracking.py (3-phase: seg/link/solve)
          project2d.py     # migrated from s04 stub (implement here)
          cell_labels.py   # migrated from s05 stub (implement here)
        config.py          # TrackingConfig Pydantic model
    analysis/                            ← cellflow-analysis (existing CellFlow backend)
      pyproject.toml
      src/cellflow/
        # NO __init__.py here — namespace package
        backend/           # existing code, all imports unchanged
          graph.py
          tracking.py
          segmentation.py
          topology.py
          trajectories.py
          tagging.py
          labels.py
          forsys.py
          analysis_modules.py
          api.py
          builtins/
        utils/
          structures.py
          io.py            # extended: condition_groups in DatasetCatalog, /provenance in HDF5
          mechanics.py
        dashboard/
    napari-plugin/                       ← cellflow-napari (existing CellFlow frontend)
      pyproject.toml
      src/cellflow/
        # NO __init__.py here — namespace package
        __init__.py        # napari entry point
        napari.yaml
        napari/
          __init__.py
          _plugin.py       # stage discovery via entry points, widget assembly
          widget.py        # top-level QTabWidget (thin shell)
          project_panel.py # extended: project_dir, Import from pipeline, status badges
          analysis_widget.py
          correction_widget.py
          tracking_widget.py
          segmentation_widget.py
          forces_widget.py
          edge_analysis_widget.py  # extracted from analysis_widget.py
          visualization.py
          workers.py       # migrated to generator pattern
          registry.py      # ViewerState: add project_dir, pipeline_schema fields
          ultrack_widgets/ # migrated from ultrack_wrapper/widgets/
            cellpose.py
            ultrack_widget.py
            flow_watershed.py
            data_prep.py
          viewers/
            loader.py      # implemented: load_stage() dispatch table
          runners/
            terminal.py
            script.py
  tests/
    core/
    cellpose/
    ultrack/
    analysis/
    integration/
  generate_sample_data.py
```

---

## Package Dependency Graph

```
cellflow-core          (no cellflow deps; only pydantic + tifffile)
      ↑
  ┌───┼────────────┐
  │   │            │
cellflow-  cellflow-  cellflow-
cellpose   ultrack    analysis
      \       |        /
       \      |       /
        cellflow-napari
              ↑
         (discovers all
          installed stages
          via entry points
          at runtime)
```

**Invariant:** no stage package (`cellpose`, `ultrack`, `analysis`) imports from another
stage package. `cellflow-napari` has no hard imports of stage packages.

---

## Stage Entry Points

Each stage package registers itself in `pyproject.toml`:

```toml
# packages/cellpose/pyproject.toml
[project.entry-points."cellflow.stages"]
raw_import       = "cellflow.cellpose.stages.raw_import:RawImportStage"
cellpose_nucleus = "cellflow.cellpose.stages.nucleus_3d:CellposeNucleusStage"
cellpose_cell    = "cellflow.cellpose.stages.cell_2d:CellposeCellStage"
flow_watershed   = "cellflow.cellpose.stages.flow_watershed:FlowWatershedStage"
contours         = "cellflow.cellpose.stages.contours:ContoursStage"

# packages/ultrack/pyproject.toml
[project.entry-points."cellflow.stages"]
tracking   = "cellflow.ultrack.stages.tracking:TrackingStage"
project2d  = "cellflow.ultrack.stages.project2d:Project2DStage"
cell_labels = "cellflow.ultrack.stages.cell_labels:CellLabelsStage"

# packages/analysis/pyproject.toml
[project.entry-points."cellflow.stages"]
segmentation      = "cellflow.backend.segmentation:SegmentationStage"
graph_extraction  = "cellflow.backend.graph:GraphExtractionStage"
topology_analysis = "cellflow.backend.topology:TopologyStage"
```

The napari plugin at startup:

```python
from importlib.metadata import entry_points
STAGES = {
    ep.name: ep.load()
    for ep in entry_points(group="cellflow.stages")
}
```

Only the stages whose packages are installed appear in the UI.

---

## The Stage Protocol (`cellflow.core.protocol`)

```python
from typing import Generator, NamedTuple, Protocol
from pydantic import BaseModel

class StageProgress(NamedTuple):
    done: int
    total: int
    message: str

class StageProtocol(Protocol):
    name: str          # must match entry-point key
    display_name: str  # shown in UI tab header
    config: BaseModel  # Pydantic model for this stage's parameters

    def run(self, **kwargs) -> Generator[StageProgress, None, None]: ...
    def validate_inputs(self, schema, root_dir, pos) -> "ValidationResult": ...
    def is_complete(self, root_dir, pos) -> bool: ...
```

Every `run()` generator:
- Yields `StageProgress(done, total, message)` — universal progress contract
- Accepts `overwrite: bool` keyword argument
- Is wrapped by `cellflow.core.runner.run_in_thread()` in the widget layer (never called directly by widgets)

---

## The Pipeline Schema File

`<experiment_root>/pipeline_schema.json` — written once by the "New Project" dialog.

```json
{
  "schema_version": "1.0",
  "created": "2026-04-14",
  "stages": [
    "raw_import", "cellpose_nucleus", "cellpose_cell",
    "flow_watershed", "tracking", "graph_extraction", "topology_analysis"
  ],
  "interfaces": {
    "cellpose_nucleus.output.dp": {
      "path_template": "pos{pos:02d}/1a_cellpose_nucleus/{stem}_dp.tif",
      "shape": "CZYX",
      "dtype": "float32",
      "entry_note": "One file per timepoint; C=3 flow axes"
    },
    "tracking.output.tracked_labels": {
      "path_template": "pos{pos:02d}/3_tracking/tracked_labels.tif",
      "shape": "THW",
      "dtype": "uint32",
      "entry_note": "Integer label stack; cell IDs must be consistent across T"
    }
  },
  "metadata": {
    "pixel_size_um": null,
    "time_interval_s": null,
    "conditions": {}
  }
}
```

When a stage is **disabled**, its interface entries become documented **external entry
points** — a user bringing data from another tool can read this file to know exactly
what format and path is expected. For publication, this file is the machine-readable
"data formats" section of the methods.

---

## The Pipeline Manifest (per position, dynamic)

`<experiment_root>/pos00/pipeline_manifest.json` — updated after each stage run.

```json
{
  "stages": {
    "cellpose_nucleus": {
      "status": "complete",
      "config_hash": "abc123",
      "finished_at": "2026-04-14T10:30:00"
    },
    "tracking": { "status": "pending" },
    "graph_extraction": { "status": "pending" }
  }
}
```

Status values: `pending | running | complete | stale | failed`

A stage goes `stale` when its upstream config hash changes. "Run All" skips `complete`
non-stale stages unless `overwrite=True`. This is the mechanism for crash/resume — the
manifest is the single source of truth for what has been done.

---

## Unified Directory Layout (both plugins' outputs)

```
<experiment_root>/
  pipeline_schema.json
  project.json                   ← full ProjectConfig (all stage configs, Pydantic)
  pos00/
    pipeline_manifest.json
    pipeline.log                 ← JSON-lines log, all stages
    0_raw/
    1a_cellpose_nucleus/
    1b_cellpose_cell/
    2_flow_watershed/
    2b_contours/
    3_tracking/                  ← ultrack data.db + tracked_labels.tif
    4_analysis/                  ← CellFlow HDF5 files
      tracked_labels.tif
      <condition>.h5
  pos01/
    ...
```

`cellflow.core.paths` provides all path-resolution functions. The `2_ultrack`
hardcoded path in the current ultrack_wrapper is replaced by `3_tracking` here.

---

## Implementation Phases

### Phase 0 — Monorepo scaffold *(prerequisite for everything)*
1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Create workspace `pyproject.toml` at CellFlow root
3. Create `packages/` with one `pyproject.toml` per sub-package
4. Set up namespace packages (no `__init__.py` at `src/cellflow/` level in `analysis/` and `napari-plugin/`)
5. Move existing CellFlow code with `git mv` (preserves history):
   - `cellflow/backend/` → `packages/analysis/src/cellflow/backend/`
   - `cellflow/utils/` → `packages/analysis/src/cellflow/utils/`
   - `cellflow/dashboard/` → `packages/analysis/src/cellflow/dashboard/`
   - `cellflow/frontend/` → `packages/napari-plugin/src/cellflow/napari/`
   - `cellflow/__init__.py` + `cellflow/napari.yaml` → `packages/napari-plugin/src/cellflow/`
6. Update all relative imports in frontend (`..backend.*` → `cellflow.backend.*`)
7. `pip install -e packages/core -e packages/analysis -e packages/napari-plugin`
8. Verify all existing CellFlow tests pass before proceeding

### Phase 1 — Core package
Define the contracts everything else will implement.
- `cellflow.core.protocol`: `StageProtocol`, `StageProgress`
- `cellflow.core.schema`: `PipelineSchema`, `InterfaceSpec`; `load()` / `save()`
- `cellflow.core.manifest`: `PipelineManifest`, `StageRecord`; atomic JSON write
- `cellflow.core.paths`: unified path resolution; fixes `3_tracking` naming
- `cellflow.core.validation`: `validate_inputs()` using `tifffile` header-only reads
- `cellflow.core.runner`: `run_in_thread()` wrapping `napari.qt.threading.thread_worker` + manifest update
- `cellflow.core.logging`: `StageLogger` context manager writing `run.log` + `pipeline.log`

### Phase 2 — Migrate ultrack_wrapper
- `git subtree add --prefix=_ultrack_import <ultrack_wrapper_path> HEAD` to import history
- Migrate code to `packages/cellpose/` and `packages/ultrack/` at new paths
- Unify duplicate `FlowWatershedConfig`: merge extra fields into Pydantic version
- Fix s00 progress yield arity: 4-tuple → 3-tuple
- Remove hardcoded `2_ultrack` path; use `cellflow.core.paths`
- Register all stages as entry points in their `pyproject.toml`
- Each stage `run()` wraps in `StageLogger`; each stage implements `validate_inputs()`

### Phase 3 — Napari plugin: stage discovery + status UI
- `_plugin.py`: discover stages via entry points at import time
- "New Project" dialog: lists discovered stages with checkboxes → writes `pipeline_schema.json` + directory skeleton
- Per-tab status badge: ✓/⚠/✗ shown in tab header based on `validate_inputs()` result
- "Pipeline Status" strip in Project Panel: one status row per stage
- Stage completion auto-fills downstream input paths
- Migrate ultrack_wrapper widget UIs into `cellflow.napari.ultrack_widgets/`

### Phase 4 — CellFlow frontend cleanup
- Migrate Edge Analysis from raw `QThread+QObject` → `thread_worker` generator
- Migrate `ForceInferenceWorker` similarly
- Extract `EdgeAnalysisWidget` from the root `CellFlowWidget`
- `ViewerState`: add `project_dir`, `pipeline_schema` fields; retire `_dataset`/`preview_series` legacy shims
- "Import from pipeline" button: reads schema-resolved path, validates, loads, auto-fills metadata

### Phase 5 — Checkpointing + logging
- After segmentation/tracking: auto-save checkpoint TIFFs to `4_analysis/`
- On startup with `project_dir`: detect checkpoints, offer "Resume" if manifest is incomplete
- `StageLogger` writing JSON-lines to `pipeline.log`
- Collapsible log viewer in each stage widget (last N lines of `pipeline.log`)

### Phase 6 — DatasetCatalog + pooling groundwork
- `cellflow.utils.io`: extend `DatasetCatalog` with `condition_groups: Dict[str, List[str]]`
- Update `.cfproj` to v3.0 (backward-compatible reader)
- Add `/provenance` group to HDF5 (pipeline manifest + config hashes at save time)
- Optional: `build_summary_db(catalog)` → SQLite for fast cross-position queries
- Defer pooling UI to a future milestone

---

## Migration Map

### From ultrack_wrapper

| Source | Destination |
|---|---|
| `stages/s00_raw.py` | `packages/cellpose/.../stages/raw_import.py` |
| `stages/s01a_cellpose_nucleus.py` | `packages/cellpose/.../stages/nucleus_3d.py` |
| `stages/s01b_cellpose_cell.py` | `packages/cellpose/.../stages/cell_2d.py` |
| `stages/s02_flow_watershed.py` | `packages/cellpose/.../stages/flow_watershed.py` |
| `stages/s02c_cellpose_contours.py` | `packages/cellpose/.../stages/contours.py` |
| `stages/s03_tracking.py` | `packages/ultrack/.../stages/tracking.py` |
| `stages/s04_project2d.py` (stub) | `packages/ultrack/.../stages/project2d.py` (implement) |
| `stages/s05_cell_labels.py` (stub) | `packages/ultrack/.../stages/cell_labels.py` (implement) |
| `processing/flow_watershed*.py` | `packages/cellpose/.../processing/` |
| `_config.py` | split to `cellflow.core` (shared) + each stage package |
| `_paths.py` | `packages/core/.../paths.py` |
| `runners/local.py` (stub) | `packages/core/.../runner.py` (implemented) |
| `runners/terminal.py` | `packages/napari-plugin/.../runners/terminal.py` |
| `runners/script.py` (stub) | `packages/napari-plugin/.../runners/script.py` (implement) |
| `viewers/loader.py` (stub) | `packages/napari-plugin/.../viewers/loader.py` (implemented) |
| `widgets/` | `packages/napari-plugin/.../ultrack_widgets/` |
| `_widget_data_prep.py` | `packages/napari-plugin/.../ultrack_widgets/data_prep.py` |

### From CellFlow

| Source | Destination |
|---|---|
| `cellflow/backend/` | `packages/analysis/src/cellflow/backend/` (unchanged imports) |
| `cellflow/utils/` | `packages/analysis/src/cellflow/utils/` |
| `cellflow/dashboard/` | `packages/analysis/src/cellflow/dashboard/` |
| `cellflow/frontend/` | `packages/napari-plugin/src/cellflow/napari/` |
| `cellflow/__init__.py` | `packages/napari-plugin/src/cellflow/__init__.py` |
| `cellflow/napari.yaml` | `packages/napari-plugin/src/cellflow/napari.yaml` |

---

## Installation (post-refactor)

```bash
# Full install (paper users, everything)
pip install cellflow[all]

# Analysis only — no Cellpose, no Ultrack
pip install cellflow[analysis]

# Custom subset
pip install cellflow-core cellflow-ultrack cellflow-analysis cellflow-napari

# Development (all packages, editable)
uv sync --all-packages
```

---

## Verification Checklist

1. `python -c "from cellflow.core import PipelineSchema; print('core ok')"`
2. `python -c "from cellflow.backend.graph import build_from_labels; print('analysis ok')"`
3. `pytest tests/` — all existing CellFlow tests pass after Phase 0
4. `python -c "from importlib.metadata import entry_points; print([e.name for e in entry_points(group='cellflow.stages')])"`
5. Launch napari → CellFlow plugin loads → all expected tabs present
6. "New Project" dialog → `pipeline_schema.json` written to experiment root
7. Run one stage → `pipeline_manifest.json` updated → `pipeline.log` written
