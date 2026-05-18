# Cellpose Widget — Local Per-Channel Runner

**Date:** 2026-05-18
**Scope:** `src/cellflow/napari/cellpose_widget.py` (new),
`src/cellflow/segmentation/cellpose_runner.py` (new),
`src/cellflow/napari/main_widget.py` (replace `_CellposePanel`),
`pyproject.toml` (add `cellpose>=4.0`), and accompanying tests.

## Problem

The Cellpose section in `main_widget.py` (`_CellposePanel`, lines 36–99) is a
placeholder. It only documents that "Cellpose runs externally on the cluster"
and exposes a `PipelineFilesWidget` plus the `CellposeZavgVizWidget`. There is
no in-app way to actually run Cellpose; users have to use a separate HPC
launcher (`HpcCellposeWidget`, available from the personal-tools entry point)
that depends on Pasteur-specific paths and SLURM scaffolding and is therefore
not transferable.

The other workflow stages (nucleus, cell) have already been redesigned to use
the per-stage row pattern (⚙ params toggle, optional ▷ preview, ▶ run/cancel,
collapsed inline parameters, shared status + progress). The Cellpose section
should join that family with a real local runner — one row per channel.

## Goals

- Replace `_CellposePanel` with a real, local Cellpose runner widget.
- One row per channel: **Nucleus Cellpose** and **Cell Cellpose** — each with
  its own ⚙ / ▷ / ▶ controls and its own params.
- Preview button runs Cellpose on the current frame (and current z-slice for
  2D cases), producing in-memory napari layers; no file writes.
- Run button processes all frames and writes the canonical outputs under
  `1_cellpose/`.
- Reuse the well-tested processing logic from
  `/home/aruppel/Projects/HPC/cellpose_full/cellpose_full.py` by vendoring it
  into `src/cellflow/segmentation/cellpose_runner.py`.
- Keep the existing `CellposeZavgVizWidget` embedded at the bottom unchanged.
- Use the `cpsam` (Cellpose-SAM v4) model, loaded lazily once per session.

## Non-goals

- No change to or removal of `HpcCellposeWidget`.
- No custom-model loader, model picker, or tile/batch/normalization controls.
- No multi-GPU or explicit device selection — auto-detect only.
- No frame-range UI; full-stack runs only (preview covers per-frame inspection).
- No new CLI; the runner module is import-only.
- No change to `CellposeZavgVizWidget` internals or to downstream pipeline
  files / formats.

## Final Layout

```text
Pipeline Files                                  [unchanged external header]
──────────────────────────────────────────────
Nucleus Cellpose                     ⚙   ▷   ▶
  ┊ ☐ 3D mode
  ┊ Anisotropy   [ 1.50 ]
  ┊ Diameter     [ 25   ]
  ┊ Min size     [ 0    ]
  ┊ Gamma        [ 1.00 ]
Cell Cellpose                        ⚙   ▷   ▶
  ┊ Diameter     [ 0    ]
  ┊ Min size     [ 0    ]
  ┊ Gamma        [ 1.00 ]
[██████░░░░] Nucleus: frame 3/120...           [shared progress + status]
──────────────────────────────────────────────
Z-avg Visualization (existing CellposeZavgVizWidget, unchanged)
```

Per-row buttons use the same `_tool_btn` helpers as the cell workflow. While
one row is running, the other row's ⚙ ▷ ▶ are disabled (mirrors the
`_set_running_stage` pattern in `CellWorkflowWidget`).

## Architecture

### New module: `src/cellflow/segmentation/cellpose_runner.py`

Pure-Python, no Qt. Vendored and adapted from
`/home/aruppel/Projects/HPC/cellpose_full/cellpose_full.py`. Public surface:

```python
@dataclass(frozen=True)
class NucleusParams:
    do_3d: bool
    anisotropy: float
    diameter: float  # 0 means "let cpsam decide" → None passed to model
    min_size: int
    gamma: float

@dataclass(frozen=True)
class CellParams:
    diameter: float
    min_size: int
    gamma: float

def get_model() -> CellposeModel:
    """Lazy-load the `cpsam` model once per process; cached at module level."""

def is_model_loaded() -> bool:
    """For the widget to decide whether to show 'Loading model...' status."""

def run_nucleus_stack(
    stack: np.ndarray,             # (T, Z, Y, X)
    params: NucleusParams,
    *,
    progress_cb=None,              # (done_t, total_t, message)
    cancel_cb=None,                # returns True between frames to abort
) -> tuple[np.ndarray, np.ndarray]:  # (prob_3dt, dp_3dt)

def run_cell_stack(
    stack: np.ndarray,             # (T, Z, Y, X), always 2D slice-by-slice
    params: CellParams,
    *,
    progress_cb=None,
    cancel_cb=None,
) -> tuple[np.ndarray, np.ndarray]:

def run_nucleus_frame(
    frame: np.ndarray,             # (Z, Y, X)
    z: int | None,                 # None when do_3d=True
    params: NucleusParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Preview helper: single frame. If z is None, runs full 3D on (Z,Y,X);
    otherwise runs 2D on frame[z]."""

def run_cell_frame(
    frame: np.ndarray,             # (Z, Y, X)
    z: int,
    params: CellParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Preview helper: single 2D slice."""

def write_outputs(
    prob_3dt: np.ndarray,
    dp_3dt: np.ndarray,
    output_dir: Path,
    channel: Literal["nucleus", "cell"],
) -> None:
    """Writes `{prefix}_prob_3dt.tif`, `{prefix}_dp_3dt.tif`, and
    `{prefix}_prob_zavg.tif` under output_dir using existing canonical names."""
```

`diameter` is exposed in the dataclasses (and thus from the widget) because
upstream docs confirm it is still functional in cpsam, even if often optional.
A value of `0` is treated as "omit kwarg" by the runner.

### New widget: `src/cellflow/napari/cellpose_widget.py`

`CellposeWidget(QWidget)` follows the structure of `CellWorkflowWidget`:

- Owns its own `PipelineFilesWidget` with the same Inputs/Outputs lists used
  by `_CellposePanel` today.
- Exposes `output_files_tracker` as a public attribute (same name as the
  current placeholder) so `pipeline_status_from_files` in `main_widget.py`
  keeps working unchanged.
- Embeds `CellposeZavgVizWidget` at the bottom unchanged.
- Builds two stage rows (Nucleus, Cell) with the established
  `stage_header_label` / `_stage_row` / `CollapsibleSection` helpers and the
  `cellpose` stage accent color.
- Shares one status label and one progress bar between the two rows.
- Implements `_set_running_stage("nucleus" | "cell" | None)` to disable the
  other row's controls during a run and swap ▶ ↔ ✕.

### Public widget API (matches sibling workflows)

```python
def refresh(self, pos_dir: Path | None) -> None: ...
def get_state(self) -> dict: ...
def set_state(self, state: dict) -> None: ...
```

`get_state()` returns a dict shaped like:

```python
{
    "nucleus": {"do_3d": bool, "anisotropy": float, "diameter": float,
                "min_size": int, "gamma": float},
    "cell":    {"diameter": float, "min_size": int, "gamma": float},
}
```

Persisted under a new top-level `"cellpose"` key in the project state JSON.

## Behavior

### Run flow (per channel)

1. Read input tif (`0_input/nucleus_3dt.tif` or `cell_3dt.tif`); validate
   shape and that the file exists. If missing, set status `"Missing: ..."`.
2. Build params dataclass from spinboxes / checkbox.
3. If `is_model_loaded()` is False, set status to the device-aware first-load
   message (see Device handling below) before the worker starts heavy work.
4. Launch a `thread_worker`:
   - Calls `get_model()` (slow first time, instant after).
   - Calls `run_nucleus_stack` / `run_cell_stack` with `progress_cb` yielding
     `(done_t, total_t, "Nucleus: frame t/T...")` and `cancel_cb` reading
     the widget's cancel flag.
5. On return, `write_outputs(...)` writes the three canonical tifs; refresh
   `PipelineFilesWidget`; status `"Nucleus Cellpose complete — wrote ..."`.
6. On error, log + show `"Error: <exc>"` in status, clear progress.
7. On cancel between frames, show `"Cancelled."` and clean up worker handle.

### Preview flow (per channel)

Runs synchronously on the main thread (no progress bar, no cancel):

1. Read current `t` and `z` from `viewer.dims.current_step`.
2. Load the channel's input tif and extract:
   - Nucleus + `do_3d=True`:   `stack[t]` → `(Z, Y, X)`
   - Nucleus + `do_3d=False`:  `stack[t, z]` → `(Y, X)`
   - Cell (always 2D):         `stack[t, z]` → `(Y, X)`
3. Call the matching `run_*_frame` helper.
4. Sigmoid prob; compute flow magnitude. Embed into a zero-padded array
   shaped like the full input so napari's time slider still works:
   - 2D preview: `(T, Y, X)` arrays with the slice at `[t]`.
   - 3D nucleus preview: `(T, Z, Y, X)` arrays with the volume at `[t]`.
5. Show layers via `_show_layer` (replace, not duplicate):
   - `"Preview: Nucleus prob"` / `"Preview: Cell prob"` — Image,
     `colormap="viridis"`, `blending="additive"`.
   - `"Preview: Nucleus flow"` / `"Preview: Cell flow"` — Image, magnitude
     only, `colormap="inferno"`, `blending="additive"`.
6. Status reflects mode:
   - `"Preview: nucleus 3D t=3 (Z=15, anisotropy=1.5)"` when `do_3d=True`.
   - `"Preview: nucleus 2D t=3 z=5 (diameter=25)"` otherwise.
   - `"Preview: cell t=3 z=5 (diameter=0)"` for cells.

A 3D nucleus preview can take 5–30s depending on Z and GPU; UI freezes
briefly. If this becomes annoying in practice, lift preview into a
`thread_worker` later. Out of scope for now.

### Device handling

Auto-detect via `torch.cuda.is_available()`. No UI control. Status messages
include the device on the first run of a session, e.g.
`"Loading Cellpose-SAM model on cuda:0 (~10s on first run)..."` or
`"Loading Cellpose-SAM model on cpu (slow)..."`.

### Mutual exclusion

While a stage is running, both rows are locked except the running row's ▶
which becomes a cancel button. `_set_running_stage(None)` restores both.

## Integration with `main_widget.py`

Replace the entire `_CellposePanel` class (lines 36–99) and the construction
site at line 142 with:

```python
from cellflow.napari.cellpose_widget import CellposeWidget
...
self._cellpose_widget = CellposeWidget(self.viewer)
```

`CellposeWidget` exposes `output_files_tracker` (same attribute name as
`_CellposePanel`) so `pipeline_status_from_files` at line 401 keeps working
without modification. `refresh(pos_dir)` and the section wiring stay
unchanged.

## Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    ...
    "cellpose>=4.0",
]
```

Hard dependency, no extras group. This pulls in `torch` transitively.

## Testing

### `tests/segmentation/test_cellpose_runner.py`
- Mock `cellpose.models.CellposeModel` so tests run without torch/GPU.
- Cover:
  - `NucleusParams` / `CellParams` defaults and equality.
  - `get_model()` caches across calls; `is_model_loaded()` reflects state.
  - Gamma application (verify `np.power` is applied with the right exponent).
  - `write_outputs` writes the three canonical filenames with correct dtypes
    and the zavg is computed as `mean(axis=Z)`.
  - `run_nucleus_stack` and `run_cell_stack` iterate over frames in order,
    forward `progress_cb` calls with `(done, total, msg)`, and respect
    `cancel_cb` returning `True` between frames.
  - `run_*_frame` helpers return arrays of the expected shape for both the
    2D and 3D cases.

### `tests/napari/test_cellpose_widget.py`
- Qt smoke test: construct the widget against a fake viewer; toggle the ⚙
  buttons; verify the corresponding params section expands.
- `get_state` / `set_state` round-trip preserves every spinbox + checkbox.
- `refresh(None)` clears state cleanly; `refresh(pos_dir)` propagates to the
  embedded `PipelineFilesWidget` and `CellposeZavgVizWidget`.
- Preview button is wired and calls `cellpose_runner.run_*_frame` (mocked).
- `_set_running_stage("nucleus")` disables the cell row's ⚙ ▷ ▶ and swaps
  the nucleus ▶ to ✕; `_set_running_stage(None)` restores.

No GPU-requiring tests. No actual cellpose model load in tests.
