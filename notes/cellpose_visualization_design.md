# Cellpose Visualization Design

Date: 2026-05-16

## Purpose

Provide a more interpretable view of the Cellpose pipeline outputs in napari. The current `PipelineFilesWidget` exposes the raw `prob` (logits) and `dp` (flow) tifs as plain Image layers, which are hard to read. This feature adds an opt-in visualization that shows the **sigmoid-transformed probability** as a heatmap with the **flow field** rendered as a vector overlay.

The visualization is available in two modes — **z-avg** and **3D+t** — and for both **nuclei** and **cells** channels.

## Placement

The trigger UI lives in `HpcCellposeWidget` (`src/cellflow/napari/hpc_cellpose_widget.py`), placed below the existing status indicators (`input_status_lbl` and `status_lbl`, lines 113 and 122). Two `QGroupBox` sections — one for nuclei, one for cells — keep the per-channel state isolated and the layout symmetric.

Reasoning: the cellpose widget is where the user thinks about cellpose outputs and runs the pipeline. The viz controls belong next to the run button, not inside the downstream nucleus/cell workflow widgets.

## Architecture

### New module: `src/cellflow/napari/cellpose_visualization.py`

Pure-numpy / napari helpers, no Qt. Mirrors how `artifact_visualization.py` is organized so that the math stays testable without bringing up the Qt event loop.

```python
def load_sigmoid_prob(
    output_dir: Path,
    channel: Literal["nucleus", "cell"],
    mode: Literal["zavg", "3dt"],
) -> np.ndarray:
    """Read prob_3dt.tif, sigmoid, then for zavg average over Z.

    Returns:
        zavg: (T, Y, X) float32 in [0, 1]
        3dt:  (T, Z, Y, X) float32 in [0, 1]
    """

def load_flow_vectors(
    output_dir: Path,
    channel: Literal["nucleus", "cell"],
    mode: Literal["zavg", "3dt"],
    *,
    stride: int,
    scale: float,
) -> np.ndarray:
    """Read dp_3dt.tif, average over Z for zavg mode, subsample by stride,
    return a napari Vectors array.

    dp_3dt has shape (T, Z, 2, Y, X) where the 2 channels are (dy, dx).

    Returns:
        zavg: (N, 2, 3) array of [[t, y, x], [0, scale*dy, scale*dx]]
        3dt:  (N, 2, 4) array of [[t, z, y, x], [0, 0, scale*dy, scale*dx]]
    """

def add_cellpose_viz_layers(
    viewer: napari.Viewer,
    output_dir: Path,
    channel: Literal["nucleus", "cell"],
    mode: Literal["zavg", "3dt"],
    *,
    stride: int,
    scale: float,
) -> list[napari.layers.Layer]:
    """Remove any existing viz layers for (channel, mode) and add fresh ones.

    Layer names:
        Cellpose viz: <channel> prob (<mode>)
        Cellpose viz: <channel> flow (<mode>)
    """
```

The `add_*` helper is the only entry point the widget code calls. It does layer cleanup (so re-clicking refreshes) and applies sensible Image defaults (probability colormap, contrast 0–1, blending=translucent).

### UI changes in `hpc_cellpose_widget.py`

After `layout.addWidget(self.status_lbl)`, add two `QGroupBox`es:

```
┌─ Nuclei visualization ──────────────────────┐
│  Stride:        [16  ]   Vector scale: [1.0]│
│  [ Load z-avg ]   [ Load 3D+t ]             │
│  status: loaded nuclei z-avg (1024 vectors) │
└─────────────────────────────────────────────┘
┌─ Cells visualization ───────────────────────┐
│  Stride:        [16  ]   Vector scale: [1.0]│
│  [ Load z-avg ]   [ Load 3D+t ]             │
│  status: —                                  │
└─────────────────────────────────────────────┘
```

Each section is independent state. Buttons are disabled when `output_dir` is empty or the required tifs are missing, and the status line shows what was loaded or what's missing.

### State persisted in `get_state` / `set_state`

The two parameters per channel (`stride`, `scale`) are added to the existing widget state dict so they round-trip with the rest of the cellpose widget settings.

## Data flow

**z-avg mode:**
1. Read `<output_dir>/<channel>_prob_3dt.tif` → `(T, Z, Y, X)` logits.
2. `sigmoid` per voxel → `(T, Z, Y, X)` in [0, 1].
3. Mean over Z → `(T, Y, X)` probability heatmap.
4. Read `<output_dir>/<channel>_dp_3dt.tif` → `(T, Z, 2, Y, X)`.
5. Mean over Z → `(T, 2, Y, X)`. Subsample y/x by `stride`, scale by `scale`.
6. Build `(N, 2, 3)` Vectors array; add to viewer.

**3D+t mode:**
1. Read prob_3dt → sigmoid → `(T, Z, Y, X)` heatmap (Image layer).
2. Read dp_3dt → `(T, Z, 2, Y, X)`. Subsample y/x by `stride`. Per Z-slice, build vectors with `dz=0`.
3. Vectors array shape `(N, 2, 4)`; add to viewer.

## Parameters

| Param | Default | Purpose |
|------|--------|--------|
| `stride` | `16` | Subsample step in Y/X. At 512×512 this is ~1024 vectors per Z-slice → ~8200 per timepoint in 3D+t. |
| `scale` | `1.0` | Multiplier on each vector's length so users can tune visibility. |

No magnitude threshold for v1 — keep simple. Can be added later if vectors in low-probability regions are too noisy.

## Layer naming

- `Cellpose viz: nucleus prob (z-avg)` / `(3D+t)`
- `Cellpose viz: nucleus flow (z-avg)` / `(3D+t)`
- `Cellpose viz: cell prob (z-avg)` / `(3D+t)`
- `Cellpose viz: cell flow (z-avg)` / `(3D+t)`

Distinct from the raw `Nucleus prob z-avg` / `Cell prob 3D+t` layers loaded via `PipelineFilesWidget`, so both can coexist if the user wants to compare.

Re-clicking `Load z-avg` removes the existing viz layers for that (channel, mode) before adding new ones, so changing stride/scale and re-loading doesn't pile up duplicates.

## Error handling

- Missing `output_dir` → buttons disabled, status shows "no output directory selected".
- Missing `<channel>_prob_3dt.tif` or `<channel>_dp_3dt.tif` → button enabled but click reports "missing: <filename>" in the status line, no layers added.
- `dp_3dt.tif` shape inconsistent with `prob_3dt.tif` (e.g. different T/Z/Y/X) → status line shows the shape mismatch, no layers added.

## Testing

Tests live in `tests/napari/test_cellpose_visualization.py`. Cover:

- `load_sigmoid_prob` z-avg returns `(T, Y, X)` in [0, 1]; sigmoid-then-mean differs from mean-then-sigmoid (regression check that we got the order right).
- `load_sigmoid_prob` 3D+t returns `(T, Z, Y, X)` in [0, 1].
- `load_flow_vectors` zavg averages over Z and respects stride; output shape is `(N, 2, 3)` with `N == T * (Y // stride) * (X // stride)`.
- `load_flow_vectors` 3D+t output shape is `(N, 2, 4)` with dz components all 0.
- `add_cellpose_viz_layers` re-call removes prior layers with the same (channel, mode) before adding new ones.
- `add_cellpose_viz_layers` raises a clear error (or returns empty) when the required tifs are missing.

Widget-level tests in `tests/napari/` for `HpcCellposeWidget` cover button enable/disable based on file presence and `get_state` / `set_state` round-trip including the new params.

## Out of scope

- Any change to the upstream HPC cellpose pipeline scripts.
- Reactive auto-update of viz layers when project / position changes — user re-clicks Load.
- Magnitude-based vector filtering — deferred until we see whether it's needed in practice.
- Replacing or removing the existing raw prob/dp entries in `PipelineFilesWidget` / `nucleus_workflow_widget.py` / `cell_workflow_widget.py`.
