# Graphcut Cell Boundary Widget Design

## Goal

Implement the existing "Track-Conditioned Boundary Selection" step in
`src/cellflow/napari/cell_boundary_workflow_widget.py` using the current
2D+T multi-label graphcut experiment script. This is an additive integration
inside the current 4b section, not a cleanup or backend refactor.

The widget should let a user run the graphcut/ICM cell-label assignment from
napari, inspect progress, and write the final cell labels to the canonical cell
tracking output path.

## Placement

Use the existing section:

```text
2. Track-Conditioned Boundary Selection
```

Replace the current stub button behavior with graphcut controls and execution.
Do not create a standalone widget file in this pass. Do not modify the broader
main widget composition beyond what is needed for this section to work.

## Inputs

The section should display and validate these files:

```text
2_nucleus/tracked_labels.tif    nucleus track labels used as seeds/track IDs
3_cell/contour_maps.tif         contour signal for geodesic and pairwise terms
3_cell/foreground_masks.tif     foreground domain for graph nodes
3_cell/foreground_scores.tif    optional input for foreground_inverse boundary mode
1_cellpose/cell_dp_3dt.tif      raw Cellpose cell flow field for flow unaries
```

For flow-based unaries, pass the raw Cellpose flow file explicitly:

```text
--flow-field-path <pos_dir>/1_cellpose/cell_dp_3dt.tif
```

The experiment script handles raw Cellpose flow shaped `(T, Z, 2, Y, X)` by
Z-averaging to `(T, 2, Y, X)` without applying the downstream median/Gaussian
filtering used to create `3_cell/filtered_dp.tif`.

## Output

The experiment script writes run artifacts under:

```text
4_cell_graphcut/<timestamp>/
```

including:

```text
cell_labels.tif
params.json
metrics.json
comparison_frame.png
```

After a successful run, the widget copies or writes the produced labels to the
canonical pipeline output:

```text
3_cell/tracked_labels.tif
```

The widget then refreshes file status and adds or updates one napari labels
layer for the graphcut result, using a stable name such as
`cell_labels_graphcut`.

## Controls

Core controls are always visible:

```text
solver          graphcut | icm
unary_mode      flow | geodesic_flow | geodesic | euclidean
boundary_mode   contour | foreground_inverse
n_iters         integer
n_workers       integer
```

Defaults:

```text
solver = graphcut
unary_mode = flow
boundary_mode = contour
n_iters = 1
n_workers = 1
```

Advanced controls live in a collapsible section, expanded by default:

```text
Unary
  alpha_unary
  lambda_geodesic
  lambda_flow

Spatial Pairwise
  lambda_s
  beta_s
  lambda_contour

Temporal
  lambda_t

Solver
  init_mode
  min_round_flips
```

Every control should have a concise tooltip. The `flow` unary tooltip must say
that it uses raw Cellpose flows from `1_cellpose/cell_dp_3dt.tif`.

## Execution

The Run button launches a `thread_worker` that executes the experiment script
through `subprocess`, rather than importing a refactored backend.

The command should be assembled from the current widget state. The default
command shape is:

```bash
python scripts/experiment_cell_2d_t_multilabel_graphcut.py \
  --pos-dir <pos_dir> \
  --solver graphcut \
  --unary-mode flow \
  --flow-field-path <pos_dir>/1_cellpose/cell_dp_3dt.tif \
  --boundary-mode contour \
  --n-iters 1 \
  --n-workers 1 \
  --timestamp <generated> \
  --overwrite
```

When `boundary_mode=foreground_inverse`, also pass:

```text
--foreground-score-path <pos_dir>/3_cell/foreground_scores.tif
```

The worker should surface status text and progress messages where practical.
On process failure, show the subprocess error in the status label and do not
overwrite `3_cell/tracked_labels.tif`.

## State

Implement `get_state()` and `set_state()` for all core and advanced parameters.
Use the existing widget conventions for state dictionaries and Qt control
restoration.

## Out Of Scope

Do not implement a connectivity constraint in this pass.
Do not refactor the experiment script into a production backend module.
Do not add a single-frame preview, because temporal smoothing makes it
misleading.
Do not remove the existing DP boundary-selection backend code or tests.

## Verification

Focused verification should cover:

- widget construction exposes the expected controls and file widgets
- default state matches the selected graphcut defaults
- `get_state()`/`set_state()` round-trip all parameters
- subprocess command assembly includes `--unary-mode flow` and
  `--flow-field-path 1_cellpose/cell_dp_3dt.tif`
- successful completion copies `4_cell_graphcut/<timestamp>/cell_labels.tif`
  to `3_cell/tracked_labels.tif` and updates/creates the napari labels layer
- subprocess failure reports an error and does not overwrite the canonical
  output

