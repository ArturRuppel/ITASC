# Plan: IoU-weighted Cell Tracking via Hypothesis Selection

## Goal

Use the seeded-watershed hypothesis sweep as input to ultrack's ILP to select,
for each nucleus at each timepoint, the single best hypothesis — ranked by
temporal IoU consistency between consecutive frames.

## Background

The `ultrack-exploration` branch contains a working proof-of-concept in
`packages/ultrack/src/cellflow/ultrack/stages/cell_tracking.py` with the right
architecture: direct node injection into ultrack's SQLite DB + ILP constraint
`set_nodes_sum(per_nucleus_nodes, 1)`.

The critical missing piece: all link weights were hardcoded to `1.0`. The ILP
had no signal to prefer temporally consistent hypotheses.

## Architecture

Inputs:
- `4_seeded_watershed/hypotheses/hypothesis_NNN.tif` — (T, Z, Y, X) or
  (T, Y, X) label stacks; label values equal nucleus IDs
- `CellTrackingConfig` parameters

Pipeline stages (matching the existing `run_segmentation / run_linking /
run_solve` pattern in `tracking.py`):

1. **Create Database** — insert nodes + IoU-weighted links into ultrack DB
2. **Solve** — run ILP with `set_nodes_sum` constraints; export tracked labels

Output: `5_cell_tracking/tracked_labels.tif`

## Implementation Steps

### Step 1 — Port and clean `cell_tracking.py`

File: `packages/ultrack/src/cellflow/ultrack/stages/cell_tracking.py`

Port from `ultrack-exploration` branch, removing:
- Any dependency on `seeded_contours`
- The `PLAN_ultrack_cleanup.md` detour
- CSV export (not needed, only tracked_labels.tif)

Keep:
- `_build_cell_tracking_ultrack_config`
- `_load_hypothesis_stacks`
- `_export_hypothesis_projections`
- `_insert_hypothesis_nodes` (unchanged)
- `run_solve` (strip CSV export)
- `__main__` CLI

### Step 2 — Replace `_insert_nucleus_identity_links` with IoU-weighted version

Replace the flat `weight=1.0` implementation with:

```python
def _insert_iou_links(engine, stacks, cfg):
    """Insert links weighted by IoU^power between consecutive frames.

    For each (t, t+1) pair and each (h1, h2) hypothesis pair, compute
    per-nucleus IoU using the bincount overlap trick (one O(H*W) pass per
    hypothesis pair per timepoint), then insert LinkDB entries.
    """
    T = stacks[0].shape[0]
    max_label = max(s.max() for s in stacks) + 1

    for t in range(T - 1):
        # frames_t[h] = 2D/3D label array for hypothesis h at timepoint t
        # (project Z via max if 4-D, or use directly if 3-D)
        frames_t  = [_project(s[t])  for s in stacks]
        frames_t1 = [_project(s[t+1]) for s in stacks]

        links = []
        for h1, f1 in enumerate(frames_t):
            area1 = np.bincount(f1.ravel(), minlength=max_label)

            for h2, f2 in enumerate(frames_t1):
                area2 = np.bincount(f2.ravel(), minlength=max_label)

                # Full overlap matrix in one pass
                flat = f1.ravel().astype(np.int64) * max_label + f2.ravel()
                overlap = np.bincount(flat, minlength=max_label * max_label)
                overlap = overlap.reshape(max_label, max_label)

                # Per-nucleus IoU: diagonal entries (same nucleus n → n)
                inter = np.diag(overlap)  # shape (max_label,)
                union = area1 + area2 - inter
                iou = np.where(union > 0, inter / union, 0.0)

                for n in range(1, max_label):
                    if iou[n] < cfg.min_link_iou:
                        continue
                    weight = float(iou[n] ** cfg.power)
                    src = _node_id(h1, t,   n, T, max_label)
                    tgt = _node_id(h2, t+1, n, T, max_label)
                    links.append(LinkDB(source_id=src, target_id=tgt, weight=weight))

        _flush_links(engine, links)
```

Key points:
- `np.diag(overlap)` extracts same-nucleus overlaps — no per-nucleus Python loop
  needed for the heavy part; the outer `for n` loop only filters/appends rows
  that survive the IoU threshold
- For 3-D (Z, Y, X) label stacks, project via `labels.max(axis=0)` before
  computing overlap (matches what the scrapped branch did for export)
- Batch-insert links in chunks of 10 000 (already done in scrapped branch)

### Step 3 — Update `CellTrackingConfig`

File: `packages/ultrack/src/cellflow/ultrack/config.py`

Add:
```python
min_link_iou: float = 0.1   # links below this IoU are dropped
power: float = 4.0           # IoU^power link weight (matches TrackingConfig)
```

`power` already exists in the scrapped branch config; `min_link_iou` is new.

### Step 4 — Port and clean the widget

File: `packages/napari-plugin/src/cellflow/napari/ultrack_widgets/cell_tracking_widget.py`

Port from `ultrack-exploration`, stripping:
- The seeded_contours / s02d tab
- `_db_candidates_df` preview dataframe
- Tracks CSV loading / visualization
- `get_tracks_layer` call

Keep:
- Hypothesis directory picker (reads from `4_seeded_watershed/hypotheses/`)
- Create Database button + progress
- Solve button + progress
- Load Labels button → napari Labels layer

### Step 5 — Wire into plugin and paths

- Register `CellTrackingWidget` in `_plugin.py`
- Add `cell_tracking` stage dir to `paths.py` → `5_cell_tracking/`

## Config parameters (final `CellTrackingConfig`)

| Field | Default | Notes |
|---|---|---|
| `min_area` | 100 | Skip tiny candidates |
| `max_area` | 1_000_000 | Skip huge candidates |
| `min_link_iou` | 0.1 | Drop links below this threshold |
| `power` | 4.0 | Link weight = IoU^power |
| `appear_weight` | -0.001 | Ultrack ILP penalty |
| `disappear_weight` | -0.001 | Ultrack ILP penalty |
| `division_weight` | -0.001 | Ultrack ILP penalty (effectively disabled) |
| `link_function` | "power" | Ultrack link scoring function |
| `bias` | 0.0 | Ultrack ILP bias |
| `solution_gap` | 0.001 | ILP optimality gap |
| `time_limit` | 36000 | ILP time limit (seconds) |
| `window_size` | 0 | Ultrack sliding window (0 = full) |

## What is NOT needed

- Ultrack's `run_segmentation` / `run_linking` stages — bypassed entirely
- foreground.tif / contours.tif from seeded watershed — not used here
- CSV export

## Open questions / future work

- Deduplication of near-identical cells across hypotheses (IoU > 0.95) — deferred
- Parallelism for the IoU computation loop — deferred (embarrassingly parallel
  over timepoints, use ProcessPoolExecutor when scale demands it)
- 3-D vs 2-D: current plan projects Z via max; could do full 3-D IoU if needed
