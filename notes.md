# Plan: Ultrack database test with foreground/contour threshold sweep

## Goal

Test whether a merged multi-input Ultrack database improves candidate coverage for:

```text
/home/aruppel/Data/2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/
```

The experiment will generate multiple Ultrack candidate databases from threshold-swept foreground and contour inputs, then merge them into one final `data.db` so the Ultrack ILP can choose among candidate segmentations from different input settings.

## Output location

Create a separate experiment folder inside the data directory, for example:

```text
/home/aruppel/Data/2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/2_nucleus/ultrack_threshold_sweep_experiment/
```

Suggested structure:

```text
ultrack_threshold_sweep_experiment/
├── inputs/
│   ├── nucleus_prob_sigmoid_zavg.tif
│   ├── foreground_thr_0.1.tif
│   ├── foreground_thr_0.3.tif
│   ├── foreground_thr_0.5.tif
│   ├── contours_raw.tif
│   ├── contours_thr_0.5.tif
│   └── manifest.json
├── variants/
│   ├── fg_0.1_contour_raw/
│   │   └── data.db
│   ├── fg_0.1_contour_0.5/
│   │   └── data.db
│   ├── fg_0.3_contour_raw/
│   │   └── data.db
│   ├── fg_0.3_contour_0.5/
│   │   └── data.db
│   ├── fg_0.5_contour_raw/
│   │   └── data.db
│   └── fg_0.5_contour_0.5/
│       └── data.db
├── merged_ultrack_workdir/
│   └── data.db
└── report.json
```

## Inputs

Use the existing position directory:

```text
POS_DIR=/home/aruppel/Data/2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00
```

Expected source files:

```text
$POS_DIR/1_cellpose/nucleus_prob_3dt.tif
$POS_DIR/2_nucleus/contour_maps.tif
```

The user referred to `contours.tif`; before implementing, verify whether the actual file is named:

```text
2_nucleus/contour_maps.tif
```

or whether another contour file exists in this dataset.

## Foreground construction

For the foreground input:

1. Load the nucleus probability map:

   ```text
   1_cellpose/nucleus_prob_3dt.tif
   ```

2. Apply a sigmoid transform to the probability map.

   Proposed initial transform:

   ```python
   sigmoid = 1.0 / (1.0 + np.exp(-prob))
   ```

   Caveat: if `nucleus_prob_3dt.tif` is already normalized to `[0, 1]`, this transform maps values into roughly `[0.5, 0.73]`, which may make thresholds `0.1` and `0.3` trivial. Before running the full experiment, inspect the min/max/percentiles of the source image. If it is already `[0, 1]`, consider a centered sigmoid instead:

   ```python
   sigmoid = 1.0 / (1.0 + np.exp(-k * (prob - midpoint)))
   ```

   with documented `k` and `midpoint`.

3. Average over z.

   Expected source shape is likely one of:

   ```text
   (T, Z, Y, X)
   (T, Y, X)
   ```

   If shape is `(T, Z, Y, X)`, compute:

   ```python
   nucleus_prob_sigmoid_zavg = sigmoid.mean(axis=1)
   ```

   If already `(T, Y, X)`, keep it as-is.

4. Create binary/float foreground maps by thresholding the z-average at:

   ```text
   0.1, 0.3, 0.5
   ```

   Save:

   ```text
   inputs/foreground_thr_0.1.tif
   inputs/foreground_thr_0.3.tif
   inputs/foreground_thr_0.5.tif
   ```

5. Also save the continuous foreground source:

   ```text
   inputs/nucleus_prob_sigmoid_zavg.tif
   ```

## Contour construction

Use the existing contour image as the contour source.

1. Load:

   ```text
   2_nucleus/contour_maps.tif
   ```

2. Save a raw copy:

   ```text
   inputs/contours_raw.tif
   ```

3. Create a 50% thresholded contour version:

   ```python
   contours_thr_0_5 = np.where(contours < 0.5, 0.0, contours)
   ```

4. Save:

   ```text
   inputs/contours_thr_0.5.tif
   ```

## Variant grid

Run Ultrack segmentation for all combinations:

| Variant | Foreground | Contours |
|---|---:|---|
| 1 | threshold 0.1 | raw |
| 2 | threshold 0.1 | threshold 0.5 |
| 3 | threshold 0.3 | raw |
| 4 | threshold 0.3 | threshold 0.5 |
| 5 | threshold 0.5 | raw |
| 6 | threshold 0.5 | threshold 0.5 |

Each variant writes to its own temporary/final variant directory under:

```text
ultrack_threshold_sweep_experiment/variants/
```

Do **not** let variants write concurrently into the same SQLite database.

## Ultrack segmentation per variant

For each variant:

1. Build an Ultrack `MainConfig` using the existing CellFlow config machinery:

   ```python
   from cellflow.tracking_ultrack.ingest import _build_ultrack_config
   from cellflow.tracking_ultrack.config import TrackingConfig
   ```

2. Use a per-variant working directory:

   ```text
   variants/fg_{fg_threshold}_contour_{raw_or_0.5}/
   ```

3. Run:

   ```python
   from cellflow.tracking_ultrack.db_build import _run_ultrack_segment

   _run_ultrack_segment(
       foreground_variant,
       contour_variant,
       ultrack_cfg,
       cfg,
   )
   ```

4. Confirm that each variant produced:

   ```text
   variants/.../data.db
   ```

## Parallelization

Safe parallelism:

```text
parallel workers → independent variant directories → independent data.db files
```

Unsafe parallelism:

```text
parallel workers → same final merged data.db
```

If parallelizing, use a small worker count first, e.g. 2 or 3, because Ultrack segmentation may itself use workers and memory. Keep the final merge single-threaded and deterministic.

## Merge into final database

After all six variant databases are built, merge them into:

```text
ultrack_threshold_sweep_experiment/merged_ultrack_workdir/data.db
```

Use the existing merge primitive:

```python
from cellflow.tracking_ultrack.multi_threshold import merge_ultrack_databases

merge_ultrack_databases(
    source_db_paths=[...six variant data.db paths...],
    output_db_path=experiment_dir / "merged_ultrack_workdir" / "data.db",
    frame_shape=(height, width),
)
```

The merge should:

1. Read all `NodeDB` rows from each source DB.
2. Remap node IDs globally.
3. Copy within-source overlaps.
4. Recompute cross-source overlaps from node masks.
5. Write one final merged Ultrack-compatible `data.db`.

## Post-merge scoring and linking

After merging, run the standard CellFlow downstream database steps on the merged DB:

1. Score node probabilities:

   ```python
   from cellflow.tracking_ultrack.seed_prior import write_seed_prior_node_probs

   write_seed_prior_node_probs(
       merged_workdir,
       POS_DIR / "1_cellpose" / "nucleus_prob_zavg.tif",
       cfg,
   )
   ```

   If `nucleus_prob_zavg.tif` does not exist, either create it from the same sigmoid/z-average source or decide explicitly which intensity image should be used for node scoring.

2. Link candidates:

   ```python
   from cellflow.tracking_ultrack.linking import run_linking

   for step, total, label in run_linking(merged_workdir, cfg):
       print(step, total, label)
   ```

3. Optional: run solve/export only after validating that the merged DB looks sane.

## Validation checks before solving

Before running the ILP solve, inspect the merged database:

1. Count nodes per source variant.
2. Count total merged nodes.
3. Count within-source overlaps and cross-source overlaps.
4. Verify no duplicate `NodeDB.id` values.
5. Verify node masks deserialize correctly.
6. Verify that each frame has candidates.
7. Open the DB browser, if possible, and inspect candidate coverage at representative frames.

Recommended report fields in `report.json`:

```json
{
  "pos_dir": ".../pos00",
  "foreground_thresholds": [0.1, 0.3, 0.5],
  "contour_modes": ["raw", "threshold_0.5"],
  "variants": [
    {
      "name": "fg_0.1_contour_raw",
      "foreground_path": "inputs/foreground_thr_0.1.tif",
      "contour_path": "inputs/contours_raw.tif",
      "db_path": "variants/fg_0.1_contour_raw/data.db",
      "node_count": null,
      "overlap_count": null
    }
  ],
  "merged_db_path": "merged_ultrack_workdir/data.db",
  "merged_node_count": null,
  "merged_overlap_count": null,
  "cross_source_overlap_count": null
}
```

## Expected risks / caveats

1. **Sigmoid ambiguity**

   Need to inspect `nucleus_prob_3dt.tif` value range. A plain sigmoid may be inappropriate if values are already probabilities.

2. **Foreground double-thresholding**

   If we pass binary foreground maps to Ultrack, set or interpret `cfg.seg_foreground_threshold` carefully. Otherwise foreground may effectively be thresholded twice.

3. **Hierarchy semantics**

   The merged database is best understood as a merged candidate pool with overlap constraints, not a single mathematically unified watershed hierarchy.

4. **SQLite concurrency**

   Do not write multiple variants into the same final DB in parallel. Parallelize only independent variant DB construction.

5. **Database browser behavior**

   Because source hierarchies are merged, hierarchy-cut browsing may not represent one coherent dendrogram across all variants. Solver behavior should still be the main validation target.

6. **Ultrack environment**

   The current shell may not have `ultrack` importable. Run this in the CellFlow conda environment where Ultrack is installed.

## Implementation sequence

1. Verify source file names and shapes in `pos00`.
2. Inspect intensity ranges for `nucleus_prob_3dt.tif` and `contour_maps.tif`.
3. Decide exact sigmoid formula based on observed intensity range.
4. Create experiment directory.
5. Save transformed foreground images and contour images under `inputs/`.
6. Run the six independent Ultrack segmentation variants.
7. Merge variant DBs into `merged_ultrack_workdir/data.db`.
8. Score node probabilities on the merged DB.
9. Link candidates on the merged DB.
10. Save `manifest.json` and `report.json`.
11. Inspect DB sanity metrics and representative frames.
12. Only then run solve/export if the candidate database looks reasonable.
