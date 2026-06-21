# COV2D NLS-mCherry subpopulation analysis (2026-06-21)

A record of the questions asked of the COV2D dataset, the statistics used (and why
they are sound), what was found, and how to reproduce it on new data. The reusable
code lives in `cellflow.aggregate_quantification.analysis`.

## Dataset

- **Condition:** VimKO + NLS-mCherry. **Replicates:** N = 3 experiments
  (`experiment_id` ∈ {COV2D-01, COV2D-02, COV2D-03}).
- **Classifier:** per-cell `class_label` ∈ {`negative`, `positive`} = nuclear
  NLS-mCherry status. ~82 k cell-frames, ~1.7 k tracked cells.
- Tables: the standard `aggregate_quantification/*.csv` (cell/nucleus shape &
  dynamics, contacts, neighbour counts/enrichment, contact-type z-scores, density).

## Statistical principles used throughout

1. **Replicate is the unit of inference.** Cell-frames are massively
   pseudoreplicated (frames within a cell, cells within a position, positions
   within an experiment). Every test summarises to **one value per replicate**,
   then tests across N = 3 — never over pooled cells.
2. **N = 3 is the binding constraint.** Wilcoxon (n=3) floors at p≈0.25; paired-t
   bottoms near p≈0.10. Most "ns" results are **underpowered, not null** — judge by
   effect size + sign consistency, not the p alone.
3. **Test follows the design.** Engine auto-inference picks paired-vs-independent and
   parametric-vs-robust from the spine; we pin paired-t to match the SuperPlots
   paper where wanted (`_override`).

## Questions & findings

### 1. Cell & nucleus **shape** by class (paired t on replicate means)
- **Cell shape:** the elongation metrics (eccentricity g≈1.03, aspect ratio g≈1.02,
  extent g≈−0.93) show a **large, consistent** difference — negative cells more
  elongated — but ns at N=3 (p≈0.10–0.14). Size metrics flat.
- **Nucleus shape:** effects roughly half the cell's (max |g|≈0.5), no coherent
  signal. → The label tracks **cell-body** morphology, not nucleus morphology.

### 2. Cell & nucleus **dynamics** by class
- Speed & net displacement trend higher in **negative** cells (g≈0.4–0.6, ns).
  Cell and nucleus dynamics agree (nucleus carried by the cell).

### 3. Do cells cluster by label?  ⭐ the solid positive result
- **Homotypic contact enrichment 1.06×, p = 0.009** (one-sample vs a within-frame
  label-shuffle null, replicate level); heterotypic depleted. Same direction in
  **all 3 replicates × all contact types**.
- **Corroborated by an independent analytic null** (`neighbour_enrichment`): 1.05×,
  p = 0.003 — the two nulls agree to ~0.01×, so the result is not a null-model
  artifact. Significant despite N=3 because between-replicate scatter is tiny.
- Magnitude is **modest** (~5–7 % excess homotypic contact): a slight same-label
  preference, not strong segregation.

### 4. Does speed depend on density, and differ by class?
- **Speed vs neighbour count** (clean local-density proxy): negative correlation in
  all 6 replicate×class groups (r≈−0.17) — denser → slower (jamming) — a **trend**
  (p≈0.09–0.12, N=3). **No difference between classes** (Δr≈−0.02, p=0.86).
- **Caveat:** speed vs *cell area* is significant (p=0.006) but **confounded** —
  larger cells move slower is spreading↔motility, not crowding. Area is *not* a
  clean density proxy; use neighbour count.

## Synthesis

NLS-mCherry-**negative** cells are more **elongated** and slightly more **motile**;
their **nuclei are shape-unchanged**. Cells show **modest, reproducible homotypic
clustering**. Local **density slows cells equally** in both classes. The only result
that is both significant and cross-validated is the **clustering** (§3); the rest are
consistent trends limited by N=3 — worth confirming with more replicates (a paired
effect of g≈1, as in §1, needs ≈5 replicates for 80 % power).

## Reproduce

```bash
# all recorded analyses for a dataset (figures + .iris docs):
python -m cellflow.aggregate_quantification.analysis \
    /path/to/<dataset>/aggregate_quantification \
    /path/to/<dataset>/export
```

Outputs: `export/figures/analysis/{label_clustering,label_clustering_corroboration,
speed_um_per_s_vs_n_neighbors}.{png,svg}` (editable text) and
`export/iris/{label_clustering,speed_um_per_s_vs_n_neighbors}.iris`.

Programmatic use (any columns / classifier / replicate key):

```python
from cellflow.aggregate_quantification.analysis import (
    label_clustering_report, metric_correlation_report)

label_clustering_report(agg_dir, out_dir)                       # §3
metric_correlation_report(agg_dir, out_dir,                     # §4
    x="neighbor_count.n_neighbors", y="cell_dynamics.speed_um_per_s",
    split="class_label")
```

The replicate-level stats (`replicate_correlation`, `homotypic_enrichment`) are in
`analysis/stats.py` and pinned by `tests/aggregate_quantification/analysis/test_stats.py`.

**Important:** the `.iris` docs are for interactive viewing only — their GUI p-value
is **pooled / pseudoreplicated** (the provenance records this). The valid inference is
always the replicate-level figure.
