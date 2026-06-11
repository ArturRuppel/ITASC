# Track Dynamics: Quantifier + Plots — Design Spec

Date: 2026-06-11

> **Status: design.** Aligns the *exact* calculations before building. Follows the
> quantifier seam established by Cell Shape
> (`notes/2026-06-10-cell-shape-quantifier-and-table-explorer-design.md`) and the
> `Quantifier` registry in `aggregate_quantification/quantifier.py`.

The next Aggregate Quantification quantity after the shape family: **motion**,
read off the tracked label stacks. Where shape measures a cell's form per frame,
**track dynamics** measures how its centroid *moves* over time — speed,
persistence, mean-square displacement, and tissue-scale collective metrics
(velocity correlation length, alignment order parameter). It is the
"nucleus-track kinematics / tissue dynamics" item named in `quantifier.py`'s
docstring and `TODO.md` ("nucleus track analysis … tissue dynamics").

## Decisions (locked with the user)

1. **Substrate: both cell and nucleus, selectable.** Two thin quantifiers
   (`cell_dynamics`, `nucleus_dynamics`) over **one label-agnostic core** — the
   exact `cell_shape` / `nucleus_shape` twin pattern. Each differs only by which
   `PositionInputs` field it reads and its output filename. Nuclei give compact,
   point-like centroids (the robust motility default); cells match the shape
   substrate. The studio surfaces whichever the position has.
2. **Units are physical.** `pixel_size_um` (µm/px) and a **new**
   `time_interval_s` (s/frame) are threaded through `PositionInputs`.
   `time_interval_s` resolves the same way `pixel_size_um` does — from
   `cellflow_config.json` `metadata.time_interval_s` (the key
   `main_widget.py` already writes), with a TIFF `finterval` fallback and a
   manual override. A position with no resolvable interval is not buildable.
3. **Gaps are honoured.** Tracks can miss frames. Every per-step quantity divides
   by the *actual* elapsed time `Δf · dt`, never by `dt` blindly. A "step" is
   between two consecutive **present** frames of a track.
4. **MSD → power-law.** Ensemble time-averaged MSD, fit `MSD = 2·d·D·τ^α`
   (`d = 2`) in log-log → diffusivity `D` (µm²/s) and anomalous exponent `α`
   (≈1 diffusive, →2 ballistic/persistent, <1 confined). No PRW/Fürth fit; no
   per-track MSD.
5. **Persistence → directionality ratio + DAC persistence time.** The
   per-track summary carries the confinement ratio (always) and a
   directional-autocorrelation persistence time `P` (s). No mean-turning-angle /
   arrest-coefficient columns (deferred).
6. **Correlation length → 1/e decay.** `ξ` is where the drift-subtracted velocity
   correlation `C(r)` decays to `1/e`. Robust when `C(r)` never crosses zero.

## Inputs

Per position, from a single tracked label stack `T×Y×X` (`label == track_id`,
constant across frames):

- `s = pixel_size_um` (µm/px), `dt = time_interval_s` (s/frame).
- Per track `g`: centroid trajectory `r_k = (x_k, y_k)` in µm at frame `f_k`,
  from `regionprops(frame).centroid` (same per-frame loop as shape's
  `_extract_shape_columns`). Frames where a label is absent are simply missing
  from that track's series (gaps).

## Calculations (exact)

### A. Instantaneous table — one row per `(frame, cell_id)` → the `object_table`

Forward differences between consecutive present frames, assigned to the earlier
frame:

```
Δf      = f_{k+1} − f_k                  (frames; ≥1, >1 across a gap)
v_k     = (r_{k+1} − r_k) / (Δf · dt)    µm/s   → vx_um_per_s, vy_um_per_s
speed   = |v_k|                          µm/s   → speed_um_per_s
net_disp= |r_k − r_0|                    µm     → net_disp_um  (from track start)
```

Columns: `frame, cell_id, x_um, y_um, vx_um_per_s, vy_um_per_s, speed_um_per_s,
net_disp_um`. The last present frame of a track has no forward velocity → NaN
velocity/speed (its position/net_disp are still real). This table feeds the
generic plotting/pooling/CSV path and the NLS subpopulation join (by
`(frame, cell_id)`), exactly like the shape tables.

### B. Per-track summary — one row per `cell_id`

```
T          = Σ Δf_k · dt                 total tracked time (excludes gap-spanned? NO:
                                          gaps count — elapsed time is real)
L          = Σ |r_{k+1} − r_k|           path length (µm)
D_net      = |r_last − r_first|          net displacement (µm)
curv_speed = L / T                       µm/s   (mean instantaneous speed)
net_speed  = D_net / T                   µm/s
dir_ratio  = D_net / L  ∈ [0,1]          directionality / confinement ratio
P          = DAC fit (below)             s      persistence time
```

Columns: `cell_id, n_frames, n_gaps, frame_start, frame_end, duration_s,
path_length_um, net_displacement_um, curvilinear_speed_um_per_s,
net_speed_um_per_s, directionality_ratio, persistence_time_s`. Tracks shorter
than `min_track_frames` (param, default 3) are dropped from the summary (too
short for a meaningful ratio); they still appear in the instantaneous table.

**DAC persistence time (per track).** Unit step directions
`û_k = v_k / |v_k|`. For lag `n` (in frames), the directional autocorrelation is
`C_g(n) = mean over valid origins of û_k · û_{k+n}`, using only origin pairs
where both steps are real (no gap inside either step). Fit
`C_g(n) = exp(−(n·dt)/P)` by a linear fit of `ln C_g` vs `n·dt` over the
positive-`C` prefix (`C>0`, `n ≤ n_max`); `P = −1/slope`. NaN when too few
points or a non-negative slope (no decay).

### C. Ensemble curves — per position

**MSD(τ).** Ensemble of time-averaged, overlapping origins. For lag `n`:
`MSD(n) = mean over all tracks and all present origin pairs (k, k+n) of
|r_{k+n} − r_k|²` — pairs separated by exactly `n` frames (gaps break a pair).
`n` runs `1 … n_max`, `n_max = max(1, floor(¼ · longest_track_span))`.
Table `msd/table`: `lag_s (= n·dt), msd_um2, n_samples, sem`.
Fit `log MSD = log(2dD) + α·log τ` over `n_samples ≥ min_samples` → `D, α`
(stored as `msd` group attrs `D_um2_per_s`, `alpha`, plus `r2`).

**DAC(τ) ensemble.** Same `C(n)` as B but pooled over all tracks' origin pairs.
Table `dac/table`: `lag_s, dac, n_samples, sem`; ensemble `persistence_time_s`
attr from the same exp-fit recipe as B.

### D. Collective table — one row per `frame` → per position

Per frame `f`, over cells present at both `f` and `f+1` (so each has a real
`v_i`):

```
V        = mean_i v_i                    drift (µm/s)
δv_i     = v_i − V                       fluctuation
order    = | mean_i ( v_i / |v_i| ) | ∈ [0,1]   polar alignment (uses raw v_i)
```

**Velocity correlation function.** Over unordered cell pairs `(i,j)` present at
`f`, bin by separation `|r_i − r_j|` (bin width `corr_bin_um`, param, default
`= 1 × typical cell diameter ≈ resolved from median NN distance`, see below):
`C(r) = ⟨ δv_i · δv_j ⟩_{bin r} / ⟨ δv_i · δv_i ⟩` (normalised so `C→1` as
`r→0`). **Correlation length** `ξ_f` = the `r` where `C(r)` first falls to `1/e`,
by linear interpolation between bin centres; NaN if it never does (within the
field) or `< min_pairs` cells.

Table `collective/table`: `frame, n_cells, order_param, corr_length_um,
nn_distance_um` (`nn_distance_um` = median nearest-neighbour distance that
frame — context for `ξ` and a natural length scale). The `C(r)` curve itself is
pooled/averaged across frames and stored as `corr_curve/table`
(`separation_um, corr, n_pairs`) for the plugin's curve plot.

## Persistence (artifact)

One **HDF5** per substrate — `aggregate_quantification/cell_dynamics.h5` /
`nucleus_dynamics.h5` — because the quantity is several heterogeneous tables
(unlike shape's single flat CSV), mirroring how contacts stores `cells/`,
`edges/`, `t1_events/`. Groups:

```
instantaneous/table   frame, cell_id, x_um, y_um, vx_um_per_s, vy_um_per_s,
                       speed_um_per_s, net_disp_um
tracks/table          cell_id, … (summary, §B)
msd/table             lag_s, msd_um2, n_samples, sem      (+ attrs D, alpha, r2)
dac/table             lag_s, dac, n_samples, sem          (+ attr persistence_time_s)
collective/table      frame, n_cells, order_param, corr_length_um, nn_distance_um
corr_curve/table      separation_um, corr, n_pairs
provenance            attrs: quantity_id, source/label paths, pixel_size_um,
                       time_interval_s, params, created_at, cellflow_version
```

`read()` returns the dict-of-tables. `object_table()` returns
`instantaneous/table` (the only `(frame, cell_id)` table) so the generic
plotting/pooling/CSV/subpopulation-join layer works unchanged. The other tables
are surfaced by the group plugin's bespoke plots.

## Backend layout (headless, no Qt)

```
aggregate_quantification/dynamics/
  __init__.py        build_track_dynamics, read_track_dynamics, table column tuples
  trajectories.py    label stack → {track_id: (frames, xy_um)}  (regionprops loop)
  kinematics.py      instantaneous + per-track summary + DAC persistence
  msd.py             ensemble TA-MSD curve + power-law (D, α) fit
  collective.py      per-frame order param, C(r), 1/e correlation length, NN dist
  store.py           write/read the .h5 (+ provenance), object_table extractor
quantifiers/
  cell_dynamics.py     CellDynamicsQuantifier   (cell_labels_path, px, dt)
  nucleus_dynamics.py  NucleusDynamicsQuantifier (nucleus_labels_path, px, dt)
frame_interval.py    resolve_time_interval_s(...)  — mirrors pixel_size.py
```

`PositionInputs` gains `time_interval_s: float | None = None`;
`position_inputs_from_record` resolves it (config → TIFF `finterval` → manual
override) alongside `pixel_size_um`.

## UI (napari) — Track Dynamics group plugin

`napari/aggregate_quantification/plugins/track_dynamics.py`, same Compute+Plot
shape as Cell Shape:

- **Compute** — "Build track dynamics for N in-scope positions" + substrate
  radio (cell / nucleus) + *Recompute*; delegates to the studio build callback.
- **Plot** —
  - *Per-cell / per-track*: value (speed, directionality_ratio,
    persistence_time, …) · group-by (condition/date/position/class) · hist/box/
    violin/bar/line, via the generic `plotting` layer over `object_table`
    (instantaneous) and a per-track variant (summary table pooled directly).
  - *Curves*: MSD(τ) log-log (with the fitted `D, α` annotated) and ensemble
    DAC(τ); `C(r)` with the `1/e` `ξ` marked — bespoke, read from the extra `.h5`
    groups, rendered on the shared matplotlib canvas.
  - CSV export of any pooled/aggregated table.

## Out of scope (v1)

- PRW/Fürth MSD fit; per-track D/α; mean turning angle; arrest coefficient.
- Vorticity/divergence fields (owned by `notes/divergence_maps_spec.md`).
- 3-D (z) motion — trajectories are 2-D centroids as everywhere else here.
- VACF and mean-squared-angular-displacement (candidate follow-ons).
```
