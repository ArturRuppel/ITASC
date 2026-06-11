# Implementation Plan — Reduce dynamics curves to per-track & per-tissue scalars

**Date:** 2026-06-11
**Branch state:** clean (`main`)

## Goal

The Track Dynamics plugin currently produces three *ensemble curves* (MSD, directional
autocorrelation, velocity correlation `C(r)`) that are only viewable as overlaid lines and
annotated with per-position scalars in text. Make the curve information **boxplot-able** by:

1. **Per-track MSD fit** — fit each track's own MSD over a shared fixed lag window, adding
   `msd_D_um2_per_s`, `msd_alpha`, `msd_r2` columns to the per-track summary table so they
   appear in the existing **Per-track** plot view, groupable by `class_label` / condition.
2. **Per-tissue scalar table** — a new **Per-tissue** plot view: one row per built position
   carrying the ensemble `D`, `α`, persistence time, and a single correlation-length scalar
   `ξ` (the 1/e length of the pooled `C(r)` curve), boxplottable across conditions.

Design decisions already settled with the user:
- Fixed lag window `L = 8` frames for the per-track MSD fit (exposed as a build param so it
  can be overridden). A fixed window makes every track's `D`/`α` comparable and excludes the
  noisy long-lag tail. Tracks that can't fill ≥2 distinct lags in the window → `NaN`.
- Correlation length is inherently a neighbourhood/tissue quantity → per-tissue only.
- The curve math itself is **unchanged**; these are reductions layered on top.

What already exists and needs **no** work: per-track `curvilinear_speed`, `net_speed`,
`directionality_ratio`, `persistence_time`, `path_length`, `net_displacement`, `duration`
are already in the per-track table and already boxplot by `class_label`. The DAC curve is
*already* reduced to per-track `persistence_time_s`.

---

## Step 1 — `msd.py`: per-track fixed-window MSD fit

**File:** `src/cellflow/aggregate_quantification/dynamics/msd.py`

Add module constants:
```python
#: Per-track MSD-fit value columns, merged into the per-track summary table.
MSD_TRACK_COLUMNS = ("msd_D_um2_per_s", "msd_alpha", "msd_r2")
#: Default fixed lag window (frames) for the per-track MSD fit.
DEFAULT_MSD_TRACK_WINDOW = 8
```

Add `per_track_msd_fit(trajectories, *, time_interval_s, window_frames=DEFAULT_MSD_TRACK_WINDOW)
-> dict[int, MsdFit]`:
- For each track, build `sq_by_lag` over lags `1…window_frames` using the existing
  `_accumulate_track_sq` helper (its own present `(k, k+n)` pairs only).
- If fewer than 2 distinct lags have samples → `MsdFit(nan, nan, nan)`.
- Otherwise compute `msd[n] = mean(sq_by_lag[n])` and call the existing `fit_msd_power_law`
  on `lags * dt` vs `msd`. Key the result by `traj.track_id`.

Reuses `_accumulate_track_sq` and `fit_msd_power_law` — no new fit math.

---

## Step 2 — `collective.py`: pooled correlation-length helper

**File:** `src/cellflow/aggregate_quantification/dynamics/collective.py`

Add `pooled_corr_length(corr_curve: dict[str, np.ndarray]) -> float`: apply the existing
`_one_over_e_length` to the pooled curve's `separation_um` / `corr` arrays; return `NaN` when
the curve is empty or never decays to `1/e`. (Single per-tissue ξ; the per-frame
`corr_length_um` column in the collective table is unchanged.)

---

## Step 3 — `store.py`: persist the new scalars

**File:** `src/cellflow/aggregate_quantification/dynamics/store.py`

1. **`DEFAULT_PARAMS`**: add `"msd_track_window_frames": 8`.
2. **`build_track_dynamics`**: after `tracks = track_summary_table(...)`, call
   `per_track_msd_fit(trajectories, time_interval_s=..., window_frames=int(p["msd_track_window_frames"]))`
   and merge three columns into the `tracks` dict aligned to `tracks["cell_id"]` (NaN for
   any id not in the fit map). Compute `corr_length = pooled_corr_length(corr_curve)` and pass
   it into `_write_h5`.
3. **`_write_h5`**: write `h5["corr_curve/table"].attrs["corr_length_um"] = float(corr_length)`
   (symmetric with the existing `msd` fit attrs and `dac` persistence attr). The per-track MSD
   columns ride along automatically since they're now in the `tracks` table dict.
4. **`TrackDynamics`**: add field `corr_length_um: float`.
5. **`read_track_dynamics`**: read `corr_length_um` from the `corr_curve/table` attrs, falling
   back to `float("nan")` when absent → **backward compatible** with `.h5` files built before
   this change (the per-track MSD columns are simply missing from the older `tracks` table;
   downstream `.get(...)` access tolerates that).

Helper `_merge_track_msd(tracks, track_msd)` lives in `store.py` (it bridges kinematics +
msd, neither of which should import the other).

---

## Step 4 — `__init__.py`: exports

**File:** `src/cellflow/aggregate_quantification/dynamics/__init__.py`

Export `MSD_TRACK_COLUMNS` and `per_track_msd_fit` from `.msd`, and `pooled_corr_length` from
`.collective`; add them to `__all__`.

---

## Step 5 — `track_dynamics.py` plugin: surface the new numbers

**File:** `src/cellflow/napari/aggregate_quantification/plugins/track_dynamics.py`

1. **Per-track values**: append `"msd_D_um2_per_s"`, `"msd_alpha"` to `_TRACK_VALUES` (they
   now show up as selectable y-axis columns in the Per-track view; `msd_r2` stays in the table
   for reference but is not offered as a plotting value).
2. **New tissue column sets**:
   ```python
   _TISSUE_VALUES = ("msd_D_um2_per_s", "msd_alpha", "persistence_time_s",
                     "corr_length_um", "order_param")
   _TISSUE_GROUPS = _METADATA_GROUPS  # condition, date, position_id
   ```
3. **View dropdown** (`_build_plot`): add `self._view_combo.addItem("Per-tissue (ensemble D, α, ξ…)", "tissue")`
   between the Per-track and Curves items.
4. **`_on_plot` worker**: add a `view == "tissue"` branch returning
   `("tissue", _tissue_records(quantifier, records))`.
5. **`_on_pool_done`**: `tissue` routes to `_open_distribution` (the `curves` branch stays the
   only special case).
6. **`_open_distribution`**: replace the binary `frame`/else value-group choice with a small
   `{"frame": (...), "track": (...), "tissue": (...)}` lookup.
7. **`_tissue_records(quantifier, records) -> pd.DataFrame`**: one row per *built* position via
   `read_track_dynamics`, columns = `_metadata(record)` plus `msd_D_um2_per_s`,
   `msd_alpha` (ensemble attrs), `persistence_time_s` (`dac_persistence_time_s`),
   `corr_length_um` (`dyn.corr_length_um`), and `order_param` = NaN-safe median of
   `dyn.collective["order_param"]`. Returns an empty DataFrame when nothing is built (handled
   by the existing `pooled.empty` guard).

---

## Step 6 — Tests

**`tests/aggregate_quantification/test_dynamics_core.py`**
- `per_track_msd_fit`: a long straight track → `α ≈ 2`; verify a track shorter than the window
  with <2 fillable lags → NaN `D`/`α`; verify only lags ≤ `window_frames` contribute (a track
  longer than the window gives the same fit as one truncated to the window).
- `pooled_corr_length`: on a synthetic decaying pooled curve returns the expected 1/e crossing;
  empty curve → NaN.

**`tests/aggregate_quantification/test_dynamics_quantifier.py`** (or store-level test)
- After `build_track_dynamics`, the `tracks` table has `msd_D_um2_per_s` / `msd_alpha` /
  `msd_r2`, and `read_track_dynamics(...).corr_length_um` is finite/NaN as expected.
- Backward-compat: a hand-written `.h5` lacking the `corr_length_um` attr loads with
  `corr_length_um == NaN`.

**`tests/napari/test_track_dynamics_plugin.py`**
- `_pool_records(..., "track")` now includes `msd_D_um2_per_s` / `msd_alpha`.
- New `_tissue_records(...)` returns one row per built position with the five tissue columns
  plus `condition` / `date` / `position_id`.
- Extend `test_plot_opens_dock_for_each_view` to drive `_on_pool_done(("tissue", _tissue_records(...)))`
  and assert a dock opens.

---

## Out of scope / explicitly not doing

- No change to the ensemble curve math or the `DynamicsCurvesPanel`.
- No per-track velocity-correlation length (ill-defined for a single track).
- No rebuild migration tooling — old `.h5` files load (with NaN ξ / missing per-track MSD) and
  populate the new numbers only on an explicit Recompute.

## Verification

`uv run --frozen pytest tests/aggregate_quantification/test_dynamics_core.py
tests/aggregate_quantification/test_dynamics_quantifier.py
tests/napari/test_track_dynamics_plugin.py` green; manual smoke in napari: Build → Per-track
view shows `msd_D_um2_per_s` selectable; Per-tissue view opens with one point per position.
