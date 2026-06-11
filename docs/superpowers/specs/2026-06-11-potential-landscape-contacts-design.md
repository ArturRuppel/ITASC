# Potential Landscape (Contacts Energetics) — Design Spec

Date: 2026-06-11

> **Status: design.** Aligns the *exact* calculations before building. Follows the
> quantifier seam (`aggregate_quantification/quantifier.py`), the headless
> plotting backend (`aggregate_quantification/plotting.py`), and the
> pool-a-snapshot / launch-a-panel plugin pattern established by Track Dynamics
> (`notes/2026-06-11-track-dynamics-quantifier-design.md`).

The next Aggregate Quantification analysis: the **potential landscape** of the
cell–cell contact network — a Boltzmann inversion of a reaction coordinate into
an effective potential `U(x) = −ln P(x)` (units of kT), and the **effective
barrier `ΔE_eff`** read off it. The headline application is the **signed central
junction length** of T1 transitions: pooled and inverted it produces a
double-well potential whose barrier is the energy a junction must climb to reach
the four-fold vertex (the T1 transition state).

Modeled on `morphogenesis-on-chip_analysis/fig5.py`
(`plot_signed_lengths_neg_log_p_histogram` → `−ln P(L)` curve) and `fig6.py`
(barrier `ΔE_eff` correlated with intercalation rate), in the reference repo
`~/Projects/inter-s-cale/morphogenesis-on-chip_analysis/` (context only — not a
CellFlow dependency). The reference signs the central junction length upstream in
curated "quad" JSONs; CellFlow derives the same sign directly from the
`t1_events` table (`losing_pair` ↔ `gaining_pair`).

## Decisions (locked with the user)

1. **Layered, both layers this increment.**
   - **Layer 1 — reusable `potential` plot mode** in `plotting.py`: a new plot
     type that Boltzmann-inverts *any* pooled scalar. Reusable across every
     quantifier (cell area, speed, junction length…).
   - **Layer 2 — contacts signed reaction coordinate**: a headless
     `contacts/energetics.py` deriving the signed central junction length from a
     read `PositionContactAnalysis`, plus a thin "Potential landscape" plugin
     that pools it and opens the generic `PlotPanel` on the `potential` view.
2. **Naming.** UI section / plot view: **Potential landscape**. Plot mode key:
   **`potential`**. Y axis: **`U(x) = −ln P(x)` [kT]**. Scalar readout:
   **effective barrier `ΔE_eff` [kT]**.
3. **Barrier estimator: `U(0) − U_min`.** `ΔE_eff` = `U` **linearly
   interpolated at `x = 0`** (the transition state, junction length → 0) minus
   the curve's minimum. Interpolation between the occupied bins straddling zero
   (the v1 `_compute_energy_barrier` approach) keeps a sparse zero-bin from
   biasing it. No well fitting; NaN when 0 is not bracketed by the data.
4. **Frame window: all frames the edge exists.** Every frame where an event's
   losing/gaining edge is present contributes a sample — matching the reference,
   which histograms the whole curated-quad movie. No ± window parameter.
5. **Adaptive (sinh) binning, tighter near 0.** `bin_mode="adaptive"` lays the
   `potential` histogram on sinh-spaced edges (`adaptive_bin_edges`), narrowest at
   `x = 0` and widening toward the extremes — the reference / v1 sinh trick — to
   resolve the barrier at the transition state. A UI checkbox ("Tighter bins near
   0") drives it, defaulted **on** for the contact-energetics launch. `"uniform"`
   stays the default everywhere else.
6. **Raw-sample pooling for `potential`.** Unlike `hist`/`box`/…, the `potential`
   mode does **not** run `reduce_to_units`; it histograms raw frame-level
   samples. The landscape is the *shape of the fluctuation distribution*, not a
   per-track comparison, so unit reduction would be wrong (and, for the
   cell_id-less energetics table, degenerate). Documented in the function and the
   panel. This is a deliberate pseudoreplication tradeoff that mirrors the
   reference.
7. **Physical units.** `signed_length` is in µm when `pixel_size_um` resolves
   (via `resolve_pixel_size_um`, as Track Dynamics does), else pixels with a
   labeled fallback.
8. **Contact type = NLS transition pair, per event.** "Contact type" is the
   just-added NLS-subpopulation contact label (`label_contacts` /
   `contact_label_for`), **not** the build's `edge_label` (which is blank for
   cell-cell edges). A T1's losing and gaining junctions are different cell pairs
   with possibly different labels, so each **event** gets one
   `"<losing>→<gaining>"` transition label stamped on both lobes — chosen with the
   user over per-side or losing-only — so a per-type curve still spans `L=0` and
   `ΔE_eff` stays defined. The plugin sources the `cell_id → label` map from the
   position's `nls_classification.csv`.

## Layer 1 — the `potential` plot mode (`plotting.py`)

### New helpers (pure, unit-tested)

```python
def potential_landscape(
    values: np.ndarray, *, bins: int, value_range: tuple[float, float] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boltzmann-invert a 1-D sample into an effective potential.

    Returns (centers, U, counts) over occupied bins only. P = counts / N;
    U = -ln P (natural log → units of kT). Empty bins are dropped (U → ∞).
    """

def effective_barrier(centers: np.ndarray, U: np.ndarray) -> float:
    """ΔE_eff = U linearly interpolated at x = 0  minus  min(U).

    NaN when 0 is not bracketed by the occupied range or there are
    fewer than 2 occupied bins.
    """
```

- `N = len(values)` after dropping NaN. `np.histogram(values, bins, range)`.
- `centers = (edges[:-1] + edges[1:]) / 2`; keep `counts > 0`.
- `U = -np.log(counts / N)`.
- `effective_barrier`: require 0 bracketed by the occupied range, then `U(0) =
  np.interp(0, sorted centers, U)`; `ΔE = U(0) −
  U.min()`.

### `PlotSpec` / `build_figure` integration

- Add `"potential"` to a new `_CURVE_PLOTS = ("potential",)`; extend `_PLOTS`.
  It is **not** in `DISTRIBUTION_PLOTS` (no seaborn, no unit reduction).
- `build_figure`: route `spec.plot == "potential"` to `_plot_potential`, which:
  - For each group (via `_group_series` over `spec.group_by`), takes the **raw**
    `pd.to_numeric(chunk[spec.value])` (dropna) — *not* `reduce_to_units`.
  - Calls `potential_landscape(values, bins=spec.bins, value_range=…)` with a
    **shared** range across groups (min/max over the whole pooled `spec.value`)
    so curves are comparable.
  - Plots `U` vs `centers` as line+markers per group; draws `axvline(0)` when 0
    is in range.
  - Annotates `ΔE_eff` per group in the legend label
    (`f"{label} (ΔE_eff={be:.2f} kT)"`); NaN renders as `ΔE_eff=n/a`.
  - `ax.set_xlabel(spec.value)`, `ax.set_ylabel("U = −ln P  [kT]")`.
- `needs_value` guard already covers a missing `spec.value` column.
- `pickable_points`: returns `[]` for `potential` (no per-unit points).

### CSV

- `write_csv` is unchanged. The panel's CSV export for the potential view writes
  the **curve** (`group, center, U, counts, delta_e_eff`) via a small
  `potential_table(df, spec)` helper in `plotting.py`, not the raw pooled frame —
  so the exported numbers match the plotted curve.

## Layer 2 — the signed reaction coordinate (`contacts/energetics.py`)

Headless, Qt-free, operates on an already-read
`PositionContactAnalysis` (so it never opens HDF5 itself).

```python
def signed_central_junction_lengths(
    analysis: PositionContactAnalysis, *, pixel_size_um: float | None = None
) -> dict[str, np.ndarray]:
    """Signed central-junction length per T1 event per frame.

    Columns: t1_event_id, frame, signed_length, role ("losing"|"gaining"),
    contact_type. signed_length is +length for the gaining edge, −length for the
    losing edge, in µm when pixel_size_um is given, else pixels. contact_type is
    the event's NLS transition pair "<losing>→<gaining>" (e.g. "A-A→A-B") from the
    optional `labels` map (cell_id → NLS label), "" when no labels are given.
    """
```

Algorithm:

- **Join fragments first.** The build splits one cell-cell boundary into several
  `edges` rows (one per disconnected segment from `_coordinate_segments`), so the
  lengths of all rows sharing a `(frame, frozenset{cell_a, cell_b})` are
  **summed** into one total junction length. This is the headless analogue of the
  archived v1 `find_shared_boundary` / `order_boundary_pixels` join
  (`napariTissueGraph/core/labels.py`) that produced one length per junction;
  without it a fragmented contact would enter the landscape as several short
  samples.
- Build a lookup from the joined edges: `(frame, frozenset{cell_a, cell_b}) →
  total_length`.
- For each row of `t1_events` (`t1_event_id`, `losing_cell_a/b`,
  `gaining_cell_a/b`):
  - Compute **one** `contact_type` for the event: the transition pair
    `f"{losing}→{gaining}"`, where each side is `contact_label_for(labels, …)` —
    the sorted NLS-subpopulation pair label of that junction's two cells (shared
    with `label_contacts`). `""` when no `labels` map. **The same label is
    stamped on both lobes** (see decision 8) so a grouped curve stays two-sided.
  - For **every** frame where the losing pair `{lA,lB}` has an edge: emit a row
    `role="losing"`, `signed_length = −length` (× `pixel_size_um` if given).
  - For **every** frame where the gaining pair `{gA,gB}` has an edge: emit a row
    `role="gaining"`, `signed_length = +length`.
- Concatenate into column-major arrays. An event contributing no edge frames is
  skipped. Returns empty-but-typed arrays when there are no events.

Sign convention rationale: around a single T1 the losing edge exists before the
transition (junction shrinking toward 0, negative orientation) and the gaining
edge after (growing, positive orientation); the magnitude is the edge length and
crosses 0 at the four-fold vertex. This reproduces the reference's signed central
junction length without curated quads.

Assumptions / documented caveats:
- A pair that re-forms or pre-exists outside its single transition still signs by
  its event role (losing=−, gaining=+). With "all frames the edge exists" pooling
  this is the reference's behavior and is documented.
- `edges.length` units: the build computes it from coordinates (pixels);
  `pixel_size_um` converts to µm. When unresolved, the panel labels the axis
  "(px)".

## Wiring — `plugins/contact_energetics.py` ("Potential landscape")

A thin `AnalysisPlugin`, same shape as Track Dynamics' pool-and-launch path
(no bespoke panel — the curve fits the generic `PlotPanel`).

- `plugin_id = "contact_energetics"`, `display_name = "Potential landscape"`.
- **Plot section** only (no Compute: it reads the existing `contact_analysis.h5`;
  the contacts quantifier already builds it). A **"Plot…"** button.
- On click, a `thread_worker` pools every in-scope record:
  - `analysis = ctx.load(record)` (cached loader → `PositionContactAnalysis`).
  - `pixel = resolve_pixel_size_um(record_position_dir)` (best-effort; None ok).
  - `table = signed_central_junction_lengths(analysis, pixel_size_um=pixel,
    labels=read_nls_classification_csv(...))`.
  - Wrap as `PositionSource(metadata={condition,date,position_id}, table=table)`.
  - `pool_object_tables(sources)` → pooled DataFrame.
- Opens `PlotPanel` (via `PlotDockTabs`, exactly like Track Dynamics) with:
  - `value_columns = ("signed_length",)`,
  - `group_columns = ("condition", "date", "position_id")` (metadata only — the
    table has no `class_label`/`cell_id`), **plus `"contact_type"` when the
    pooled samples carry >1 transition type** (so a "group by contact type"
    checkbox appears only when the position is NLS-classified; blanks read as
    `"unlabelled"`). The plugin reads each position's `nls_classification.csv`
    (via `read_nls_classification_csv`, as shape / track-dynamics do) and passes
    the `cell_id → label` map into the energetics.
  - the plot type **defaulted to `potential`**.
- Matplotlib-Qt availability probe + disabled-button fallback, copied from
  Track Dynamics.
- `PlotPanel` change: add `"potential"` to its plot-type combo (it reads
  `_PLOTS`/`DISTRIBUTION_PLOTS` from the backend; surface the new key). The
  bins/range styling controls already exist.

The contacts `ContactsQuantifier.object_table` stays `cells` — unchanged. The
energetics table is plugin-pooled, never the default object table.

## Files

| File | Change |
|---|---|
| `aggregate_quantification/plotting.py` | `potential_landscape`, `effective_barrier`, `potential_table`, `adaptive_bin_edges` helpers; `PlotSpec.bin_mode`; `_CURVE_PLOTS`, `_plot_potential`; `build_figure`/`pickable_points` routing; exports. |
| `aggregate_quantification/contacts/energetics.py` | **new** — `signed_central_junction_lengths` (`labels` map → `contact_type` transition pair). |
| `aggregate_quantification/contacts/contact_labels.py` | factor out shared `contact_label_for(labels, a, b)` (used by `label_contacts` + energetics). |
| `aggregate_quantification/contacts/__init__.py` | export the new function. |
| `napari/aggregate_quantification/plugins/contact_energetics.py` | **new** — "Potential landscape" plugin. |
| `napari/aggregate_quantification/plot_panel.py` | surface `potential` in the plot-type combo; "Tighter bins near 0" checkbox → `bin_mode`; `default_plot` / `default_adaptive_bins` params. |
| `tests/aggregate_quantification/test_plotting.py` | `potential` math + raw-pooling + barrier + curve CSV. |
| `tests/aggregate_quantification/test_energetics.py` | **new** — sign convention, pixel-size, missing-edge. |
| `tests/napari/test_contact_energetics.py` | **new** — pool + headless launch. |

## Testing

- **`potential_landscape`**: known sample → expected `U`; empty bins dropped;
  `N`-normalization; single-bin / all-equal edge cases.
- **`adaptive_bin_edges`**: `bins+1` edges, endpoints preserved, strictly
  increasing, central bin narrowest / outer widest, symmetric for a symmetric
  range, `sharpness → 0` ≈ uniform, degenerate range safe. `bin_mode="adaptive"`
  resolves more occupied bins near 0 than uniform at the same count;
  `PlotSpec` rejects an unknown `bin_mode`.
- **`effective_barrier`**: synthetic bimodal (two Gaussians at ±L) → positive
  finite barrier; unimodal away from 0 → barrier from `U(0)`; no-zero-span → NaN.
- **`build_figure(plot="potential")`**: one curve per group; raw pooling
  verified (a table with a `cell_id`/`position_id` that *would* collapse under
  `reduce_to_units` still yields a full-resolution curve); legend carries
  `ΔE_eff`.
- **`signed_central_junction_lengths`**: hand-built `t1_events` + `edges` fixture
  — losing edge → negative across its frames, gaining → positive; **fragmented
  contact (several rows sharing a `(frame, pair)`) sums to one total-length
  sample, not one per fragment**; pixel-size scales magnitude (after the join);
  an event whose edges are absent is skipped; empty input → empty typed table;
  **`contact_type` is the NLS transition pair `"<losing>→<gaining>"` shared by
  both lobes of an event (sorted/orientation-independent; `unclassified` for a
  cell absent from `labels`; `""` when no `labels`)**.
- **`contact_energetics` contact-type grouping**: the plugin reads the NLS
  sidecar CSV per position and passes the map in; it adds `contact_type` to the
  panel's group columns only when the pooled samples carry >1 transition type,
  and maps blanks to `"unlabelled"`.
- **`contact_energetics` plugin**: pools a fake two-position context and launches
  the panel headlessly (existing matplotlib-Qt-guarded launch pattern); disabled
  state when the backend is unavailable.

## Out of scope (deferred)

- Barrier-vs-aspect-ratio scatter (`fig6` right panel): needs a per-condition AR
  covariate CellFlow does not carry yet.
- Well/double-well curve fitting; `ΔE_eff` stays the documented bin-based
  estimate.
- Other contact reaction coordinates (rosette order, neighbor number) — additive
  later via the same `potential` engine.
