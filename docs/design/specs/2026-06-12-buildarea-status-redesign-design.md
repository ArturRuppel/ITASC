# Aggregate Quantification BuildArea status redesign

**Date:** 2026-06-12
**Status:** Design — approved for planning

## Problem

The BuildArea (in `src/cellflow/napari/studio_plugins.py`) shows one status dot per
metric: grey (not applicable), **red** (buildable but not built for every in-scope
position), green (built for all). The red dot reads as "blocked / can't run" when it
actually means "ready to build, just not built everywhere yet" — the opposite of the
intended message. It also collapses two separate facts (which inputs a metric needs,
and how much is built) into a single colour, so a red dot never explains *why*.

A prior fix corrected an over-counting bug (the catalogue stamps a
`contact_analysis_path` on every position whether or not contacts is built, so the
contacts-derived metrics counted positions that could never build them). That fix
stands; this redesign replaces the colour-coded dot with an explicit, self-explaining
status display.

## Goal

Adopt the main app's project-status pattern: a deduplicated **input legend** at the
top, then each **output** (metric) row showing which inputs it consumes (reusing the
same abbreviations) plus its own built state. No red. The studio pools many in-scope
positions, so availability is expressed as **coverage** ("built for X of Y"), not a
bare yes/no.

## Decisions

- **Partial state → count badge.** Every input and output shows `X/Y`. Full coverage
  also gets a `✓` (accent/green); partial shows the plain count; none shows `0/N`.
- **Dependencies split into two kinds:**
  - **File inputs** (per-position, from a quantifier's `requires`): `cell_labels_path`,
    `nucleus_labels_path`, `contact_analysis_path`. Carry per-position coverage badges.
  - **Params** (global, set in the Parameters panel): `pixel_size_um`,
    `time_interval_s`, `fov_area_mm2`. Shown as **set / unset**, applied to all positions.
- **Drop per-position param auto-resolution.** Pixel size / frame interval are no
  longer read from each position's `config.json` or TIFF tags. The user sets them
  globally and they apply to every position. They move from `requires` into the
  existing global build-param mechanism (joining FOV area).
- **Contacts intermediate** gets an abbreviation (`CA`) used as a chip on the derived
  metrics, and keeps its own output row. Its chip mirrors the contacts row's built
  coverage. Derived metrics stay nested under the contacts group.
- **Encoding is text abbreviations**, not icons (theme-independent, screen-reader
  friendly).

## Abbreviation registry

| Abbr | Dependency        | Field / source             | Kind  |
|------|-------------------|----------------------------|-------|
| `C`  | cell labels       | `cell_labels_path`         | file  |
| `N`  | nucleus labels    | `nucleus_labels_path`      | file  |
| `CA` | contacts          | `contact_analysis_path` (produced) | file (intermediate) |
| `px` | pixel size        | `pixel_size_um`            | param |
| `Δt` | frame interval    | `time_interval_s`          | param |
| `A`  | FOV area          | `fov_area_mm2`             | param |

The legend lists only the dependencies referenced by the registered metrics, so it
adapts if metrics are added or removed.

## Layout

```
INPUTS
  Params                         Files
   px pixel size      set ✓       C cell labels    5/5 ✓
   Δt frame interval  set ✓       N nucleus labels  2/5
   A  FOV area        unset

OUTPUTS
  Cell
    Cell dynamics      C px Δt    5/5 ✓
    Cell shape         C px       5/5 ✓
    Cell density       C A        5/5 ✓        (A muted when FOV unset)
    Cell–cell contacts C          5/5 ✓  ⟶ CA
      ↳ Neighbor count        CA   3/5
      ↳ Contact energetics    CA   3/5
  Nucleus
    Nucleus shape      N px       2/5
```

## States & behavior (replacing the dots)

- **Output built badge** = `built / applicable`, where *applicable* = in-scope
  positions that have all of the metric's file inputs. `✓` when `built == applicable`;
  plain count when partial; `0/N` when none built.
- **Buildable** (checkbox enabled) = at least one applicable position **and** every
  required global param is set. Same gating as today, without the red.
- **Blocked dependency** is shown by *which chip is unsatisfied*, not by a red row: a
  chip whose dependency is missing for **all** positions (a file no position has, or an
  unset param) renders muted/struck. "Why can't I build this" reads off the row.
- The **INPUTS legend** is the single source of truth for coverage; output-row chips
  are just abbreviations, and a chip dims when its legend entry is fully unmet.

## Implementation shape

- **`studio_plugins.py`**
  - Replace `_INPUT_LABELS` with an ordered registry: dependency → `(abbrev, label, kind)`.
  - Add a small `InputLegend` widget (deduped param + file chips with coverage badges)
    rendered above the metric rows.
  - `_MetricRow` drops `dot`; gains a chip-strip and a built-coverage badge label.
  - `_refresh` computes per-input coverage once (params: set/unset; files: count of
    in-scope positions having the file) and per-output `built/applicable`, then updates
    the legend, the rows, and checkbox enablement.
  - Remove the `_resolve_pixel_size` / `_resolve_time_interval` wrappers;
    `position_inputs_from_record` takes px/Δt only from the stamped global params. Leave
    the standalone `resolve_pixel_size_um` / `resolve_time_interval_s` modules in place
    unless they become unused (they may prefill the Parameters panel).
- **Quantifiers**
  - Move `pixel_size_um` / `time_interval_s` out of `requires` into
    `required_build_params` (joining `fov_area_mm2`); set `wants_build_params = True`
    where needed. Affected: `cell_dynamics`, `nucleus_dynamics`, `cell_shape`,
    `nucleus_shape`, `shape_relational` (px), and the dynamics quantifiers (Δt).
  - Separate the two roles cleanly: `required_build_params` drives **gating** (the
    legend's set/unset, `missing_build_params`, checkbox enablement). The param
    **values** still reach `build()` through the stamped `PositionInputs` —
    `shared.stamp(records)` writes the global px/Δt onto each record and
    `position_inputs_from_record` carries them into `PositionInputs.pixel_size_um` /
    `.time_interval_s`. So no `build()` signature changes; only the source of the value
    (global stamp, never config/TIFF) and the gating mechanism change.
- **Tests**
  - `tests/napari/test_studio_plugins.py`: replace dot-colour assertions
    (`_DOT_NONE/_DOT_MISSING/_DOT_ALL`) with legend/badge assertions; add cases for
    partial coverage and unset params.
  - Adjust quantifier tests that pass px/Δt via `requires` or `PositionInputs`.

## Out of scope

- Changing the plot area / pooling.
- Reworking the Parameters panel beyond what global-only params requires.
- The already-merged `contact_analysis_path`-must-exist fix (kept as-is).
