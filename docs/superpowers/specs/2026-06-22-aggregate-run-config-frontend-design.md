# Aggregate Quantification — UI-authored config + single Run front-end (design)

> Deferred follow-up of the 4-spec aggregate consolidation (see
> `2026-06-22-aggregate-napari-frontend-refocus-design.md`). The studio's
> piecemeal **Compute** + **Aggregate** sections are replaced by one **Run**
> section that authors a `config.toml` from the UI and drives the whole
> `pipeline.run()` (classify → build → aggregate → export → figures).

**Date:** 2026-06-22

## Goal

Make the napari studio a true *front-end* to the headless `run(config_path)`
entry point. The analyst fills in the catalogue (already there), picks run-level
choices in one **Run** section, and clicks **Run** — the studio writes
`catalog.csv` + `config.toml` to the catalogue root and executes the same
pipeline a CLI/notebook run would. A separate **Save config…** button writes the
two files *without* running, so the exact run can be reproduced headlessly later.

This removes the two ad-hoc, in-memory build/aggregate paths (`BuildArea`'s
per-metric Run, `AggregateArea`'s Run Aggregate) in favour of the single
config-driven `run()` — one code path, reproducible by construction.

## Decisions (locked with the user)

1. **Full schema in the UI.** The Run section exposes *quantities* (which metrics
   to build), the `[nls]` step (enable + method + image + threshold), and
   `[plots]` (render + formats). The existing **Parameters** section
   (`pixel_size_um` / `time_interval_s` / `fov_area_mm2` / `shuffles`) supplies
   `[params]`. The authored `config.toml` is complete and immediately runnable —
   nothing needs hand-editing.
2. **Two buttons: Save config… + Run.** "Save config…" writes `catalog.csv` +
   `config.toml` only (the headless/CLI hand-off). "Run" writes both, then
   executes `run()` in-app on a worker thread. Both share one authoring path.
3. **Whole-catalogue scope.** Every catalogue record is written to `catalog.csv`
   every time; the `config.toml`'s `catalog` *is* the dataset. Table selection
   stays an inspection/curation gesture (as in the curation tool), never a run
   filter. This matches `run()`'s "the catalog is the dataset" model.

## Architecture

Three layers, each independently testable; Qt stays a thin glue layer.

### 1. Config writer (headless) — `aggregate_quantification/config.py`

The inverse of `load_config`, kept beside it for cohesion. A small, dependency-free
TOML serializer for our **closed** value set (str, bool, int, float, list[str],
tables) — we control every value, so a general TOML writer (and a new dependency
into the frozen lock) is unnecessary.

```python
def write_config(
    path: Path | str,
    *,
    catalog: str = "catalog.csv",
    export_dir: str = _DEFAULT_EXPORT_DIR,      # "export"
    curation: str = _DEFAULT_CURATION,          # "curation.csv"
    quantities: Sequence[str] = (),
    params: Mapping[str, object] | None = None,
    nls: NlsConfig | None = None,
    render_plots: bool = False,
    plot_formats: Sequence[str] = ("png", "svg"),
) -> Path: ...
```

Behaviour:
- Writes **relative** paths verbatim (so the project folder stays relocatable, as
  `load_config` documents).
- Emits `quantities` as an array only when non-empty (empty ⇒ omit ⇒ `load_config`
  reads `()` ⇒ "all"). The UI always passes the explicit checked list, so the file
  records exactly what will build.
- `[params]` emits only the keys present and non-`None` (drops unset
  `pixel_size_um` etc.; keeps `shuffles`).
- `[nls]` table emitted only when `nls is not None`, with
  `enabled`/`image`/`method`/`threshold`.
- `[plots]` table always emitted (`render` + `formats`).
- **Round-trip invariant:** `load_config(write_config(p, ...))` reproduces the
  inputs (paths resolved against `p.parent`). This is the primary test.

String escaping: escape `\` and `"` in emitted strings (defensive; our values are
paths/identifiers/enums that rarely need it).

### 2. Authoring + run orchestration (headless) — `aggregate_quantification/pipeline.py`

```python
def author_config(
    out_dir: Path | str,
    records: Sequence[dict],
    *,
    quantities: Sequence[str] = (),
    params: Mapping[str, object] | None = None,
    nls: NlsConfig | None = None,
    render_plots: bool = False,
    plot_formats: Sequence[str] = ("png", "svg"),
    catalog_name: str = "catalog.csv",
    config_name: str = "config.toml",
) -> Path:
    """Write catalog.csv + config.toml into out_dir; return the config path."""
```

`author_config` is the composition point: `save_catalog(out_dir/catalog_name,
records)` then `write_config(out_dir/config_name, catalog=catalog_name, ...)`.
Creates `out_dir` if missing. The "Save config…" action is exactly this; "Run"
is `run(author_config(...))`.

**`run()` gains progress.** Add an optional `progress_cb` forwarded to
`build_quantities` and `classify` so the UI can show stage progress without the
studio re-implementing the loop:

```python
def run(config_path: Path | str, *, progress_cb=None) -> list[Path]: ...
```

Default `None` preserves today's behaviour.

### 3. Run section (napari) — `napari/aggregate_quantification_run_area.py`

A `RunArea(QWidget)` mirroring `AggregateArea`/`BuildArea`: it owns the controls
and reads UI state into a small `RunChoices` value; the studio owns threading and
calls `author_config` / `run`.

Controls:
- **Quantities:** one checkbox per registered quantifier (id + display name),
  all checked by default. ≥1 must be checked or Save/Run is disabled with a hint.
- **NLS:** `enabled` checkbox, `method` combo (`auto`/`otsu`/`two_cluster`/`fixed`),
  `image` line-edit (default `0_input/NLS_zavg.tif`), `threshold` line-edit
  (used by `fixed`).
- **Plots:** `render figures` checkbox, `formats` line-edit (comma-separated,
  default `png,svg`).
- **Save config…** and **Run** buttons; a status label.

```python
@dataclass
class RunChoices:
    quantities: tuple[str, ...]
    nls: NlsConfig | None       # None when the NLS group is unchecked
    render_plots: bool
    plot_formats: tuple[str, ...]
```

`RunArea` exposes `choices() -> RunChoices`, `set_context(ctx)` (records → enable
state), `set_status(msg)`, and invokes `save_callback(choices)` /
`run_callback(choices)`. It is rebuilt by the studio when the quantifier registry
changes (same `_reload_*` pattern the Build area used), so a runtime-registered
metric appears.

### Studio wiring — `napari/aggregate_quantification_studio.py`

- Remove `_build_build_section`, `_build_aggregate_section`, and the build/aggregate
  worker plumbing (`_run_quantity_builds`, `_begin_build`, `_on_build_*`,
  `_run_aggregate`, `_on_aggregate_*`, `_reload_build_area`, the `_build_*` /
  `_aggregate_worker` state, `_set_build_status`). Drop the `BuildArea`,
  `AggregateArea`, `build_quantities`, and `aggregate as aggregate_tables` imports.
- Keep `_scoped_records` (it still feeds every analysis plugin via
  `_push_context_to`), `catalogue_root`, `available_quantifiers`,
  `_ProgressEmitter`.
- Add `_build_run_section` + `_reload_run_area`; wire:
  - `save_callback`: `author_config(catalogue_root(self._records), self._records,
    quantities=…, params=self._shared_params.build_params(), nls=…,
    render_plots=…, plot_formats=…)`, then status `Wrote config.toml + catalog.csv`.
  - `run_callback`: author as above, then `run(config_path, progress_cb=emit)` on a
    `thread_worker`; status shows progress then `Exported N .iris → export/`.
- Section title is **Run** (single collapsible), replacing the **Compute** and
  **Aggregate** collapsibles.

### Retirements

- Delete `napari/aggregate_quantification_aggregate_area.py` and
  `tests/napari/test_aggregate_area.py` (aggregate-only; the latter also carries
  the one known pre-existing failure, removed with it).
- Leave `BuildArea` in `studio_plugins.py` (a generic, separately-tested building
  block) — only the studio's *use* of it goes away.

## Testing

- **Config writer (headless):** round-trip `write_config` → `load_config`
  (paths, quantities, params drop-None, `[nls]` present/absent, `[plots]`); string
  escaping; relative paths preserved.
- **author_config (headless):** writes both files into `out_dir`; `load_config` on
  the result yields the records' catalog and the chosen knobs; creates missing
  `out_dir`.
- **run() progress (headless):** `progress_cb` receives build (and, when
  `[nls].enabled`, classify) callbacks; default `None` unchanged.
- **RunArea (offscreen Qt):** quantity checkboxes default-checked; `choices()`
  reflects toggles; NLS group disabled→`nls is None`, enabled→populated
  `NlsConfig`; Save/Run disabled with zero quantities; buttons invoke callbacks
  with the built `RunChoices`.
- **Studio (offscreen Qt):** sections are `Catalogue/Parameters/Tools/Run` (no
  `Compute`/`Aggregate`); a stubbed `run`/`author_config` is invoked on Run with
  the catalogue root + UI choices.

## Out of scope

- No new runtime dependency (hand-rolled writer).
- No change to the headless stages themselves beyond the `progress_cb` pass-through.
- No per-row run filtering (whole-catalogue, per decision 3).
