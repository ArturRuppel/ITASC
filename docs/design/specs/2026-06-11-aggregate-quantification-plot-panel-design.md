# Aggregate Quantification — Detached Plot Panel

**Date:** 2026-06-11
**Status:** Design approved, pending implementation plan

## Problem

The Aggregate Quantification studio crunches pooled per-object tables and renders
matplotlib figures, but the figure is embedded in a thin napari side dock
(`CellShapePlugin`'s "Plot" section). napari's layout gives the image canvas ~70%
of the screen and leaves the dock a narrow vertical strip — the wrong place to
look at, tune, and export a statistical figure. We want a real plotting surface
with interactive styling (colors, labels, fonts, dimensions, style) that can claim
screen space when needed and give it back when not.

## Constraints that shaped the design

- **On one screen, a big plot and a big image canvas cannot coexist.** Every
  option just trades that space differently. But while styling a stats figure the
  user is *not* looking at the image data, so a surface that can *temporarily*
  claim space and yield it back is the right shape.
- **No free-floating OS popups** (napari designs against them), and **don't fight
  napari internals** (no overlaying/replacing the central canvas).
- The headless plotting backend (`build_figure`) is already Qt/napari-free and
  reused by scripts/notebooks/the standalone wheel — keep it that way.

## Decisions

1. **Host as a floatable napari dock widget**, not a popup and not the central
   canvas. A napari dock starts docked (View menu, re-dockable, napari-managed),
   and the user **floats it or drag-resizes it** to get real estate on demand.
   This is the idiomatic napari answer (what `napari-matplotlib` does) and the
   only one that honors "no popup by default + real estate on demand + work with
   napari." The single napari touch-point is one `add_dock_widget` call.

2. **The panel owns every control** — analytical *and* styling. The plugin's
   "Plot" section collapses to a single **"Plot…"** button. The panel is the whole
   plotting experience; the dock is just a launcher. (No split of controls across
   two places → one source of truth.)

3. **Snapshot at launch + multiple independent dock windows.** Clicking "Plot…"
   captures the current catalog scope, pools it, and opens a dock bound to that
   snapshot. Changing the selection afterward does not touch open docks; click
   "Plot…" again for another. Two docks can be floated side-by-side to compare
   (condition A vs B, hist vs box). The panel never listens back to the studio.

4. **Pool once; never re-pool.** At launch the plugin **always left-joins the
   contacts `class_label`** when a position has it (absent → `unclassified`). This
   removes the special "split by subpopulation" toggle entirely — `class_label`
   becomes just another group-by column. Combined with snapshot semantics, the
   panel holds one tidy DataFrame for its whole life and every control is a pure,
   cheap **re-render**.

5. **Engine = matplotlib; seaborn as the statistical layer; no plotly.**
   - matplotlib is the engine: the backend already returns a `Figure`; style
     sheets, palettes, font scaling, figure dimensions, all five plot types, and
     vector PNG/SVG export are its home turf; the headless Agg path and unit tests
     survive only with it.
   - seaborn sits *on top of* matplotlib (returns matplotlib `Figure`s, so
     embedding/export/`StyleSpec`/pick events are unchanged) and is adopted for the
     **distribution family** (hist/box/violin + a natural new strip/swarm), where
     its tidy-DataFrame `hue=` maps exactly onto our group-by / `class_label`
     model and collapses the hand-rolled grouping code.
   - **bar/line stay custom matplotlib** driven by the existing `aggregate()`, to
     preserve the deliberate position-level **pseudoreplication guard** (aggregate
     tissues, not cells). seaborn must not be allowed to silently re-aggregate raw
     cells on these paths.
   - plotly is rejected: web/JS engine, embedding needs QtWebEngine (Chromium) +
     a `QWebChannel` bridge, discards the headless backend, weaker static export.
     Its web interactivity is wasted in an embedded desktop dock.

6. **Design the napari-highlight feature in, build it later.** The panel stays
   napari-free but reserves the seam (see "Future" below).

## Architecture

Three pieces, layered by dependency:

### `plotting.py` (existing, headless — gains `StyleSpec`)

A frozen `StyleSpec` dataclass beside `PlotSpec`, threaded through `build_figure`,
defaulting to today's look so existing output is unchanged until a field is
touched:

- **Colors** — a named qualitative palette applied across groups.
- **Text** — optional `title` / `xlabel` / `ylabel` overrides (blank = current
  auto labels).
- **Style & dimensions** — style sheet (`default`, `seaborn-*`, `ggplot`, …),
  figure width/height (inches), grid on/off, legend on/off + location.
- **Font** — a base font size driving title/labels/ticks.

`build_figure(df, plot_spec, style_spec=StyleSpec())` stays headless and
unit-testable. Distribution plots route through seaborn; bar/line stay custom.

### `plot_panel.py` (new — `cellflow/napari/aggregate_quantification/`)

A generic `PlotPanel(QWidget)`. **Imports qtpy + matplotlib + seaborn + the
headless backend only — no napari.** Constructed with
`(dataframe, value_columns, group_columns)` — the only quantity-specific
knowledge, supplied by the caller. Contains:

- **Analytical controls** → build a `PlotSpec`: value, level, plot type, stat,
  error, group-by checkboxes (one per `group_column`).
- **Styling controls** → build a `StyleSpec`: the four groups above.
- **Canvas + matplotlib `NavigationToolbar`** (native pan/zoom/reset/save).
- **Export**: pooled CSV, aggregated CSV, figure (PNG/SVG).
- Any control change → pure re-render from the held DataFrame. **Never re-pools.**
- A dormant `selection_changed` signal (see Future).

Lives under the napari package for proximity to its consumer but does not import
napari, so it stays unit-testable without a viewer and reusable by any future
quantity that hands it a different DataFrame + column roles.

### `cell_shape.py` (existing plugin — slimmed)

- **Compute** section: unchanged.
- **Plot** section → a single **"Plot…"** button (enabled when ≥1 in-scope
  position is built).
- On click: pool the in-scope positions off-thread (as today) **always joining
  the contacts `class_label` when present**, then construct a `PlotPanel` from the
  snapshot and add it via
  `viewer.window.add_dock_widget(panel, area="right", name="Cell shape plot N")`.
  Each click → a new, independently floatable dock.
- Removed (their jobs move into the panel): the embedded `FigureCanvasQTAgg` and
  its `_render`, the `split` checkbox and its pooling-invalidation, and the
  in-dock export buttons.

## Data flow

```
catalog scope ──"Plot…"──▶ pool once (off-thread, always-join class_label)
                                   │  tidy DataFrame (snapshot)
                                   ▼
              PlotPanel(df, value_columns, group_columns)   ── add_dock_widget ──▶ floatable napari dock
                 │  controls → PlotSpec + StyleSpec
                 ▼
              build_figure(df, plot_spec, style_spec)  ──▶  matplotlib Figure on the canvas
                 │
                 └── export: write_csv(pooled) / write_csv(aggregate(...)) / fig.savefig(PNG|SVG)
```

## Future (designed-in, not built now): highlight picked cells in napari

- `PlotPanel` exposes an optional Qt signal `selection_changed(rows)` where `rows`
  is plain data — the identity columns (`position_id, frame, cell_id`) of the
  picked points, read from the DataFrame the panel already holds. The panel knows
  nothing about napari; it just announces which rows were picked. Today the signal
  exists but nothing connects to it.
- The **Cell Shape plugin** is the future consumer: it connects the signal and
  paints an outline layer for the matching cells, reusing the existing pattern in
  `nls_classification.py` (`np.isin(labels, ids)` → `add_labels(..., contour=2)`).
  The cross-boundary payload is plain row identity, so napari coupling never leaks
  into the panel.
- **Cost to enable later:** `build_figure` sets `picker=True` on the pickable
  artists (scatter/strip/hist) and stashes an artist-index → DataFrame-row map; the
  panel wires `pick_event` → look up rows → emit the signal. Not written now;
  pooling already keeps row identity (`frame, cell_id, position_id` are columns),
  so this is a purely additive change — no panel refactor.

## Testing

- **`plotting.py`**: `StyleSpec` defaults reproduce current figures; each style
  field measurably changes the figure (title text, fig size, font size, palette);
  distribution plots render via seaborn; bar/line still honor the position-level
  aggregation.
- **`plot_panel.py`**: with a small synthetic DataFrame + a `QApplication` (no
  viewer), constructing the panel renders; flipping each analytical/styling control
  re-renders without error; export writes CSV and image files; `selection_changed`
  exists and is emit-able with a row payload (even though unconnected).
- **`cell_shape.py`**: "Plot…" pools and calls `add_dock_widget` once per click
  (viewer stubbed/mocked); a second click adds a second dock.

## Explicitly out of scope (YAGNI)

- The napari highlight feature itself (designed in above, not built).
- Saving/restoring style presets.
- Per-position CSV export (was a single-position-scope convenience in the dock); 
  dropped unless requested.
- Live-linked or viewer-coupled plotting (rejected in favor of snapshot + the
  napari-free panel).

## Open items to confirm during implementation

- Add `seaborn` as a dependency (pulls `scipy` if not already present); confirm
  it's acceptable and wire it into the relevant `pyproject.toml` extras.
