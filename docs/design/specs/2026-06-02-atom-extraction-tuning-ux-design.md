# Atom Extraction tuning UX — design

Date: 2026-06-02
Component: `src/cellflow/napari/nucleus_atom_extraction_widget.py`
(`NucleusAtomExtractionWidget` + `NucleusAtomExtractionMixin`)

## Problem

The live preview creates four layers, but the seven knobs do not all act on
all four layers — they fall into two natural tuning stages:

- **Foreground params** (`fg_window`, `fg_cutoff`, `fg_strength`) drive only the
  `residual_foreground` image and the `territory` mask.
- **Contour params** (`contour_window`, `contour_floor`, `contour_strength`) plus
  `atom_min_area` drive only the `residual_contour` image and the atoms mask.

With all four layers and all seven knobs live at once, it is impossible to tune
one stage without the other's layers cluttering the view. We want to let the
user focus on one stage at a time while keeping both available.

A second problem surfaced while diagnosing "weird atoms": the atom split is
driven entirely by the **binary ridge mask** `residual_contour > contour_floor`,
not by the contour residual's height. That mask is computed inside
`extract_atoms_frame` and immediately discarded, so the user cannot see the one
array that actually decides where territory is cut. Symptom: pronounced ridges
are sometimes ignored before fainter ones — consistent with `residual()`
high-pass-hollowing the centers of wide/strong ridges below `contour_floor`,
which would be directly visible if the mask were a layer. We therefore **return
the ridge mask from `extract_atoms_frame` itself** — the exact array it carves
out of territory — and surface it as a fifth preview layer, so `contour_floor`
is tuned against the same bytes the watershed uses (no chance of a widget-side
re-derivation drifting from the real split).

## Design

### 1. Layer rename & styling

- Rename the atoms label layer `[Atoms] preview` → `[Atoms] atoms`
  (constant `_ATOM_PREVIEW_LAYER` value becomes `"[Atoms] atoms"`).
- Both label layers become masks at **opacity 0.7** (was 0.55), each ordered
  **directly on top of the residual image it is judged against**:
  - `[Atoms] territory` on top of `[Atoms] residual_foreground`
  - `[Atoms] atoms` on top of `[Atoms] residual_contour`
- A new **`[Atoms] ridge`** mask (constant `_ATOM_RIDGE_LAYER`,
  value `"[Atoms] ridge"`) shows the binarized `residual_contour > contour_floor`.
  It is a label/mask layer at **opacity 0.7**, ordered on top of
  `residual_contour` (just under `atoms`), so the user sees the exact wall that
  the atoms are cut along.
- Fixed stack order, bottom → top:
  `residual_foreground`, `territory`, `residual_contour`, `ridge`, `atoms`.
  Each mask sits immediately above the image it is judged against.

### 2. Params split into two groups

The single "Atom Extraction Params" section becomes two collapsible subgroups:

- **Foreground** — `fg_window`, `fg_cutoff`, `fg_strength`
  → drive `residual_foreground` + `territory`.
- **Contour** — `contour_window`, `contour_floor`, `contour_strength`,
  **and `atom_min_area`** → drive `residual_contour`, `ridge`, and `atoms`.

`atom_min_area` moves into the Contour group because it only post-processes
atoms. The header ⚙ button still expands/collapses the whole params area.

### 3. Per-group layer toggle (the core fix)

Each group's header gets a small **eye toggle** controlling the visibility of
*that group's two layers only*, independently:

- Foreground eye → show/hide `residual_foreground` + `territory`.
- Contour eye → show/hide `residual_contour` + `ridge` + `atoms` (all three
  Contour-group layers move together).
- Independent: 0, 1, or both groups may be shown at any time.

### 4. Ridge mask returned from the core (the source of truth)

The ridge mask is `residual_contour > contour_floor` — the binary wall that
`extract_atoms_frame` carves out of `territory` to seed the watershed. It is
already computed there (today's `ridge = residual_contour > contour_floor`,
`atoms.py:70`) and thrown away. Make it an **output** so it is authoritative:
the displayed mask is the same array the watershed used, by construction.

Core changes in `src/cellflow/tracking_ultrack/atoms.py`:

- **`extract_atoms_frame` returns `(atoms, ridge)`** — `atoms` unchanged;
  `ridge` is the `residual_contour > contour_floor` boolean it already builds,
  returned as `uint8` `(Y, X)`. (The `atom_min_area` re-flood does not change the
  ridge; the ridge is purely the threshold, so it is captured once up front.)
- **`extract_atoms_stack_with_maps` returns a 5-tuple**
  `(atoms, territory, residual_foreground, residual_contour, ridge)`, with
  `ridge` a `(T, Y, X)` `uint8` stack accumulated per frame.
- **`extract_atoms_stack` is unchanged externally** — it already unpacks with
  `atoms, *_ = extract_atoms_stack_with_maps(...)`, which still takes the first
  element.

Caller updates (positional unpacking changes):

- `extract_atoms_stack_with_maps` internal loop: `atoms_out[t], ridge_out[t] =
  extract_atoms_frame(...)`.
- Widget ◉ preview (`nucleus_atom_extraction_widget.py:286`): now
  `atoms, ridge = extract_atoms_frame(...)`.
- Widget full-stack (`:389`): unpack the 5-tuple including `ridge`.
- `scripts/experiment_atoms_to_db.py:274` and
  `scripts/experiment_residual_atoms.py:114` only want labels →
  `atoms, _ = extract_atoms_frame(...)`.

Layer:

- Render `ridge` as a single-value mask (label `1` where ridge, `0` elsewhere)
  at opacity 0.7 so it reads as a colored wall over `residual_contour`.
- `_run_atom_extraction` (full-stack) takes `ridge` straight from the 5-tuple;
  the ◉ preview takes it from the per-frame `extract_atoms_frame` return.

Diagnostic expectation: if the standing "pronounced ridges ignored first"
hypothesis holds, this layer will show wide/strong ridges as **hollow
double-rails** (broken in the middle) while faint thin ridges show as solid
connected lines — the broken mask is exactly what lets cores stay merged.

The `atoms.tif` format and fingerprint are **not** touched — `ridge` is an
in-memory return for display, never written to disk.

### 5. Behavior

- The ◉ live-preview worker still computes everything in one pass each refresh;
  `ridge` now comes back from `extract_atoms_frame` alongside `atoms` rather than
  being recomputed. The group eye toggles are pure napari visibility.
- On activating ◉: **Foreground pair visible, Contour group hidden** by default
  (Contour group = `residual_contour` + `ridge` + `atoms`). The eye toggles
  reflect this initial state.
- `_run_atom_extraction` (full-stack path) reuses the same layers, rename, and
  visibility rules.

### 6. Preserve user-set contrast on slider changes

Tuning a knob must **not** reset the contrast limits the user has dialed in on
any image layer (`residual_foreground`, `residual_contour`). Today both refresh
paths explicitly re-assign `layer.contrast_limits = (frame.min(), frame.max())`
on every update — `_fill_atom_image_slice` (the per-frame ◉ preview) and
`_set_atom_image_stack` (the full-stack path). So every slider drag snaps the
display back to the data's raw min/max and makes fine tuning impossible.

- **Remove the per-refresh `contrast_limits` assignment** from both
  `_fill_atom_image_slice` and `_set_atom_image_stack`. Updating `layer.data`
  alone does not move an existing layer's contrast limits, so the user's setting
  survives.
- Apply auto-contrast (`(min, max)`) **only when the layer is first created**, so
  a freshly added layer still gets a sane initial range.
- The `ridge`, `territory`, and `atoms` mask/label layers have no contrast to
  preserve, but follow the same "update `data`, don't recreate" rule so their
  per-label colors stay stable across refreshes.

## Out of scope

- No change to `AtomParams` or the `atoms.tif` format / fingerprint. The core
  function signatures *do* change (§4) to return `ridge`, but the params and the
  on-disk artifact are untouched.
- No change to the debounced worker / pending-coalesce logic.
- No change to the residual / threshold math itself — the hollow-ridge
  hypothesis is only *visualized* here, not fixed. Any change to `residual()`
  (window, top-hat, strength default) is a separate follow-up once the mask
  confirms the cause.
- The deferred `_reject_conflicting_viewer_activity` exclusive-mode gap noted in
  the stage-① memory remains out of scope.

## Testing

- Update `tests/napari/test_nucleus_atom_extraction_widget.py` for the layer
  rename and the new per-group visibility toggles (assert that toggling the
  Contour eye flips all three of its layers' `visible` — `residual_contour`,
  `ridge`, `atoms` — and the Foreground eye flips its two; and the default
  on-activation visibility state).
- `tests/tracking_ultrack/test_atoms.py`: update the existing
  `extract_atoms_frame` call sites to the `(atoms, ridge)` return, and the
  `extract_atoms_stack_with_maps` test (`:171`) to the 5-tuple. Add: `ridge`
  equals `residual_contour > contour_floor`, is `uint8`, shares the frame shape,
  and `extract_atoms_stack` still returns labels equal to the 5-tuple's first
  element.
- Assert the `ridge` layer exists after a refresh and that its data is exactly
  the mask returned by the core (not a widget re-derivation), so the layer can
  never drift from the split.
- Assert contrast is preserved across slider changes: set a non-default
  `contrast_limits` on `residual_contour` (and/or `residual_foreground`), trigger
  a param refresh, and assert the limits are unchanged afterwards.
