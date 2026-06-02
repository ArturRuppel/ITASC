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

## Design

### 1. Layer rename & styling

- Rename the atoms label layer `[Atoms] preview` → `[Atoms] atoms`
  (constant `_ATOM_PREVIEW_LAYER` value becomes `"[Atoms] atoms"`).
- Both label layers become masks at **opacity 0.7** (was 0.55), each ordered
  **directly on top of the residual image it is judged against**:
  - `[Atoms] territory` on top of `[Atoms] residual_foreground`
  - `[Atoms] atoms` on top of `[Atoms] residual_contour`
- Fixed stack order, bottom → top:
  `residual_foreground`, `territory`, `residual_contour`, `atoms`.
  Each mask sits immediately above the image it is judged against.

### 2. Params split into two groups

The single "Atom Extraction Params" section becomes two collapsible subgroups:

- **Foreground** — `fg_window`, `fg_cutoff`, `fg_strength`
  → drive `residual_foreground` + `territory`.
- **Contour** — `contour_window`, `contour_floor`, `contour_strength`,
  **and `atom_min_area`** → drive `residual_contour` + `atoms`.

`atom_min_area` moves into the Contour group because it only post-processes
atoms. The header ⚙ button still expands/collapses the whole params area.

### 3. Per-group layer toggle (the core fix)

Each group's header gets a small **eye toggle** controlling the visibility of
*that group's two layers only*, independently:

- Foreground eye → show/hide `residual_foreground` + `territory`.
- Contour eye → show/hide `residual_contour` + `atoms`.
- Independent: 0, 1, or both pairs may be shown at any time.

### 4. Behavior

- The ◉ live-preview worker is **unchanged**: it still computes all four arrays
  in one pass each refresh (atoms requires `territory`, so there is no compute
  to skip). The group eye toggles are pure napari visibility.
- On activating ◉: **Foreground pair visible, Contour pair hidden** by default.
  The eye toggles reflect this initial state.
- `_run_atom_extraction` (full-stack path) reuses the same layers, rename, and
  visibility rules.

## Out of scope

- No change to `atoms.py` core functions, `AtomParams`, or the `atoms.tif`
  format / fingerprint.
- No change to the debounced worker / pending-coalesce logic.
- The deferred `_reject_conflicting_viewer_activity` exclusive-mode gap noted in
  the stage-① memory remains out of scope.

## Testing

- Update `tests/napari/test_nucleus_atom_extraction_widget.py` for the layer
  rename and the new per-group visibility toggles (assert that toggling a
  group's eye flips only that group's two layers' `visible`, and the default
  on-activation visibility state).
