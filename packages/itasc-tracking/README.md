# itasc-tracking

Independent ITASC piece for **nucleus tracking and interactive correction**:
turn 2D+t foreground + contour maps into Ultrack candidate segments, a tracking
database, solved tracks, and validated/corrected nucleus labels — all from a
napari plugin.

## Install

```bash
pip install itasc-tracking          # correction + browsing
pip install "itasc-tracking[solve]" # + the Ultrack solver (db build / track)
```

This pulls in `itasc-core`. Everything installs into the shared `itasc.*`
namespace (PEP 420), so `import itasc.tracking_ultrack` works whether or not
the full ITASC orchestrator is present. The Ultrack engine is only imported
lazily when you actually run candidate generation / linking / solving, so
correction-only use does not require the `[solve]` extra.

## Use

- **napari plugin:** launch napari and add the *Ultrack Segment + Track*
  widget. It exposes three path fields:
  - **Foreground** — the foreground probability/intensity `.tif` (2D+t),
  - **Contours** — the contour/boundary `.tif` (2D+t),
  - **Output dir** — where every artifact is written (defaults to the
    foreground file's folder when left blank).

  The two inputs can have any name and live anywhere.
- **Standalone path contract:**

  ```text
  <foreground.tif>            (in, any name/location)
  <contours.tif>             (in, any name/location)
  output_dir/
    atoms.tif                 (out)
    ultrack_workdir/data.db   (out)
    tracked_labels.tif        (out)
    validated_cells.json      (out)
    corrections.json          (out)
  ```

The full `itasc` orchestrator drives the same widget through its staged
`<pos>/1_cellpose` + `<pos>/2_nucleus` layout instead; the piece supports both
via `itasc.napari._paths.NucleusWorkspace` (`files()` / `flat()` vs
`staged()`).

## Backend

The headless backend lives under `itasc.tracking_ultrack` (candidate atoms,
database build, linking, solving, export, correction primitives, and the
`validation_state` validation/correction annotation store) and
`itasc.correction` (label-edit helpers). Generic tracked-label-stack IO is
shared substrate in `itasc.core.label_store`.
