# cellflow-tracking

Independent CellFlow piece for **nucleus tracking and interactive correction**:
turn 2D+t foreground + contour maps into Ultrack candidate segments, a tracking
database, solved tracks, and validated/corrected nucleus labels — all from a
napari plugin.

## Install

```bash
pip install cellflow-tracking          # correction + browsing
pip install "cellflow-tracking[solve]" # + the Ultrack solver (db build / track)
```

This pulls in `cellflow-core`. Everything installs into the shared `cellflow.*`
namespace (PEP 420), so `import cellflow.tracking_ultrack` works whether or not
the full CellFlow orchestrator is present. The Ultrack engine is only imported
lazily when you actually run candidate generation / linking / solving, so
correction-only use does not require the `[solve]` extra.

## Use

- **napari plugin:** launch napari and add the *Nucleus Tracking & Correction*
  widget. Pick a **working directory** that holds `foreground.tif` and
  `contours.tif` (2D+t); every output is written back into it.
- **Working-directory contract (standalone, flat layout):**

  ```text
  workdir/
    foreground.tif            (in)
    contours.tif              (in)
    atoms.tif                 (out)
    ultrack_workdir/data.db   (out)
    tracked_labels.tif        (out)
    validated_cells.json      (out)
    corrections.json          (out)
  ```

The full `cellflow` orchestrator drives the same widget through its staged
`<pos>/1_cellpose` + `<pos>/2_nucleus` layout instead; the piece supports both
via `cellflow.napari._paths.NucleusWorkspace` (`flat()` vs `staged()`).

## Backend

The headless backend lives under `cellflow.tracking_ultrack` (candidate atoms,
database build, linking, solving, export, correction primitives),
`cellflow.database` (validation/correction annotation store), and
`cellflow.correction` (label-edit helpers).
