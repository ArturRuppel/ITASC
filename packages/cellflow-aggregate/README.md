# cellflow-aggregate

Independent CellFlow piece for **aggregate quantification**. It hosts per-position
quantifiers; the bundled one is **contacts**: extract cell-cell edges, border
edges, and T1 events from 2D+t cell-label stacks (optionally validated against
nucleus labels) into a self-describing HDF5 file, and visualize the result in
napari.

## Install

```bash
pip install cellflow-aggregate
```

This pulls in `cellflow-core`. Both install into the shared `cellflow.*`
namespace (PEP 420), so `import cellflow.contact_analysis` works whether
or not the full CellFlow orchestrator is present.

## Use

The widget is **discovery-first**: you give it a top folder and the names of the
three files (cell labels, optional nucleus labels, output `.h5`); it lists every
discovered **position** and you pick which to view. The `.h5` is a derived
artifact, computed on demand only when missing.

- **napari plugin:** add the *Aggregate Quantification* widget and set the **Top folder**
  plus the three file names. Each **position** — a top-level subfolder of that
  folder — appears as a row showing whether a nucleus was paired and whether its
  `.h5` is built. The named files are discovered recursively within a position, so
  the cell and nucleus may live in different subfolders (e.g. `pos01/3_cell/` and
  `pos01/2_nucleus/`); the nucleus is associated only when exactly one is found
  (zero or several → cell-only). **Double-click a row** to visualize it (its `.h5`
  is computed first if missing); **Recompute** forces a rebuild of the selection.
- **Process all:** computes every discovered `.h5` at once (skip existing unless
  **Overwrite**), headlessly, with progress.
- **Headless / scripting:**

  ```python
  from cellflow.contact_analysis import (
      ensure_contacts,        # build only if missing (or overwrite=True)
      discover_contact_batch_jobs,
      run_contact_batch,
  )

  # Single position (missing-only):
  ensure_contacts(
      cell_labels_path="cells.tif",
      nucleus_labels_path=None,            # optional
      output_path="contact_analysis.h5",
  )

  # Batch by name-based autodiscovery:
  jobs = discover_contact_batch_jobs(
      "/data/study",
      cell_name="cell_labels.tif",
      nucleus_name="nucleus_labels.tif",  # optional
      h5_name="contact_analysis.h5",
  )
  results = run_contact_batch(jobs, overwrite=False)
  ```

  `build_contacts(...)` remains available for an unconditional build.

## I/O contract

- **Input:** `cell_labels` (2D+t TIFF, required); `nucleus_labels` (2D+t TIFF,
  optional — when given, the `cell_id == nucleus_id` invariant is enforced).
- **Output:** an HDF5 file with `cells`, `edges`, `t1_events`, and `provenance`.
