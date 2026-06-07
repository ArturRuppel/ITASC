# cellflow-contact

Independent CellFlow piece for **contact analysis**: extract cell-cell edges,
border edges, and T1 events from 2D+t cell-label stacks (optionally validated
against nucleus labels) into a self-describing HDF5 file, and visualize the
result in napari.

## Install

```bash
pip install cellflow-contact
```

This pulls in `cellflow-core`. Both install into the shared `cellflow.*`
namespace (PEP 420), so `import cellflow.contact_analysis` works whether or not
the full CellFlow orchestrator is present.

## Use

The widget is **visualizer-first**: the `.h5` is a derived artifact, so you point
at the inputs and hit **Visualize** — the file is computed on demand only when it
is missing, then shown. **Recompute** forces a rebuild.

- **napari plugin (single position):** add the *Contact Analysis* widget, pick a
  cell-labels TIFF (2D+t) and an optional nucleus-labels TIFF. The output `.h5`
  defaults to `<cell_labels_dir>/contact_analysis.h5` (override with the optional
  output picker). Click **Visualize**.
- **napari plugin (batch):** expand the **Batch** panel, name the three files
  (cell labels, optional nucleus labels, output `.h5`) and pick a top-level folder.
  Every folder under it that contains a cell-labels file becomes one job; a nucleus
  file is associated only when it sits in that same folder. Existing outputs are
  skipped unless **Overwrite** is checked. Runs headlessly (no visualization).
- **Headless / scripting:**

  ```python
  from cellflow.contact_analysis import (
      ensure_contact_analysis,        # build only if missing (or overwrite=True)
      discover_contact_batch_jobs,
      run_contact_batch,
  )

  # Single position (missing-only):
  ensure_contact_analysis(
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

  `build_contact_analysis(...)` remains available for an unconditional build.

## I/O contract

- **Input:** `cell_labels` (2D+t TIFF, required); `nucleus_labels` (2D+t TIFF,
  optional — when given, the `cell_id == nucleus_id` invariant is enforced).
- **Output:** an HDF5 file with `cells`, `edges`, `t1_events`, and `provenance`.
