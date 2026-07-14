# itasc-aggregate

Independent ITASC piece for **aggregate quantification**. It hosts per-position
quantifiers; the bundled one is **contacts**: extract cell-cell edges, border
edges, and T1 events from 2D+t cell-label stacks (optionally validated against
nucleus labels) into a self-describing HDF5 file, and visualize the result in
napari.

## Install

```bash
pip install itasc-aggregate
```

This pulls in `itasc-core`. Both install into the shared `itasc.*`
namespace (PEP 420), so `import itasc.contact_analysis` works whether
or not the full ITASC orchestrator is present.

## Use

The napari plugin is the full ITASC catalog app restricted to the contact
step: it does not run segmentation or tracking, so it treats the committed
`cell_labels.tif` (required) and `nucleus_labels.tif` (optional) as *inputs* and
produces the contact-analysis `.h5`.

- **napari plugin:** add the *ITASC Aggregate* widget.
  - **Data folders**: point *Find* at a parent directory to discover every
    **position** (a folder containing `cell_labels.tif`) beneath it; the columns
    are derived from each position's nesting under that directory. Each row carries
    a three-dot **status rail** (cell labels, nucleus labels, contact analysis)
    that shows, at a glance, which inputs are present and whether the `.h5` is
    built. Click a rail dot to load that input into the viewer (or, for the
    contact dot, open the overlays).
  - **Contact Analysis**: select a row to retarget the stage to that position;
    run or re-run its contact analysis and visualize the result.
  - **Aggregate**: the project-level capstone pools across the whole catalog,
    grouped by your columns, once positions have been added.
- **Headless / scripting:**

  ```python
  from itasc.contact_analysis import (
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
  optional: when given, the `cell_id == nucleus_id` invariant is enforced).
- **Output:** an HDF5 file with `cells`, `edges`, `t1_events`, and `provenance`.
