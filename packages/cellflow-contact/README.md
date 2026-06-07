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

- **napari plugin:** launch napari and add the *Contact Analysis* widget. Pick a
  cell-labels TIFF (2D+t), an optional nucleus-labels TIFF, and an output `.h5`.
- **Headless / scripting:**

  ```python
  from cellflow.contact_analysis import build_contact_analysis

  build_contact_analysis(
      cell_labels_path="cells.tif",
      nucleus_labels_path=None,            # optional
      output_path="contact_analysis.h5",
  )
  ```

## I/O contract

- **Input:** `cell_labels` (2D+t TIFF, required); `nucleus_labels` (2D+t TIFF,
  optional — when given, the `cell_id == nucleus_id` invariant is enforced).
- **Output:** an HDF5 file with `cells`, `edges`, `t1_events`, and `provenance`.
