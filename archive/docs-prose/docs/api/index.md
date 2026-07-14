# API Reference

This reference is **generated from the source docstrings** — browse from each
package's overview down to individual functions and classes.

CellFlow's supported programmatic entry points are the workflow subpackages
below. `import cellflow` itself stays intentionally light (only `__version__`)
and avoids importing napari, Cellpose, or Ultrack. The napari UI layer
(`cellflow.napari`) is documented in the [User Manual](../manual/index.md)
rather than here.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   cellflow.core
   cellflow.cellpose
   cellflow.segmentation
   cellflow.tracking_ultrack
   cellflow.correction
   cellflow.contact_analysis
```
