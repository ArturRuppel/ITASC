# API Reference

This reference is **generated from the source docstrings** — browse from each
package's overview down to individual functions and classes.

ITASC's supported programmatic entry points are the workflow subpackages
below. `import itasc` itself stays intentionally light (only `__version__`)
and avoids importing napari, Cellpose, or Ultrack. The napari UI layer
(`itasc.napari`) is documented in the [User Manual](../manual/index.md)
rather than here.

```{eval-rst}
.. autosummary::
   :toctree: generated
   :recursive:

   itasc.core
   itasc.cellpose
   itasc.segmentation
   itasc.tracking_ultrack
   itasc.correction
   itasc.contact_analysis
```
