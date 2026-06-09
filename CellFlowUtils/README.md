# CellFlow Utils

Private, site-specific napari utilities for local CellFlow workflows.

This project intentionally lives outside the public `cellflow` package. It contains utilities for NDTiff data preparation, launching a site-specific external/HPC Cellpose workflow, and NLS classification helpers.

## Development

From a checkout with CellFlow available locally:

```bash
PYTHONPATH=/home/aruppel/Projects/CellFlow/src:/home/aruppel/Projects/CellFlowUtils/src python -m pytest
```

To load the plugin in napari from source:

```bash
PYTHONPATH=/home/aruppel/Projects/CellFlow/src:/home/aruppel/Projects/CellFlowUtils/src napari
```
