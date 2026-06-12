# Installation

CellFlow requires **Python 3.10+**.

## Full workflow (the napari plugin)

```bash
pip install cellflow[all]
```

This pulls the core scientific stack plus the optional workflow engines —
Cellpose-SAM (`cellpose`, `torch`, `torchvision`) and the Ultrack solver. From a
local checkout:

```bash
python -m pip install -e .[all]
```

## A single piece

Each stage installs on its own and shares the `cellflow.*` namespace, so
`import cellflow.<stage>` works whether or not the full orchestrator is present.

```bash
pip install "cellflow-cellpose[cellpose]"   # raw stacks  → prob/flow + maps
pip install "cellflow-tracking[solve]"      # maps        → tracks + correction
pip install cellflow-segmentation           # maps + seeds → cell labels
pip install cellflow-aggregate              # cell labels → contacts (HDF5)
pip install cellflow-core                   # shared library only
```

## Optional engines

These are heavy and imported lazily, so they are only needed when you actually
run the stage that uses them:

- **Cellpose / Cellpose-SAM** (`cellpose>=4.0`, PyTorch): the local map-building
  runner. GPU is used when PyTorch detects it; CPU is the fallback. Extra:
  `[cellpose]`.
- **Ultrack**: candidate segmentation, database construction, linking, and
  solving for nucleus tracking. Extra: `[solve]` (or `[tracking]` on the full
  `cellflow` distribution).

## Development install

```bash
python -m pip install -e .[dev]    # + ruff, pytest
python -m pip install -e .[docs]   # to build this documentation locally
```
