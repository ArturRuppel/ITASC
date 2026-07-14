# Installation

ITASC requires **Python 3.10+**. What you install depends on where you enter
the pipeline: the full app for the whole workflow, or a single piece for one
stage. If you have not decided yet, [Choosing your install](index.md) routes you
by your data.

## Full workflow (the napari plugin)

```bash
pip install itasc[all]
```

The `[all]` extra pulls the core scientific stack plus the optional workflow
engines: Cellpose-SAM (`cellpose`, `torch`, `torchvision`) and the Ultrack
solver. From a local checkout:

```bash
python -m pip install -e .[all]
```

## A single piece

Each tool installs on its own and shares the `itasc.*` namespace, so
`import itasc.<stage>` works whether or not the full app is present.

```bash
pip install "itasc-cellpose[cellpose,laptrack]"  # sparse cells: segment + track
pip install "itasc-tracking[solve]"              # maps → tracks + correction
pip install itasc-aggregate                      # tracked labels → contacts (HDF5)
pip install itasc-core                           # shared library only
```

Divergence-based cell segmentation is not published as a standalone wheel: it
ships inside the full `itasc` app (`pip install itasc[all]`).

## Optional engines

The extras below pull in heavy dependencies. They are imported lazily, loaded
only when you run the stage that uses them, so a browse-or-correct session
without the extra still works.

- **Cellpose / Cellpose-SAM** (`cellpose>=4.0`, PyTorch): the local map-building
  runner. GPU is used when PyTorch detects it; CPU is the fallback. Extra:
  `[cellpose]`.
- **Ultrack**: candidate segmentation, database construction, linking, and
  solving for nucleus tracking. Extra: `[solve]`, or `[tracking]` on the full
  `itasc` distribution.

## Development install

```bash
python -m pip install -e .[dev]    # + ruff, pytest
python -m pip install -e .[docs]   # to build this documentation locally
```

With the full app installed, the [Full app guide](full-app.md) walks through what
each stage reads and writes, and how to drive the plugin in napari.
