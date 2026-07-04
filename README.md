# CellFlow

CellFlow is a napari-based research software project for time-lapse cell
microscopy. It brings together Cellpose-derived probability and flow outputs,
Ultrack-based nucleus tracking, interactive correction, cell-label propagation,
validation-aware resolving, and downstream contact analysis.

## Status

CellFlow is under active research development. The repository is being prepared
for eventual public release and possible JOSS submission, but the workflow,
installation details, and public API should still be treated as provisional.
Claims in this README are descriptive only; no benchmarking or performance
superiority is claimed here.

## Independent packages

CellFlow is being factored into independently-installable pieces that share the
`cellflow.*` namespace (PEP 420), so others can install just the part they need.
The full plugin orchestrates them into the unified workflow described below.

- [`cellflow-tracking`](packages/cellflow-tracking) — Ultrack-based nucleus
  tracking + interactive correction (napari). Flat working-directory contract:
  2D+t `foreground.tif` + `contours.tif` → atoms, database, tracked labels, and
  validated/corrected annotations. The Ultrack solver is an optional `[solve]`
  extra. Depends on `cellflow-core`.
- [`cellflow-aggregate`](packages/cellflow-aggregate) — aggregate quantification;
  bundled quantifier is contacts (cell-cell edges, T1 events) + napari
  visualization. Headless I/O: 2D+t cell labels (optional nucleus labels) → HDF5.
  Depends on `cellflow-core`.
- [`cellflow-cellpose`](packages/cellflow-cellpose) — **Cellpose Segment +
  Track** (napari): a local Cellpose-SAM native-mask runner + laptrack tracking
  for one or two channels, with embedded basic correction. It also ships the
  Cellpose runner + divergence-map builder that the full `cellflow` app uses for
  its in-app Cellpose stage. The Cellpose model is an optional `[cellpose]`
  extra. Depends on `cellflow-core`.
- [`cellflow-core`](packages/cellflow-core) — shared TIFF/path helpers, generic
  image ops + label-stack IO, the shared interactive-correction base, the track
  lineage model, and reusable napari UI primitives.

Divergence-based cell segmentation is no longer published as a standalone wheel;
it ships inside the full `cellflow` plugin (`pip install cellflow[all]`), reachable
as the **Cell** stage of the `CellFlow` workflow widget.

## Main Capabilities

- napari plugin UI with a unified `CellFlow` workflow widget.
- Local Cellpose-SAM runner for nucleus and cell channels.
- Nucleus segmentation-input generation, Ultrack database building, solving,
  database browsing, and interactive correction.
- Cell segmentation workflow using Cellpose-derived probabilities, flow
  filtering, foreground masks, contour maps, and tracked label output.
- Interactive nucleus/cell correction tools with synchronized label selection.
- Validation/anchor-aware resolving for corrected tracks.
- Contact-analysis export to HDF5 and napari visualization of cell contacts,
  edges, and T1 events.

## High-Level Workflow

CellFlow expects a project directory containing one directory per position, for
example `pos00`, `pos01`, and so on. The current workflow uses staged
subdirectories inside each position:

```text
pos00/
  0_input/             raw prepared input stacks
  1_cellpose/          Cellpose probability and flow outputs
  2_nucleus/           nucleus segmentation, Ultrack database, tracked labels
  3_cell/              cell segmentation and tracked labels
  aggregate_quantification/  quantification output (contact_analysis.h5 + tables)
```

Typical use is:

1. Provide input stacks under `0_input/`.
2. Run Cellpose for nucleus and cell channels to create probability, flow, and
   z-average TIFFs under `1_cellpose/`.
3. Build nucleus contour/foreground sources, create an Ultrack database, solve
   tracks, and correct/validate nucleus labels under `2_nucleus/`.
4. Generate cell foregrounds/contours and tracked cell labels under `3_cell/`.
5. Build quantification under `aggregate_quantification/` and inspect the results
   in napari.

## Installation

CellFlow is packaged with `pyproject.toml` and currently requires Python 3.10 or
newer.

For development from a local checkout:

```bash
python -m pip install -e .
```

For the full interactive workflow, install the optional Cellpose and Ultrack
dependencies:

```bash
python -m pip install -e .[all]
```

For linting support declared by the project:

```bash
python -m pip install -e .[dev]
```

The core declared dependencies include napari, Qt support through `qtpy`,
NumPy/SciPy/scikit-image, pandas, tifffile, h5py, SQLAlchemy, matplotlib,
pymaxflow, pydantic, and numba. Cellpose, PyTorch, torchvision, and Ultrack are
declared as optional workflow extras.

## External and Optional Tools

- **Cellpose / Cellpose-SAM**: CellFlow includes a local runner based on
  `cellpose>=4.0`. Install with `python -m pip install -e .[cellpose]`.
  GPU use is detected through PyTorch when available; CPU use is the fallback.
- **Ultrack**: The nucleus-tracking stages import Ultrack for candidate
  segmentation, database construction, linking, and solving. Install with
  `python -m pip install -e .[tracking]`.

## Basic Usage in napari

After installation, start napari:

```bash
napari
```

Then open the main plugin widget from the napari plugin menu:

- `Plugins > CellFlow > CellFlow`

In the main `CellFlow` widget:

1. Select a project directory.
2. Set or load project metadata such as pixel size, time interval, condition,
   and position.
3. Expand the workflow sections in order: project status, Cellpose, nucleus
   segmentation/tracking, cell segmentation, and contact analysis.

The widget can save and load `cellflow_config.json` files in the selected
project directory.

## Public API Boundary

CellFlow is currently published primarily as a napari plugin and research
workflow package. The top-level `import cellflow` exposes only `__version__`
and intentionally avoids importing napari, Cellpose, Ultrack, or other heavy
workflow dependencies.

Programmatic use should import from the relevant subpackages, such as
`cellflow.segmentation`, `cellflow.tracking_ultrack`,
`cellflow.correction`, and `cellflow.contact_analysis` (with generic
label-stack IO in `cellflow.core.label_store`). These subpackage APIs
are useful for scripting and testing, but they should still be treated as
provisional until the public release and manuscript stabilize.

## Testing

Install the package with the declared development dependencies, then run:

```bash
python -m pip install -e .[dev]
```

```bash
python -m pytest
```

For headless systems, set Qt to offscreen before running napari/Qt tests:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest tests/napari
```

Focused test examples:

```bash
python -m pytest tests/segmentation
python -m pytest tests/tracking_ultrack
python -m pytest tests/contact_analysis
```

Some tests or workflow stages may require optional packages such as Ultrack,
Cellpose, or napari/Qt.

## Archived Experiments

Exploratory one-off scripts are kept under `notes/archived_scripts/` so the
installable package and active test surface stay focused on maintained code.

## AI Usage Disclosure

Generative AI tools were used during CellFlow development and documentation
preparation. The tools included OpenAI GPT-5.5 and Anthropic Claude Opus 4.7,
Claude Opus 4.6, and Claude Sonnet 4.6.

AI assistance was used for software-development support, including code
drafting, refactoring suggestions, test scaffolding, debugging assistance,
documentation drafting, and editorial review. Human authors made the core
scientific, architectural, and design decisions. Human authors reviewed,
modified, and validated AI-assisted outputs before incorporating them into the
repository and remain responsible for the accuracy, originality, licensing, and
ethical/legal compliance of the submitted work.

## Citation

If you use CellFlow, please cite the software using the metadata in
`CITATION.cff`. A DOI and manuscript citation will be added when the public
release and associated manuscript are ready. For pre-publication citation
questions, contact Artur Ruppel at `artur@ruppel.pro`.

## License

CellFlow is distributed under the AGPL-3.0 license. See the repository-level
`LICENSE` file for the full license terms.
