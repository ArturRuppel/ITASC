# Choosing your install

CellFlow is factored into independently installable distributions that share the
`cellflow.*` namespace ([PEP 420](https://peps.python.org/pep-0420/)). The full
plugin orchestrates them into one unified workflow, but each piece installs and
runs on its own. Pick the row that matches what you actually do.

| If you are… | Install | What you get | Depends on |
| --- | --- | --- | --- |
| Running the **whole pipeline** in napari | `pip install cellflow[all]` | The unified `CellFlow` workflow widget end to end | everything |
| **Segmenting & tracking** with Cellpose | `pip install "cellflow-cellpose[cellpose,laptrack]"` | Cellpose-SAM native masks + laptrack tracking of 1–2 channels, with correction | `cellflow-core` |
| **Tracking & correcting nuclei** with Ultrack | `pip install "cellflow-tracking[solve]"` | Ultrack candidate DB, solving, browsing, interactive correction | `cellflow-core` |
| Doing **contact / aggregate analysis** | `pip install cellflow-contact-analysis` | Cell-cell edges, T1 events, NLS classes → HDF5 + napari views | `cellflow-core` |
| Building **on top of CellFlow** as a library | `pip install cellflow-core` | TIFF/path/label-IO helpers, lineage model, napari UI primitives | — |

The optional extras (`[cellpose]`, `[laptrack]`, `[solve]`) pull in heavy engines
(Cellpose + PyTorch, laptrack, the Ultrack solver) that are imported lazily — so
correction-only use does not require them.

Divergence-based cell segmentation is no longer a standalone wheel; it ships
inside the full `cellflow` plugin (`pip install cellflow[all]`) as the **Cell**
stage of the `CellFlow` workflow widget.

## How the pieces fit

CellFlow processes a project directory with **one subdirectory per position**
(`pos00`, `pos01`, …). Each position is processed through staged subdirectories;
the full `CellFlow` app drives them end to end, and most stages are also
available as a standalone wheel:

```{mermaid}
flowchart LR
    raw["0_input<br/>raw stacks"] --> cp
    cp["cellflow-cellpose<br/>1_cellpose: prob/flow + maps"] --> trk
    cp --> seg
    trk["cellflow-tracking<br/>2_nucleus: DB, tracks, labels"] --> seg
    seg["cellflow (full app)<br/>3_cell: tracked cell labels"] --> agg
    trk --> agg
    agg["cellflow-contact-analysis<br/>aggregate_quantification: HDF5"]
```

See [The staged workflow](workflow.md) for what happens at each stage, and
[Installation](install.md) for the dependency/extra matrix.
