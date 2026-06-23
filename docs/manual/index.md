# Choosing your install

CellFlow is factored into independently installable distributions that share the
`cellflow.*` namespace ([PEP 420](https://peps.python.org/pep-0420/)). The full
plugin orchestrates them into one unified workflow, but each piece installs and
runs on its own. Pick the row that matches what you actually do.

| If you are… | Install | What you get | Depends on |
| --- | --- | --- | --- |
| Running the **whole pipeline** in napari | `pip install cellflow[all]` | The unified `CellFlow` workflow widget end to end | everything |
| Producing **probability/flow + divergence maps** from raw stacks | `pip install "cellflow-cellpose[cellpose]"` | Local Cellpose-SAM runner + foreground/contour map building | `cellflow-core` |
| **Tracking & correcting nuclei** | `pip install "cellflow-tracking[solve]"` | Ultrack candidate DB, solving, browsing, interactive correction | `cellflow-core` |
| **Segmenting & correcting cells** | `pip install cellflow-segmentation` | Divergence-based geodesic-Voronoi cell labels + correction | `cellflow-core` |
| Doing **contact / aggregate analysis** | `pip install cellflow-aggregate` | Cell-cell edges, T1 events, NLS classes → HDF5 + napari views | `cellflow-core` |
| Building **on top of CellFlow** as a library | `pip install cellflow-core` | TIFF/path/label-IO helpers, lineage model, napari UI primitives | — |

The optional extras (`[cellpose]`, `[solve]`) pull in heavy engines (Cellpose +
PyTorch, the Ultrack solver) that are imported lazily — so correction-only or
map-building-only use does not require them.

## How the pieces fit

CellFlow processes a project directory with **one subdirectory per position**
(`pos00`, `pos01`, …). Each position is processed through staged subdirectories,
and each distribution owns one stage:

```{mermaid}
flowchart LR
    raw["0_input<br/>raw stacks"] --> cp
    cp["cellflow-cellpose<br/>1_cellpose: prob/flow + maps"] --> trk
    cp --> seg
    trk["cellflow-tracking<br/>2_nucleus: DB, tracks, labels"] --> seg
    seg["cellflow-segmentation<br/>3_cell: tracked cell labels"] --> agg
    trk --> agg
    agg["cellflow-aggregate<br/>aggregate_quantification: HDF5"]
```

See [The staged workflow](workflow.md) for what happens at each stage, and
[Installation](install.md) for the dependency/extra matrix.
