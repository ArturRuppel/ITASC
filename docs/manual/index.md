# Choosing your install

ITASC is one pipeline, factored into independently installable pieces that
share the `itasc.*` namespace ([PEP 420](https://peps.python.org/pep-0420/)).
The full plugin orchestrates them into one workflow, but each piece installs and
runs on its own. Which one you want is decided by your data and where you enter
the pipeline, not by taste: read down the first column until a row fits.

| If you have… | Reach for | It gives you |
| --- | --- | --- |
| **Sparse, well-separated cells** with a cell and/or nucleus marker, and you want to segment and track one or both channels | `pip install "itasc-cellpose[cellpose,laptrack]"` | A local Cellpose-SAM runner for native masks, then `laptrack` linking across time, with correction. One channel or two. |
| **Dense, motile cells of varying shape** (a confluent monolayer), from raw stacks to quantified contacts | `pip install itasc[all]` | The unified `ITASC` workflow widget, every stage end to end. |
| **Foreground and contour maps already** and you want to skip segmentation | `pip install "itasc-tracking[solve]"` | Ultrack candidate database, solving, browsing, and interactive nucleus correction. |
| **Tracked cell labels already** and you are here for the analysis | `pip install itasc-aggregate` | Cell-cell edges, border edges, and T1 events to HDF5, with napari views. |
| **Code to build on** | `pip install itasc-core` | TIFF/path/label-IO helpers, the lineage model, and napari UI primitives. |

The optional extras (`[cellpose]`, `[laptrack]`, `[solve]`) pull in heavy engines
(Cellpose plus PyTorch, laptrack, the Ultrack solver). They are imported lazily,
so a correction-only session never loads them, and you can drop the extra when
you only need to browse or correct existing results.

Divergence-based cell segmentation is not a standalone wheel: it ships inside the
full `itasc` app (`pip install itasc[all]`) as the **Cell** stage of the
workflow widget.

## How the pieces fit

ITASC processes a project directory with **one subdirectory per position**
(`pos00`, `pos01`, …). Each position moves through staged subdirectories. The
full `itasc` app drives them end to end; each standalone piece owns one stage
and reads the previous stage's files off disk, so you can enter the pipeline
wherever your data already sits.

```{mermaid}
flowchart LR
    raw["0_input<br/>raw stacks"] --> cp
    cp["itasc-cellpose<br/>1_cellpose: prob/flow + maps"] --> trk
    cp --> seg
    trk["itasc-tracking<br/>2_nucleus: DB, tracks, labels"] --> seg
    seg["itasc (full app)<br/>3_cell: tracked cell labels"] --> agg
    trk --> agg
    agg["itasc-aggregate<br/>aggregate_quantification: HDF5"]
```

See [The staged workflow](workflow.md) for what each stage produces, and
[Installation](install.md) for the dependency and extras matrix.
