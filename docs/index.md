# CellFlow

CellFlow segments and tracks cells in time-lapse microscopy, corrects the result
by hand where the automatics miss, and quantifies what the tracked cells do. It
is a [napari](https://napari.org) plugin, factored into independent pieces that
share the `cellflow.*` namespace: install the whole pipeline, or install only the
stage your data actually needs.

:::{admonition} Status
:class: note
CellFlow is under active research development. Treat the workflow, installation
details, and public API as provisional until the public release and accompanying
manuscript stabilize.
:::

## Which piece do you need?

Read down the first column until a row describes your data and your goal. Each
piece installs and runs on its own; the full `cellflow` app orchestrates all of
them into one workflow.

| If you have… | Reach for | It gives you |
| --- | --- | --- |
| **Sparse, well-separated cells** with a cell and/or nucleus marker, and you want to segment and track one or both channels | `cellflow-cellpose` | A local Cellpose-SAM runner for native masks, then `laptrack` linking across time, with correction. One channel or two. |
| **Dense, motile cells of varying shape** (a confluent monolayer), from raw stacks to quantified contacts | `cellflow[all]` (the full app) | The unified `CellFlow` workflow: Cellpose maps, divergence-based cell segmentation, Ultrack tracking, correction, and contact analysis, end to end. |
| **Foreground and contour maps already** and you want to skip segmentation | `cellflow-tracking` | Ultrack candidate database, track solving, browsing, and interactive nucleus correction. |
| **Tracked cell labels already** and you are here for the analysis | `cellflow-aggregate` | Cell-cell edges, border edges, and T1 events to HDF5, with napari views. |
| **Code to build on** | `cellflow-core` | TIFF/path/label-IO helpers, the track lineage model, and reusable napari UI primitives. |

Divergence-based cell segmentation is not sold separately: it ships inside the
full `cellflow` app as the **Cell** stage of the workflow widget. The
[install guide](manual/install.md) has the full extras matrix, and
[Choosing your install](manual/index.md) explains how the pieces depend on one
another.

## Read next

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 📖 User Manual
:link: manual/index
:link-type: doc

Which distribution is for which kind of user, the staged position-directory
workflow, and how to install and drive the napari plugin.
:::

:::{grid-item-card} 🧩 API Reference
:link: api/index
:link-type: doc

The programmatic API, generated from the source. Browse from a package-level
overview down to individual functions and classes.
:::

::::

```{toctree}
:hidden:
:caption: User Manual

manual/index
manual/workflow
manual/install
```

```{toctree}
:hidden:
:caption: Reference

api/index
```

```{toctree}
:hidden:
:caption: Development

development/architecture
development/maintaining-docs
```
