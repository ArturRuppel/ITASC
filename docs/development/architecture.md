# Architecture overview

This page is the high-altitude map. It is kept deliberately conceptual so it
changes slowly; the authoritative, always-current detail lives in the
[API Reference](../api/index.md), generated from the code.

## Distribution / dependency graph

Every distribution depends only on `itasc-core` and communicates with the
others through files on disk, not Python calls — so each stage can be installed,
run, and tested in isolation.

```{mermaid}
flowchart TD
    core["itasc-core<br/><i>TIFF/path/label IO · lineage model · napari UI primitives</i>"]
    cp["itasc-cellpose"]
    trk["itasc-tracking"]
    agg["itasc-aggregate"]
    cp --> core
    trk --> core
    agg --> core
```

## Source layout

Each top-level subpackage under `src/itasc/` is one importable stage. Most map
to a standalone distribution (`itasc-cellpose`, `itasc-tracking`,
`itasc-aggregate`, `itasc-core`); `itasc.segmentation` ships only inside
the full `itasc` app.

| Subpackage | Role |
| --- | --- |
| `itasc.core` | Shared TIFF/path/logging helpers, label-stack IO, the interactive-correction base, the track lineage model |
| `itasc.cellpose` | Local Cellpose-SAM runner + divergence-based foreground/contour map building |
| `itasc.tracking_ultrack` | Ultrack candidate generation, database, linking, solving |
| `itasc.segmentation` | Divergence-based geodesic-Voronoi cell segmentation |
| `itasc.correction` | Interactive nucleus/cell label correction operations |
| `itasc.contact_analysis` | Pooling per-position sources into plottable quantities (contacts, T1 events, NLS classes) |
| `itasc.napari` | The Qt/napari UI layer that orchestrates the above into the unified workflow widget |

:::{tip}
Each subpackage's own module docstring is the canonical one-paragraph
description of what it does, and it is surfaced at the top of that package's
[API page](../api/index.md). Keeping the high-level description *in the module
docstring* is deliberate — it lives next to the code and is reviewed with it.
:::

## Design records

Point-in-time design documents under `docs/superpowers/specs/` capture *why*
specific features were built the way they were. They are snapshots, not living
documentation — read them as history, and do not expect them to track the
current code.
