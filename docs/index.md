# CellFlow

CellFlow is a [napari](https://napari.org)-based research toolkit for time-lapse
cell microscopy. It connects Cellpose-derived probability and flow outputs,
Ultrack-based nucleus tracking, interactive correction, cell-label propagation,
validation-aware resolving, and downstream contact analysis into one workflow —
and factors that workflow into independently installable pieces that share the
`cellflow.*` namespace, so you can install only the part you need.

:::{admonition} Status
:class: note
CellFlow is under active research development. The workflow, installation
details, and public API should be treated as provisional until the public
release and accompanying manuscript stabilize.
:::

## Two ways in

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 📖 User Manual
:link: manual/index
:link-type: doc

Start here. Which distribution is for which kind of user, the staged
position-directory workflow, and how to install and drive the napari plugin.
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
