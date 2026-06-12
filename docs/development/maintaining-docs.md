# Maintaining the docs

These docs are built so that they **cannot quietly drift** from the code. Three
rules keep them honest as CellFlow evolves.

## 1. The API reference is generated, never written

The entire [API Reference](../api/index.md) is produced by Sphinx
`autosummary`/`autodoc` from the package's own docstrings at build time. There
is no hand-maintained list of functions or signatures to fall out of date —
rename a function and the docs rename with it.

**What this asks of you:** write the docstring in the same pull request as the
code. That is the only upkeep the reference needs.

## 2. The narrative stays high-altitude

Hand-written pages (the User Manual, the [architecture overview](architecture.md))
describe *concepts that change slowly* — which distribution to install, the
staged workflow, the dependency graph. The moment a prose page restates a
function's arguments, it has become a drift trap; link into the generated API
instead.

The per-subpackage description lives in each subpackage's **module docstring**,
so the high-level map sits next to the code and is reviewed with it. Diagrams use
[Mermaid](https://mermaid.js.org/) (text, diffable, in-repo) rather than images.

## 3. CI makes drift a failing check

The documentation workflow builds with **warnings treated as errors**
(`sphinx-build -W`). A cross-reference to a symbol that was renamed or deleted,
a link to a page that no longer exists, or a malformed docstring fails the build
at pull-request time instead of rotting silently. The baseline build is
warning-clean, so any new warning is something you introduced.

:::{tip}
To tighten further, set `nitpicky = True` in `conf.py` (flags every unresolved
cross-reference, including in type hints) and add
[`interrogate`](https://interrogate.readthedocs.io/) to enforce docstring
coverage on new public APIs.
:::

## Building locally

```bash
python -m pip install -e .[docs]
sphinx-build -b html docs docs/_build/html
# open docs/_build/html/index.html
```
