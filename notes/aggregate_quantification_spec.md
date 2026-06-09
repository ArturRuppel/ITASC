# Aggregate Quantification — Design Spec

Date: 2026-06-09

Rename + redesign of the "Contact Analysis" study. Supersedes the contact-only
framing: contacts become *one quantifier among many*. This document covers the
**framework-seam-only** work unit (see Scope). New quantities are explicitly
deferred — they drop in as modules once the seam exists.

## Decisions (locked)

1. **Scope — framework seam only.** Build the quantifier registry, port the
   existing contact pipeline to be the single proven `Quantifier`, rename the
   package/UI/manifest. No new quantities (nucleus tracks, centroid offset, cell
   shape, tissue dynamics) in this unit — they are the payoff the seam enables,
   added later.
2. **Storage — each quantifier owns its persistence.** There is no unified
   per-position artifact and no framework-imposed schema. Source files stay as
   they are: TIFF for labels, and the contacts quantifier keeps producing
   `contact_analysis.h5` with today's exact schema. A future quantifier picks
   whatever format suits it (CSV, its own `.h5`, Parquet, …).
3. **Rename — full, including the distribution.** Python package, classes, UI
   strings, napari manifest IDs, *and* the standalone wheel
   (`cellflow-contact` → `cellflow-aggregate`) all move to "Aggregate
   Quantification". The on-disk `contact_analysis.h5` name is *unchanged*
   because that is the contacts quantifier's own storage choice, not a
   study-level name.
4. **No CellFlowUtils back-compat.** The external `cellflow_utils` NLS classifier
   is *superseded* by the classifier in this redesigned distro, so no shim or
   alias is owed to it. The old `cellflow.contact_analysis` import path is
   dropped outright (it had no other external consumer — verified).

## The core idea

Today the pipeline is welded to contacts end to end: `build.py` extracts
`cells`/`edges`/`t1_events` into a fixed HDF5 schema, `reader.py` returns a fixed
`PositionContactAnalysis`, the napari visualizer knows about Edges/T1 layers, and
the studio drives all of it.

The cross-position layer (`napari/meta_plugins/`) is *already* a plugin registry
— subclass `MetaAnalysisPlugin`, auto-register via `__init_subclass__`,
discover via `available_meta_plugins()`. The redesign applies that same pattern
one stage upstream, to the **compute** layer:

```
sources (tif labels, tracks db)
        │
        ▼
  [ Quantifier registry ]   ← NEW seam.  contacts is one plugin here.
        │   each quantifier owns: required inputs, build, persistence, read
        ▼
  per-position artifacts     ← contacts → contact_analysis.h5 (unchanged)
        │
        ▼
  [ MetaAnalysisPlugin ]     ← EXISTS. cross-position aggregation / plotting
        │
        ▼
      plots / CSV
```

"Aggregate Quantification" studio = catalog of positions × enabled quantifiers →
builds each position's artifacts → feeds the meta-plugins that plot.

## The Quantifier contract

Backend-only — **no Qt / napari import** so the standalone wheel and headless
batch runs keep working. Mirrors the meta-plugin registry mechanics.

```python
# aggregate_quantification/quantifier.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar

@dataclass(frozen=True)
class PositionInputs:
    """Resolved source files for one position. Quantifiers read what they need.

    Every field has a live consumer. A future track-based quantifier adds its
    own field (e.g. tracks_db_path) in the same commit that first reads it —
    no speculative placeholders here."""
    position_dir: Path
    cell_labels_path: Path | None = None
    nucleus_labels_path: Path | None = None

_REGISTRY: dict[str, type[Quantifier]] = {}

class Quantifier:
    """A per-position compute unit. Subclassing with a non-empty quantity_id
    registers it; the studio discovers it via available_quantifiers()."""

    quantity_id: ClassVar[str] = ""        # stable key, e.g. "contacts"
    display_name: ClassVar[str] = ""       # e.g. "Cell–cell contacts"
    requires: ClassVar[tuple[str, ...]] = ()   # PositionInputs field names

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        if cls.quantity_id:
            _REGISTRY[cls.quantity_id] = cls

    def can_build(self, inputs: PositionInputs) -> bool:
        return all(getattr(inputs, name, None) is not None for name in self.requires)

    def output_path(self, inputs: PositionInputs) -> Path:        # contacts → .../contact_analysis.h5
        raise NotImplementedError
    def is_built(self, inputs: PositionInputs) -> bool:
        return self.output_path(inputs).is_file()
    def build(self, inputs: PositionInputs, *, params: dict | None = None,
              progress_cb: Callable[[int, int, str], None] | None = None) -> Path:
        raise NotImplementedError
    def read(self, inputs: PositionInputs) -> Any:                # contacts → PositionContactAnalysis
        raise NotImplementedError

def available_quantifiers() -> list[type[Quantifier]]:
    _import_quantifier_modules()        # pkgutil sweep of quantifiers/ — same as meta_plugins
    return sorted(_REGISTRY.values(), key=lambda c: c.display_name.lower())
```

Notes:
- `read()` returns whatever object that quantifier defines; for contacts it stays
  `PositionContactAnalysis`. No common return type is imposed — the meta-plugins
  already type-narrow per quantity.
- Visualization stays on the napari side, keyed by `quantity_id` (the backend
  contract has no napari dependency). For this unit there is exactly one
  visualizer (contacts), so the registry can be a simple `quantity_id ->
  add_layers_fn` map; generalize only when a second visualizer exists.

## Package layout after rename

```
src/cellflow/aggregate_quantification/
    __init__.py            # re-exports Quantifier, available_quantifiers, PositionInputs,
                           #   and the contacts public API (read_position_contact_analysis, …)
    quantifier.py          # the contract + registry (backend, no Qt)
    quantifiers/
        __init__.py        # pkgutil self-registration sweep (like meta_plugins)
        contacts.py        # ContactsQuantifier(Quantifier): thin adapter over contacts/*
    contacts/              # moved verbatim from today's contact_analysis/
        build.py
        reader.py          # PositionContactAnalysis, read_position_contact_analysis
        nls_classification.py
        batch.py
    napari.yaml            # standalone manifest (renamed ids/titles)

src/cellflow/napari/
    aggregate_quantification_studio.py        # was contact_analysis_studio.py
    aggregate_quantification_widget.py        # was contact_analysis_widget.py
    contact_visualization.py                  # was contact_analysis_visualization.py (stays contacts-specific)
```

`ContactsQuantifier` is the only new logic — a thin adapter:

```python
# aggregate_quantification/quantifiers/contacts.py
class ContactsQuantifier(Quantifier):
    quantity_id = "contacts"
    display_name = "Cell–cell contacts"
    requires = ("cell_labels_path",)        # nucleus optional (enables NLS / ID validation)

    def output_path(self, inputs):
        return inputs.position_dir / "contact_analysis.h5"
    def build(self, inputs, *, params=None, progress_cb=None):
        return build_contact_analysis(
            cell_labels_path=inputs.cell_labels_path,
            nucleus_labels_path=inputs.nucleus_labels_path,
            output_path=self.output_path(inputs),
            source_path=inputs.position_dir,
            edge_extraction_params=params,
            progress_cb=progress_cb,
        )
    def read(self, inputs):
        return read_position_contact_analysis(self.output_path(inputs))
```

## Rename map

| From | To |
|---|---|
| `cellflow/contact_analysis/` (pkg) | `cellflow/aggregate_quantification/` |
| `…/contact_analysis/{build,reader,nls_classification,batch}.py` | `…/aggregate_quantification/contacts/…` (moved verbatim) |
| `napari/contact_analysis_studio.py` → `ContactAnalysisStudioWidget` | `aggregate_quantification_studio.py` → `AggregateQuantificationStudioWidget` |
| `napari/contact_analysis_widget.py` → `ContactAnalysisWidget`, `make_contact_analysis_widget` | `aggregate_quantification_widget.py` → `AggregateQuantificationWidget`, `make_aggregate_quantification_widget` |
| `napari/contact_analysis_visualization.py` | `napari/contact_visualization.py` (kept contacts-specific) |
| manifest cmd `cellflow.contact_analysis_widget`, title "Contact Analysis" | `cellflow.aggregate_quantification_widget`, "Aggregate Quantification" |
| standalone dist `cellflow-contact`, manifest `cellflow-contact` / `cellflow-contact.widget` | `cellflow-aggregate`, manifest `cellflow-aggregate` / `cellflow-aggregate.widget` |
| `packages/cellflow-contact/` | `packages/cellflow-aggregate/` (force-include paths repointed to the new dirs) |
| `contact_analysis.h5` (on disk) | **unchanged** — contacts quantifier's storage |
| CSV catalog columns / `meta/catalog.py` | **unchanged** — `path` still points at `contact_analysis.h5` |

## Back-compat

- **No Python shim.** The old `cellflow.contact_analysis` import path is removed,
  not aliased. Its only external would-be consumer (`cellflow_utils`' NLS
  classifier) is superseded by this distro's classifier, and nothing else imports
  it — verified by grep across the repo and `CellFlowUtils/`.
- **On-disk files unchanged**, so existing catalogs and `.h5` files keep working
  with zero migration regardless of the code/dist rename.
- *(Optional)* a hidden `cellflow.contact_analysis_widget` manifest alias for one
  release, only if saved napari layouts referencing the old command id matter —
  drop if not.

## UI changes

For this unit the studio is functionally identical — it just gains a quantifier
selector that currently has one entry ("Cell–cell contacts"), proving the seam.

- Studio renamed to **Aggregate Quantification**; section header / status strings
  updated.
- Catalog, per-position embedded view, and meta-plugin host are unchanged in
  behavior. The embedded per-position view is now "the selected quantity's view"
  (contacts today).
- "Add" / build path routes through `ContactsQuantifier.build` instead of calling
  `run_contact_batch` directly, so adding a second quantifier later needs no
  studio change.

## Test plan

- **Move-only churn:** existing `tests/contact_analysis/*` and
  `tests/napari/test_contact_analysis_*` move/rename with the code; assertions on
  layer names / behavior stay green (416 currently passing).
- **New — registry:** `test_available_quantifiers_discovers_contacts`,
  `test_quantifier_self_registration`, `test_can_build_respects_requires`.
- **New — adapter:** `ContactsQuantifier.output_path` ends in
  `contact_analysis.h5`; `build` produces a byte-identical-schema file to the old
  `build_contact_analysis` (golden compare); `read` returns a
  `PositionContactAnalysis`.
- Run: `QT_QPA_PLATFORM=offscreen uv run --quiet pytest tests/`.

## Implementation order

1. `git mv` the package + napari modules **and** `packages/cellflow-contact/` →
   `packages/cellflow-aggregate/` (no logic change); fix imports; repoint the
   dist's force-include paths, name (`cellflow-aggregate`), and manifest
   entry-point; update root `napari.yaml`. Get the full suite green on pure rename.
2. Add `quantifier.py` (contract + registry) and `quantifiers/contacts.py` adapter.
3. Route the studio's build/read through the registry (one quantifier).
4. (Optional) hidden manifest command-id alias — only if saved layouts matter.
5. New tests (registry, adapter).
6. Update `TODO.md` (check off the three Aggregate Quantification bullets that
   this unit completes; leave "broaden scope" as the follow-on).

> **Status:** Part 1 (the seam) is implemented — commits `c9e41e1` (rename) and
> `350f001` (registry + adapter + studio routing). The UI redesign below is Part 2,
> not yet built.

---

# Part 2 — UI redesign

## Why

The studio today is a flat stack of four always-expanded collapsibles (Discover &
add · Catalogue · Contact view · Analysis). Setup, the table, a single-position
viewer, and cross-position analysis all compete for height, and the single
"Analysis" dropdown silently mixes two different operating scopes: `nls_classification`
is per-position (acts on one row, writes back to its `.h5`) while `catalog_summary`
is cross-position — distinguished only by how many rows happen to be selected.

## Shape: two regions

Everything collapses to **Catalogue** + **Plugins**.

```
┌ Aggregate Quantification ─────────────────────────┐
│ ▼ Catalogue                                        │
│   Root:[_____][Browse]  Cell:[__] Nuc:[__] …[Discover]
│   [ discovered list → Add ]                        │
│   ┌ positions ─────────────────────────────────┐  │
│   │ cond date  id   inputs        status        │  │
│   │ WT  06-01 p01   cell·nuc·trk  contacts ✓    │  │
│   │ WT  06-01 p02   nuc           contacts —    │  │
│   └─────────────────────────────────────────────┘  │
│   [Load CSV][Save CSV][Remove][Clear]              │
│   ┌ visualize (selected position) ──────────────┐  │
│   │  ☑labels ☑tracks ☑edges …   [Load layers]    │  │
│   └─────────────────────────────────────────────┘  │
│ ───────────────────────────────────────────────── │
│ Plugins   ☑ Contacts (build)  ☑ NLS  ☐ Summary    │
│ ▼ Contacts (build)          ← mounted collapsible  │
│      [Build selected]  status…                     │
│ ▼ NLS classification                               │
│      …plugin UI…                                    │
└────────────────────────────────────────────────────┘
```

### Region 1 — Catalogue (one merged interface)

Merges today's *Discover & add* + *Catalogue* + *Contact view*:

- **Assemble**: discover from a tree / add loose positions / annotate
  (condition·date·id·notes) / load·save CSV / remove·clear — one block, not two.
- **Table** of positions. The catalogue **no longer builds** anything; **Add is
  register-only** (a row may show `contacts —` / missing). An `inputs` column shows
  which sources a position actually has.
- **Visualizer**, kept as-is: modifier checkboxes + a **Load layers** button. It
  loads the position's whole picture and **degrades gracefully** — whatever exists
  among Cell labels, Nucleus labels, Nucleus tracks, Edges, T1 edges; absent inputs
  are skipped, never errored.

### Region 2 — Plugins (flat checkbox registry → collapsibles)

- A **flat list of all registered plugins, each a checkbox**. Checking mounts the
  plugin as its **own collapsible**; unchecking removes it. **Multiple active at
  once** (replaces the single dropdown + host).
- Three **roles**, all in the one flat list, each reading the catalogue's current
  selection (selected rows = scope; empty selection = whole catalogue):
  - **Builder** — a Qt front over a `Quantifier`: a collapsible with a **Build**
    button that computes the quantity for the in-scope positions. *Contacts is the
    first.* **One builder = one quantity**, so the quantity axis needs no separate
    selector — the plugin list *is* the quantity menu.
  - **Per-position processor** — e.g. NLS classification; acts on the one selected
    position and writes back.
  - **Cross-position aggregator** — e.g. catalogue summary; pools the cohort → table/plot.

## All inputs optional

A position is **a folder with ≥1 recognized input** — cell labels, nucleus labels,
tracks are each independently optional (no required anchor). Consequences:

- **Discovery** (`meta.catalog.discover_catalog_entries`) is refactored from
  "cell-labels-anchored" to "union of recognized inputs": register a folder as a
  position when it contains any recognized input; collect whatever is present.
- **Every consumer declares its needs and degrades.** A plugin declares required
  inputs via `requires` (possibly a single one, e.g. `("nucleus_labels_path",)`)
  and the UI **greys its checkbox / Build** for a selection lacking them
  (generalises `Quantifier.can_build`). The visualizer has no hard requirement.
- Backend already fits: `PositionInputs` fields are all `Optional`/`None`-default;
  `requires` + `can_build` already express single-input needs.

## Backend deltas (small)

- **Unified UI-plugin layer.** One flat registry of collapsible plugin widgets with
  a `role` (builder | processor | aggregator), a `display_name`, and `requires`.
  Builder plugins wrap a `Quantifier` (backend untouched); fold the existing
  `MetaAnalysisPlugin`s (NLS, summary) into this list. The checkbox list renders
  every registered UI plugin, enabled per the selection's available inputs.
- **Discovery refactor** to optional/union inputs (above); catalogue table gains an
  `inputs` column.
- **Studio**: drop the *Analysis* dropdown + the auto-build-on-Add path; Add becomes
  register-only; building moves into the Contacts builder plugin (scope = selected
  rows, else whole catalogue — the existing scope rule).

## Test plan (Part 2)

- Discovery registers positions from any single input; from a mix; from none → skip.
- Catalogue `inputs`/status columns reflect present sources + per-quantity built state.
- Plugin registry: checkbox mounts/unmounts a collapsible; multiple coexist;
  a plugin greys out when the selection lacks its `requires`.
- Builder plugin builds in-scope positions and updates status; Add no longer builds.
- Visualizer loads only the layers whose inputs exist (no error on missing).

## Out of scope / follow-ons

- New quantifiers (nucleus track kinematics, nucleus↔cell centroid offset, cell
  shape, tissue dynamics) — each a `quantifiers/*.py` builder + its overlay/plugin.
  The seam (Part 1) + the flat plugin list (Part 2) make them additive.
- Per-quantity visualization beyond the bundled contacts overlay (each quantity's
  plugin contributing its own overlay layers).
- Dimensionality support (separate TODO track).
