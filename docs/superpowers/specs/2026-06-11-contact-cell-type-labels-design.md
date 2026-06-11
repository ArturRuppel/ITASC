# Contact Cell-Type Labels — Design Spec

Date: 2026-06-11

> **Status: design.** Pins the exact data shape before building. A single
> headless, Qt-free label function — no plotting, no aggregation, no UI. Follows
> the `contacts/energetics.py` seam: operate on an already-read
> `PositionContactAnalysis`, return a column-major table, export from the
> contacts package root.

Propagate the **NLS subpopulation labels** (the per-cell `id,label` sidecar CSV)
onto the **contacts** (the `edges` table). Each cell–cell junction is labelled by
the unordered pair of its two cells' NLS labels, so downstream analyses can ask
how the subpopulations contact each other (homotypic vs heterotypic), without any
of that aggregation living here.

## Decisions (locked with the user)

1. **Build/label function only.** No plot mode, no pooling plugin, no readout, no
   per-frame aggregation. Just the function that turns NLS labels + contacts into
   a labelled-contacts table. Counts/fractions/mixing-index are explicitly *out of
   scope* — a later consumer's job.
2. **Generic unordered-pair scheme.** The contact label is
   `"-".join(sorted([label_a, label_b]))` over the **arbitrary** NLS label
   vocabulary — not hard-wired to positive/negative. Two cells of types `A` and
   `B` give `"A-B"`; `A` and `A` give `"A-A"`. Future-proof for >2 cell types.
   A derived boolean `homotypic` (`label_a == label_b`) rides alongside.
3. **Headless, Qt-free, no I/O.** The function takes an already-read
   `PositionContactAnalysis` **and** a `labels: Mapping[int, str]` (`cell_id →
   label`) — it never opens HDF5 or the CSV itself. The caller composes
   `read_nls_classification_csv(...)` (existing,
   `contacts/nls_classification.py`) with this function. Mirrors how
   `signed_central_junction_lengths` operates purely on the read analysis.
4. **Cell–cell edges only.** Border edges (`kind == "border"`, `cell_b == 0`) are
   not contacts between two cells and are dropped. One labelled row per
   `kind == "cell_cell"` edge row.
5. **Fragments stay separate.** Unlike energetics (which sums fragment lengths for
   the landscape), labelling is per-edge-row: a boundary split across segments
   yields several rows that share the same `(frame, pair)` and therefore the same
   label. `edge_id` and `length` are carried through so a consumer can join or
   length-weight later. No fragment join here.
6. **Unclassified is a first-class token.** A cell with no CSV row gets the label
   `unclassified` (configurable via a keyword). It sorts into the pair like any
   other token (`"positive-unclassified"`), so partially-labelled contacts are
   visible rather than silently dropped. A companion `fully_classified` boolean
   marks rows where **both** cells had a label, so a consumer can filter them out
   cheaply. `homotypic` is computed on the raw labels, so two unclassified cells
   read `homotypic == True`; consumers that care should gate on
   `fully_classified`. Documented on the function.

## The function (`contacts/contact_labels.py`)

New module, sibling to `energetics.py`.

```python
def label_contacts(
    analysis: PositionContactAnalysis,
    labels: Mapping[int, str],
    *,
    unclassified: str = "unclassified",
) -> dict[str, np.ndarray]:
    """Label every cell–cell contact by its two cells' NLS subpopulation labels.

    For each ``kind == "cell_cell"`` edge, look up each endpoint's label in
    *labels* (``cell_id -> label``; a missing cell -> *unclassified*) and form the
    unordered-pair contact label ``"-".join(sorted([label_a, label_b]))``.

    Columns (column-major, all equal length, one row per cell–cell edge):

    * ``frame``            — the edge's frame.
    * ``edge_id``          — the edge's id within its frame.
    * ``cell_a``/``cell_b``— the contacting cell ids (as stored, cell_a < cell_b).
    * ``label_a``/``label_b`` — each cell's NLS label (or *unclassified*).
    * ``contact_label``    — sorted "label_a-label_b" pair.
    * ``homotypic``        — ``label_a == label_b`` (True for two unclassified).
    * ``fully_classified`` — both cells had a label in *labels*.
    * ``length``           — the edge length, carried through for weighting.

    Border edges (``kind == "border"``) are excluded. Returns empty (but typed)
    arrays when there are no cell–cell edges.
    """
```

Algorithm:

- Read `edges` columns: `frame`, `edge_id`, `cell_a`, `cell_b`, `kind`, `length`.
- Mask to `kind == "cell_cell"`.
- Per surviving row:
  - `present_a = int(cell_a) in labels`; `label_a = labels.get(cell_a, unclassified)`
    (same for `b`).
  - `contact_label = "-".join(sorted((label_a, label_b)))`.
  - `homotypic = label_a == label_b`; `fully_classified = present_a and present_b`.
- Assemble column-major arrays: string columns (`label_a`, `label_b`,
  `contact_label`) as `dtype=object`; `homotypic`/`fully_classified` as `bool`;
  `frame`/`edge_id`/`cell_a`/`cell_b` as `int64`; `length` as `float`.
- Empty input (or no cell–cell edges) → every column an empty typed array, same
  keys.

Notes:
- Pure Python/NumPy over the read table — no h5py, no csv, no Qt, no pandas.
- The label vocabulary is whatever the CSV holds; the function imposes none.

## Wiring (`contacts/__init__.py`)

Export `label_contacts` alongside `signed_central_junction_lengths`:

```python
from cellflow.aggregate_quantification.contacts.contact_labels import label_contacts
from cellflow.aggregate_quantification.contacts.energetics import (
    signed_central_junction_lengths,
)

__all__ = ["label_contacts", "signed_central_junction_lengths"]
```

Composition the caller writes (illustrative — **not** part of this increment):

```python
labels = read_nls_classification_csv(position_dir / "aggregate_quantification" / "nls_classification.csv")
analysis = read_position_contact_analysis(contact_h5_path)
contacts = label_contacts(analysis, labels)
```

## Files

| File | Change |
|---|---|
| `aggregate_quantification/contacts/contact_labels.py` | **new** — `label_contacts`. |
| `aggregate_quantification/contacts/__init__.py` | export `label_contacts`. |
| `tests/aggregate_quantification/test_contact_labels.py` | **new** — see below. |

## Testing

Hand-built `PositionContactAnalysis` fixtures (just the `edges` dict the function
reads; other tables empty):

- **Homotypic / heterotypic.** Edges `(A,A)`, `(B,B)`, `(A,B)` with a labels map
  → `contact_label` of `"A-A"`, `"B-B"`, `"A-B"`; `homotypic` `True/True/False`;
  pair always sorted (`(B,A)` input still yields `"A-B"`).
- **Generic vocabulary.** Three labels `A/B/C` → `"B-C"` etc., no hard-coded
  positive/negative assumption.
- **Unclassified.** A cell absent from *labels* → its label is `unclassified`,
  `fully_classified == False`; an edge between two absent cells →
  `homotypic == True` but `fully_classified == False`; custom `unclassified=`
  token honoured.
- **Border edges excluded.** A `kind == "border"` row (`cell_b == 0`) produces no
  output row.
- **Fragments stay separate.** Two `cell_cell` rows sharing `(frame, A, B)` with
  different `edge_id`/`length` → two output rows, same label, lengths preserved.
- **Empty / no cell–cell edges.** Empty edges, or border-only edges → every column
  an empty array of the documented dtype.

## Out of scope (deferred)

- Any aggregation: per-frame counts, fractions, total junction length per contact
  type, mixing/segregation index. Downstream consumers build these on the table.
- Plotting, pooling, and napari UI. No plugin in this increment.
- Reading the CSV / locating the sidecar inside the function — the caller composes
  `read_nls_classification_csv`.
- Writing the labels back into the `.h5` — the `.h5` stays a pure regenerable
  build artifact (NLS-sidecar redesign); labels are joined at analysis time.
