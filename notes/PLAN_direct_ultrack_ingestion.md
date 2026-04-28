# Plan: Direct Ingestion of v2 Hypotheses into Ultrack's NodeDB

**Status:** proposed, not started
**Author context:** synthesized 2026-04-27 after a code review of v1 archive + v2 current
**Audience:** a fresh agent who has not seen the prior conversation

This plan replaces v2's per-cell greedy/Viterbi propagator with a global ILP-based
tracker by feeding our existing hypothesis sweep directly into Ultrack's
candidate-node database — bypassing Ultrack's `segment()` step. The point is to
keep all the quality control we've built in v2's hypothesis layer while gaining
the joint optimisation that v1 had.

---

## 1. Project background

CellFlow is a cell segmentation + tracking pipeline. There are two generations:

- **v1 (archived under `archive/v1/`)** — used [Ultrack](https://github.com/royerlab/ultrack)
  as the tracking solver. Generated a parameter-swept watershed sweep, *averaged*
  the resulting labelmaps into a single foreground + contour map, fed those to
  `ultrack.segment()`, then ran linking + ILP solve. Quality was good but
  required heavy manual correction post-hoc.
- **v2 (current `src/cellflow/`)** — replaced Ultrack with a custom propagator.
  Goal was to integrate manual correction *into* the optimisation rather than
  always running it after the fact. Hypothesis generation in v2 is genuinely
  better than v1 (more methods: cellpose-flow, contour-watershed, seeded-
  watershed; plus a clean parameter sweep API). But the tracker is per-cell and
  per-frame greedy, which is structurally weaker than v1's joint ILP. The team
  has tried multiple cost functions (anchor-LAP, IoU composite, centroid+size
  composite, predicted-centroid composite, global LAP, per-cell Viterbi) and
  none match v1 quality. See `scripts/test_anchor_lap_propagator.py` for the
  current benchmark of all six strategies.

**Diagnosis:** The gap is architectural, not parametric. Ultrack does a global
ILP that jointly chooses (a) one non-overlapping subset of candidate nodes per
frame and (b) inter-frame links with appear/disappear/division terms, under
overlap constraints. v2 has no overlap constraint, no joint frame optimisation,
no division handling. No cost-function tweak can close that gap.

**This plan's bet:** keep v2's hypothesis generation (and its UI / DB
visualizer), but write hypotheses *directly* into Ultrack's NodeDB schema
instead of going through the lossy `segment()` path. This is cheaper than v1's
fg+contour route AND preserves more information. If it works, it can later be
extended with custom linking costs (option 3 in the design discussion) or with
seed-aware solving (already prototyped in `archive/v1/.../seeded_tracker.py`).

---

## 2. Key facts about the codebases

### 2.1 v2 hypothesis storage (current)

- **File:** `src/cellflow/database/hypotheses.py`
- **HDF5 schema:** `hypotheses/t{t:03d}/p{p:03d}/labels`, dataset shape `(Z, Y, X)`, dtype `uint32`. Layout version 2.
- **Read API:**
  - `list_hypotheses(path) -> (n_p, params_by_p_dict)` — first-timepoint introspection.
  - `read_hypothesis_labels(path, t, p) -> np.ndarray (Z, Y, X)` — single (t, p) read.
  - `iter_hypothesis_records(path)` — yields `HypothesisRecord(t, p, labels, params)` in sorted order.
- **Existing data is mostly 2D:** Z=1 in current pipelines. Code handles 2D and 3D.
- **Hypothesis methods:** `seeded_watershed`, `cellpose_flow`, `contour_watershed`, plain `watershed`. Each method has its own params dataclass; the writer dedupes by `parameter_json`.
- **Workflow widget that drives this:** `src/cellflow/napari/cell_workflow_widget.py`. Currently calls `propagate_one_frame` from `src/cellflow/tracking/propagator.py` for tracking.

### 2.2 v2 propagator (to be replaced for tracking)

- **`src/cellflow/tracking/propagator.py`** — distance-gated greedy per-cell match using shape/area/IoU score. 257 lines.
- **`src/cellflow/tracking/propagator_v2.py`** — anchor-LAP and composite variants (276 lines).
- These won't be removed yet; the new path lives alongside until validated.

### 2.3 v1 archive (reference for what worked)

- **`archive/v1/packages/ultrack/src/cellflow/ultrack/`** — the v1 Ultrack integration:
  - `ingestion.py` — `labels_batch_to_foreground_contours()` averages a bag of labelmaps into fg+contour maps. **This is the lossy path we are replacing.**
  - `linking.py` — custom IoU-aware linking (`run_iou_linking`). Reuses Ultrack's NodeDB; this code can be lifted forward.
  - `pruning.py` — circularity-based candidate pruning post-segment. Also reusable.
  - `stages/tracking.py` — top-level orchestrator (`run_segmentation`, `run_linking`, `run_solve`). Shows the canonical Ultrack flow and provides `_build_ultrack_config()`.
  - `stages/seeded_tracker.py` — manual-correction-aware solver that uses a seed labelmap to constrain the ILP. **676 lines, this is the "manual correction in the optimisation" piece — v1 already had it.**
  - `config.py` — `TrackingConfig` (Pydantic): one place for all Ultrack knobs. Reusable as-is for downstream stages.

### 2.4 Ultrack internals (verified by reading installed package)

Ultrack is installed in the `cellflow` conda env at
`/home/aruppel/miniconda3/envs/cellflow/lib/python3.10/site-packages/ultrack/`.

**NodeDB schema** (from `ultrack/core/database.py:77`):

```python
class NodeDB(Base):
    __tablename__ = "nodes"
    t = Column(Integer, primary_key=True)
    id = Column(BigInteger, primary_key=True, unique=True)
    parent_id = Column(BigInteger, default=NO_PARENT)        # set by solver (track parent)
    hier_parent_id = Column(BigInteger, default=NO_PARENT)   # set during segmentation
    t_node_id = Column(Integer)                              # 1-based per-time index
    t_hier_id = Column(Integer)                              # which hierarchy this came from
    z = Column(Float)
    y = Column(Float)
    x = Column(Float)
    z_shift = Column(Float, default=0.0)
    y_shift = Column(Float, default=0.0)
    x_shift = Column(Float, default=0.0)
    area = Column(Integer)
    frontier = Column(Float, default=-1.0)   # saliency-hierarchy thing; ok to leave default
    height = Column(Float, default=-1.0)     # persistence; ok to leave default
    selected = Column(Boolean, default=False)  # set by solver
    pickle = Column(MaybePickleType)         # the Node object (mask + bbox + centroid)
    features = Column(MaybePickleType, default=None)
    # ...annotation enum columns, all default to UNKNOWN
```

**OverlapDB schema** (`ultrack/core/database.py:106`):

```python
class OverlapDB(Base):
    __tablename__ = "overlaps"
    id = Column(Integer, primary_key=True, autoincrement=True)
    node_id = Column(BigInteger, ForeignKey("nodes.id"))
    ancestor_id = Column(BigInteger, ForeignKey("nodes.id"))
```

Just pairs. There's no overlap-direction or "is-subset" semantics; the ILP
treats them symmetrically as "cannot both be selected".

**LinkDB schema** (set by linking; we don't write this directly):

```python
class LinkDB(Base):
    source_id = Column(BigInteger, ForeignKey("nodes.id"))
    target_id = Column(BigInteger, ForeignKey("nodes.id"))
    weight = Column(Float)
```

**Node class** (`ultrack/core/segmentation/node.py:111`):

- Static factory: `Node.from_mask(time, mask, bbox=None, node_id=-1)`. Pass a
  *cropped* boolean mask plus a bbox; or pass a full-frame mask and let it
  compute the bbox via `scipy.ndimage.find_objects`.
- bbox format is **`(min_0, min_1, max_0, max_1)`** for 2D (i.e. all mins then
  all maxes — not `(start, end)` per axis). For 3D: `(min_0, min_1, min_2, max_0, max_1, max_2)`.
- `mask` *must* be `dtype=bool`.
- `Node` has `__getstate__/__setstate__` that pack the mask via `blosc2`, so
  pickling is cheap. The `pickle` column on NodeDB stores `pickle.dumps(node)`.
- Centroid is computed automatically as `int(round(...))`.

**ID generation** (`ultrack/core/segmentation/processing.py:42`):

```python
def _generate_id(index, time, max_segments=1_000_000):
    return index + (time + 1) * max_segments
```

Use this exact scheme to match what the rest of Ultrack expects: the global ID
is `index + (time+1)*max_segments_per_time`, with `index` a 1-based
within-time counter. This must match across our writes and Ultrack's reads.

**Reference write path** (`ultrack/core/segmentation/processing.py:_process`,
lines 161–339): the canonical `segment()` populates NodeDB+OverlapDB exactly
this way. Read it once before implementing — it tells you which fields
matter and which don't.

**What the solver actually reads** (verified by grepping `ultrack/core/solve/`):
- `sqltracking.py` only joins on `OverlapDB` and `NodeDB`. It does **not**
  require `hier_parent_id` to be set — that's used for downstream feature
  analysis (`get_nodes_features` with `include_persistence=True`), not the
  tracking ILP. So the **minimum-viable plan can leave `hier_parent_id` as the
  default `NO_PARENT`.**

### 2.5 The crucial insight

A v2 parameter-sweep at fixed `t` produces multiple flat partitions
(non-overlapping within each `p`, but overlapping across `p`). Ultrack's
`segment()` gets its *tree* by running watershed at multiple thresholds on a
single fg+contour map. **We don't need a tree** — the ILP only requires
`OverlapDB` pairs to enforce non-overlap. Each (t, p, label_id) becomes one
NodeDB row; for every pair of cells at the same `t` (across all `p`) whose
masks intersect, write one OverlapDB row.

If we later want hierarchy-aware ILP costs, we can derive `hier_parent_id` by
computing strict containment (`mask(B) ⊆ mask(A)`), but it's a Phase-2 nice-to-
have, not blocking.

---

## 3. Architecture of the new module

### 3.1 New package: `src/cellflow/tracking_ultrack/`

```
src/cellflow/tracking_ultrack/
├── __init__.py
├── config.py              # TrackingConfig (lifted/adapted from v1 archive)
├── ingest.py              # NEW: hypothesis HDF5 → NodeDB + OverlapDB
├── linking.py             # lifted from archive/v1/.../ultrack/linking.py
├── solve.py               # thin wrapper around ultrack.core.solve
├── export.py              # NodeDB.selected → (T, Z, Y, X) tracked labels
└── seeded_solve.py        # later: lifted/adapted from archive seeded_tracker.py
```

Keep it as a sibling of the existing `tracking/` package so we can A/B test.
Once validated, the old propagator can be deprecated.

### 3.2 The ingestion contract (`ingest.py`)

**Input:** Path to a v2 `hypotheses.h5` + a `TrackingConfig` + a working dir.

**Output side effects:**
- An empty Ultrack DB at `<working_dir>/data.db` (SQLite by default — Ultrack's
  `MainConfig` controls this).
- One `NodeDB` row per (t, p, label_id) where `label_id != 0`.
- One `OverlapDB` row per pair of overlapping nodes at the same `t`.

**Function signature:**

```python
def ingest_hypotheses_to_db(
    hypotheses_h5: Path,
    working_dir: Path,
    cfg: TrackingConfig,
    *,
    overwrite: bool = True,
    n_workers: int = 1,
    min_area: int | None = None,    # filter tiny labels before insert
    max_area: int | None = None,
) -> None:
    ...
```

**Critical implementation notes:**

1. **Per-frame batch insert.** Process one `t` at a time. Within `t`, load all
   `p` partitions, then build NodeDB rows + OverlapDB rows, then commit.
   Memory: only one frame's worth of masks in RAM at a time.

2. **ID assignment.** Use exactly the same `_generate_id(index, t, max_segments_per_time)`
   formula as Ultrack's `_process`. `max_segments_per_time` defaults to
   1_000_000 — pass it through `cfg` if you want to tune. Initialize `index = 1`
   per timepoint, increment for each (p, label_id) pair we keep.

3. **Building the Node object.** For each non-zero label in a labelmap:
   - Use `scipy.ndimage.find_objects(labelmap == label_id)[0]` to get the
     bbox slice. Convert to Ultrack's bbox format `(min_0, min_1, max_0, max_1)`.
   - Crop the boolean mask: `mask_crop = (labelmap[bbox_slice] == label_id)`.
   - Call `Node.from_mask(time=t, mask=mask_crop, bbox=bbox_arr, node_id=node_id)`.
   - Pickle: `pickle.dumps(node)`.
   - Compose NodeDB row: `t`, `id`, `t_node_id=index`, `t_hier_id=p+1`,
     `z=int(centroid[0]) if 3D else 0`, `y`, `x`, `area=int(mask_crop.sum())`,
     `pickle=pickled_bytes`. Leave `hier_parent_id`, `frontier`, `height` at
     defaults.
   - **Tip:** Use `skimage.measure.regionprops` once per labelmap to get all
     bboxes/areas/centroids in one pass instead of per-label.

4. **Overlap detection across `p` at fixed `t`.** Naive O(N²) pairwise mask
   intersection is too slow at scale. Use a two-stage filter:
   - **Stage 1 (bbox prune):** build an axis-aligned interval index on bboxes
     per axis, or just use Ultrack's own `intersects()` function (numba-jit'd,
     in `ultrack/core/segmentation/utils.py:24`) on bbox pairs.
   - **Stage 2 (mask intersection):** only for bbox-overlapping pairs, compute
     actual mask intersection (use `iou_with_bbox_2d`/`iou_with_bbox_3d` from
     `ultrack/core/segmentation/utils.py`, or just check `(crop1 &
     crop2_aligned).any()`).
   - **Within a single `p`:** cells in the same labelmap can't overlap by
     construction (it's a partition), so skip self-`p` comparisons.
   - **Symmetric writes:** Ultrack's `_process` writes `(node_id, ancestor_id)`
     for every ancestor in the hierarchy. Since we have no tree direction,
     pick a canonical order (e.g., always smaller-id as `ancestor_id`) and
     write each pair once. Verify by reading the solver code that ordering
     doesn't matter — if it does, write both directions.

5. **Filtering.** Apply `min_area`/`max_area` before NodeDB insert. Even
   though our hypothesis generator already does this, ultrack's `TrackingConfig`
   wants these knobs available so users can tighten further.

6. **Optional features.** If `cfg` requests intensity features (e.g. for
   downstream analysis), accept an `image: np.ndarray | None` parameter and
   compute via `regionprops` then store in `NodeDB.features`. Defer to Phase 2.

7. **Schema setup / overwrite.** Mirror what `segment()` does:
   ```python
   from ultrack.core.database import Base, clear_all_data
   ultrack_cfg = build_ultrack_config(cfg, working_dir)
   if overwrite:
       clear_all_data(ultrack_cfg.data_config.database_path)
   engine = sqla.create_engine(ultrack_cfg.data_config.database_path)
   Base.metadata.create_all(engine)
   ```

### 3.3 Linking and solve

After ingestion, the rest of the Ultrack pipeline runs unchanged:

- **Linking:** lift `archive/v1/.../ultrack/linking.py` verbatim into
  `tracking_ultrack/linking.py`. It already has both default and IoU-aware modes,
  driven by `cfg.linking_mode`.
- **Solve:** thin wrapper around `ultrack.core.solve.processing.track`. The
  archive's `_build_ultrack_config()` is a good starting point.
- **Export:** Ultrack writes the chosen track IDs to `NodeDB.selected` and
  `parent_id`. To produce a `(T, Z, Y, X)` labelmap, iterate selected nodes
  and call `node.paint_buffer(out, value=track_id)`. v1 had this in
  `archive/v1/.../ultrack/stages/tracking.py:export_tracked_labels` — port
  with care: it falls back through several Ultrack export APIs depending on
  what's available.

---

## 4. Implementation phases

### Phase 0 — Scaffolding + verification (half day)

- Create `src/cellflow/tracking_ultrack/` package with empty modules.
- Lift `TrackingConfig` from `archive/v1/.../ultrack/config.py`.
- Write `build_ultrack_config(cfg, working_dir) -> ultrack.config.MainConfig`
  (port from `archive/v1/.../ultrack/stages/tracking.py:53`).
- Write a tiny script `scripts/probe_ultrack_db.py` that:
  1. Builds a synthetic 2-frame (Y, X) labelmap with 2 cells each.
  2. Calls `ingest_hypotheses_to_db()` (stub) on a single hypothesis.
  3. Calls `ultrack.core.linking.processing.link()`.
  4. Calls `ultrack.core.solve.processing.track()`.
  5. Reads back NodeDB / LinkDB and prints contents.
- This confirms the ID scheme, mask format, and pickled Node round-trip
  *before* writing real ingestion. Skip this at your peril.

**Acceptance:** the probe script writes a node, a link, runs the solver,
selects one node per frame, and prints sensible output.

### Phase 1 — Core ingestion (1–2 days)

- Implement `ingest_hypotheses_to_db()` end-to-end for 2D hypotheses.
- Use `regionprops` for per-labelmap stats; build Node objects via
  `Node.from_mask`.
- Overlap detection: bbox prune → mask intersect; within-`p` skip.
- Test-data shape: take the existing benchmark dataset
  (`/home/aruppel/Data/2026-04-01_U251.../v2/pos00/2_nucleus/hypotheses.h5`)
  and ingest its first 5 frames.
- Add unit tests under `tests/`:
  - Ingesting one labelmap with N non-overlapping cells produces N NodeDB
    rows and zero OverlapDB rows.
  - Ingesting two labelmaps where every cell in `p=1` is contained in a cell
    of `p=0` produces 2N NodeDB rows and N OverlapDB rows.
  - Pickled Node round-trips: read back, check `mask.shape`, `bbox`,
    `centroid`.

**Acceptance:** for the real test dataset, ingest succeeds and the per-frame
NodeDB row count equals `sum_p (n_cells_in_partition_p)`.

### Phase 2 — End-to-end tracking on real data (1 day)

- Wire ingest → linking (use default Ultrack first, then IoU-aware) → solve →
  export.
- Run on the same dataset that v2's benchmark uses.
- Compare against `tracked_labels.tif` from the v2 propagator pipeline:
  - Total cell count per frame.
  - Track length distribution.
  - Visual diff in napari (overlay v2-tracked vs new-tracked).
- Compare against v1 if any v1 outputs are still on disk (or re-run v1 on the
  same data via the archived stage).

**Acceptance:** new pipeline produces a `tracked_labels.tif` that visually
matches or exceeds v1 on the test dataset; cell count per frame is within
±5% of expected.

### Phase 3 — Napari widget integration (half day)

- Edit `src/cellflow/napari/cell_workflow_widget.py:_on_propagate_next` and
  `_on_propagate_all` (lines 902 and 945) to optionally call the new pipeline
  instead of `propagate_one_frame`. Add a UI toggle "Tracker: v2 propagator /
  Ultrack ILP".
- Verify the existing manual-correction round-trip still works: correct a
  frame, re-run, check that corrected IDs are preserved (this is currently
  baked into v2's propagator via `validated_history`; the Ultrack path needs
  to honour the same contract).

**Acceptance:** workflow widget runs the Ultrack tracker on a real position
and the user can correct + re-run.

### Phase 4 — Seeded solving (deferred, 2–3 days)

- Port `archive/v1/.../ultrack/stages/seeded_tracker.py` into
  `tracking_ultrack/seeded_solve.py`.
- Adapt to v2's manual-correction storage: `src/cellflow/database/validation.py`
  marks frames as validated; `database/tracked.py` stores corrected labels.
  The solver should constrain its ILP to *fix* the IDs in validated frames
  (e.g. by forcing `selected=True` on matching nodes and forcing `parent_id`
  via additional ILP constraints).

**Acceptance:** correcting frame `t` and re-solving produces a track structure
that respects the corrections AND propagates them forwards/backwards.

---

## 5. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Ultrack's bbox/mask format is subtly different from what we assume (3D ZYX vs YXZ) | Medium | High | Phase 0 probe script validates round-trips before bulk ingest. |
| `pickle.dumps(node)` produces blobs that Ultrack can't reconstruct because of class-path mismatches | Low | High | Verify in Phase 0 by querying back and pickle-loading. |
| Per-frame OverlapDB row count explodes for dense param sweeps (e.g. 50 hypotheses × 500 cells = 12.5k rows × overlap rate) | Medium | Medium | Bbox-prune first; cap `n_p` if needed; SQLite handles ~100k rows easily. |
| `solver_name="GUROBI"` requires a license, falling back to `"CBC"` may be too slow for large frames | Low | Medium | The archive's `_select_solver()` already falls through to CBC; test on real data early. |
| ILP solver uses `hier_parent_id` for cost shaping in some configurations and we left it as `NO_PARENT` | Low | Medium | Phase 0 grep confirms solver doesn't read it for tracking. If a config flag flips this, derive it from containment. |
| Hypothesis sweep produces partitions that are very dissimilar (different methods, e.g. cellpose-flow vs seeded-watershed), making OverlapDB enormous | Medium | Low | OverlapDB only constrains "can't both be picked". Many overlaps just mean ILP has more flexibility. Still cap with bbox prune. |

---

## 6. Out of scope (don't expand here)

- Replacing the v2 hypothesis generator. It's good. Keep it.
- Re-implementing Ultrack's solver. We're using their ILP, not building one.
- Reverting the napari workflow UI or hypothesis DB visualizer. They stay.
- Multi-position batch processing — focus on single-position correctness first.

---

## 7. Reference: file map

**Files to create:**
- `src/cellflow/tracking_ultrack/__init__.py`
- `src/cellflow/tracking_ultrack/config.py`
- `src/cellflow/tracking_ultrack/ingest.py`              ← the heart of this work
- `src/cellflow/tracking_ultrack/linking.py`            ← lift from v1 archive
- `src/cellflow/tracking_ultrack/solve.py`
- `src/cellflow/tracking_ultrack/export.py`
- `src/cellflow/tracking_ultrack/seeded_solve.py`       ← Phase 4 only
- `tests/test_tracking_ultrack_ingest.py`
- `scripts/probe_ultrack_db.py`                          ← Phase 0 throwaway

**Files to read for context (in priority order):**
1. `src/cellflow/database/hypotheses.py` — current HDF5 schema and read API.
2. `ultrack/core/segmentation/processing.py:_process` (in installed env at
   `/home/aruppel/miniconda3/envs/cellflow/lib/python3.10/site-packages/ultrack/core/segmentation/processing.py`)
   — canonical write path. Lines 161–339.
3. `ultrack/core/database.py` lines 77–120 — exact schema.
4. `ultrack/core/segmentation/node.py` — `Node` class, especially `from_mask`
   (line 278) and `__getstate__`/`__setstate__` (line 250).
5. `archive/v1/packages/ultrack/src/cellflow/ultrack/stages/tracking.py` —
   `_build_ultrack_config()`, `run_segmentation`, `run_linking`, `run_solve`,
   `export_tracked_labels`.
6. `archive/v1/packages/ultrack/src/cellflow/ultrack/linking.py` — IoU-aware
   linking we want to lift.
7. `archive/v1/packages/ultrack/src/cellflow/ultrack/stages/seeded_tracker.py` —
   Phase 4 reference for seeded solving.
8. `scripts/test_anchor_lap_propagator.py` — current v2 benchmark; the new
   pipeline needs to score at least as well on the same metric.

**Files to edit (only in Phase 3):**
- `src/cellflow/napari/cell_workflow_widget.py` — add tracker toggle.

**Files to leave alone:**
- `src/cellflow/tracking/propagator.py` and `propagator_v2.py` — keep until
  the new path is validated. Plan to remove in a follow-up.

---

## 8. Estimated total effort

- Phase 0: 0.5 day
- Phase 1: 1.5 days
- Phase 2: 1 day
- Phase 3: 0.5 day
- Phase 4: 2–3 days (deferred; only after Phases 0–3 succeed)

**Total to validated end-to-end Ultrack tracker (Phases 0–3): ~3.5 days.**
Add ~3 days if seeded solving is in scope for this iteration.

---

## 9. Decision points / open questions for the human

1. **Single-position vs multi-position:** confirm we're targeting one
   position end-to-end first. human answer: yes
2. **Dataset choice for validation:** is
   `2026-04-01_U251.../v2/pos00/2_nucleus/hypotheses.h5` the right benchmark,
   or is there a position with v1 ground-truth tracking output for direct
   comparison? human answer: the first 10 frames used in /home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/v2/pos00/2_nucleus/tracked_labels.tif can serve as ground truth for testing
3. **Solver:** is Gurobi licensed on the workstation? If not, CBC is fine for
   the test dataset but may be the bottleneck at scale. human answer: Gurobi licence exists yes. 
4. **Phase 4 priority:** is the seeded-solver port required for this sprint
   or can it be a follow-up? The minimum viable path (Phases 0–3) gives
   v1-equivalent tracking; Phase 4 gives v1-equivalent + manual-correction
   integration (the original motivation for v2). human answer: it can be a follow-up
