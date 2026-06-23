# CellFlow — Code Review

Date: 2026-06-23 · Version reviewed: 0.2.0

Scope: full source tree (`aggregate_quantification`, `tracking_ultrack`,
`segmentation`, `correction`, `cellpose`, `core`, `napari` UI) plus packaging,
CI, and docs.

## Baseline health (good)

- `ruff check` passes clean.
- The lightweight test suites (aggregate / core / segmentation / cellpose /
  tracking) pass.
- No bare `except:`, no mutable default args, no `== None`.
- Heavy work runs off the Qt thread via napari `thread_worker`; the
  interactive-correction teardown properly disconnects per-session signals.

This is a well-structured, disciplined codebase. Findings below are grouped by
severity, most important first.

## Critical — data corruption / crashes

1. **`segmentation/cell_label_icm.py` — silent label truncation `uint32 → uint16`.**
   `commit_labels` cast `uint32` labels to `uint16` with no overflow check; track
   ids above 65535 wrapped silently, merging distinct cells on disk.
   *Status: FIXED — dtype now chosen from the actual max (uint16 when it fits,
   else uint32). Regression test added.*
2. **`correction/labels.py` — `_free_label` overflow on `uint16` arrays.**
   Returned `seg.max()+1` and wrote it back into a `uint16` array; at the dtype
   ceiling it wrapped to background / collided with an existing id.
   *Status: FIXED — raises `OverflowError` at the dtype ceiling. Regression test
   added.*
3. **`cellpose/divergence_maps.py` — `np.stack([])` crash on empty stack.**
   `build_divergence_maps` called `np.stack(frames)` unconditionally; a `T=0`
   input raised `ValueError` after the output dir was created.
   *Status: FIXED — guards `n_t == 0` with a clear message. Regression test
   added.*
4. **`core/label_store.py:36-51` — O(T²) full-stack rewrite per frame.**
   `write_tracked_frame` decompresses + re-encodes the entire zlib TIFF on every
   single-frame write → severe stalls in long correction sessions.
   *Status: FIXED — added `write_full_tracked_stack` (one O(T) encode); the two
   napari save paths now batch-write the whole stack instead of looping the
   per-frame writer. Regression tests added.*

## High

5. **`segmentation/cell_label_icm.py:45,219` — `_INF = 1e9` is finite.**
   Foreground pixels no seed reaches keep `best_ki=0` and get assigned
   `label_ids[0]` instead of background.
   *Status: FIXED — `_argmin_init_from_dict` masks unreached pixels
   (`best_cost >= _INF`) to background in the final `np.where`. Regression test
   added.*
6. **`tracking_ultrack/_node_geometry.py:108-118` — centroid-in-bbox prefilter
   drops valid matches.** A node that genuinely overlaps the source mask but whose
   centroid falls outside its bbox (elongated/crescent/merged masks) is never
   considered.
   *Status: FIXED — NodeDB has no bbox columns, so prefilter by centroid distance
   instead of bbox containment: candidates within the source bbox diagonal of the
   source centroid are admitted, then exact IoU prunes non-overlappers. Pure-gate
   + DB-backed regression tests added.*
7. **`tracking_ultrack/export.py:99-110` — three `except Exception: pass` mask
   export failures.** A real bug (corrupt DB, OOM) is hidden and a degraded CTC
   fallback may emit wrong/empty labels.
   *Status: FIXED — capability probe (import) is separated from execution; only
   `ImportError`/`AttributeError` falls through to the next strategy, runtime
   errors propagate. Regression tests added.*
8. **`aggregate_quantification/curation.py:162` — `apply_curation` crashes on NaN
   `frame`.** `out["frame"].astype("int64")` raises on shape tables whose
   outer-merge produced NaN frames.
   *Status: FIXED — uses the `pd.to_numeric(..., errors="coerce")` +
   `.notna()` pattern from `remove_exclusion`; NaN frames no longer crash the
   cast and simply don't match. Regression test added.*
9. **`core/label_store.py:73-79` — `tracked_frame_exists` uses `.any()`.** A
   legitimately-tracked all-background frame reads as "not tracked", causing
   re-processing / gaps.
   *Status: FIXED — written timepoints are recorded in a JSON sidecar; existence
   is decoupled from content. Legacy files without a sidecar fall back to the
   content heuristic. Regression tests added.*
10. **cellpose `do_3d=True` output incompatible with divergence builder.**
    `run_nucleus_stack(do_3d=True)` yields dp `(T,3,Z,Y,X)`, but
    `build_divergence_maps` requires `(T,Z,2,Y,X)` and rejects it — a supported
    option silently produces unusable output.
    *Status: FIXED — the dp-shape validator now recognises the `(T,3,Z,Y,X)`
    3D-flow layout and raises an actionable error naming `do_3d`, instead of a
    cryptic shape message or (when Z==2) a silent axis misread. Regression test
    added.*

## Medium

- ~~`aggregate_quantification/reduce.py:271-274` — `_is_numeric` uses `.any()`, so a
  categorical column with one parseable number gets averaged.~~ FIXED: requires
  every non-null value to parse as numeric.
- ~~`aggregate_quantification/reduce.py:129,142` — a `<`/`>` filter on an
  existing-but-non-numeric column drops all rows silently instead of no-op.~~
  FIXED: a fully non-numeric column under an ordered op is a no-op (keeps rows).
- ~~`aggregate_quantification/pipeline.py:129` / `quantifier.py:124-128` —
  `missing_build_params` only consults the `params` dict, so pixel size set
  per-record (not under `[params]`) silently skips shape/dynamics builds.~~
  FIXED: the build-param gate is now per-record, checking the record's own value
  overlaid on the shared params; only positions satisfying neither are skipped.
- ~~`tracking_ultrack/seed_prior.py:144-148` — per-node `UPDATE` (O(n) statements);
  use `bulk_update_mappings`. Also `normalized_base ** quality_exponent` can
  NaN/inf at 0.~~ FIXED: single `bulk_update_mappings`; prob via `_seed_node_prob`
  which returns 0.0 for a non-positive base (no `0**0`/`0**neg`).
- ~~`tracking_ultrack/db_build.py:404-406` — `engine.dispose()` inside the per-frame
  loop; dispose once after.~~ FIXED: disposed once after the build loop.
- ~~`tracking_ultrack/corrections.py:233-234` — anchor chaining assumes `t+1` is the
  next anchor; non-consecutive anchors leave the gap unbridged.~~ WON'T FIX (by
  design): LinkDB edges are frame-to-frame, so non-consecutive anchors cannot be
  link-forced; the solver bridges the gap with intermediate candidate nodes.
- ~~`segmentation/nucleus_segmentation.py:129-130` — `np.random.normal` uses the
  global RNG; `run_index` is dead → non-reproducible segmentation.~~ FIXED: noise
  now drawn from a local `default_rng(run_index)`; reproducible per run_index.
- ~~`correction/labels.py:937` — `clean_stranded_pixels` runs `expand_labels` over
  the whole frame per fragment (≈O(n_fragments × frame)). Restrict to a padded
  bbox.~~ FIXED: cropped to the fragment bbox padded by the expansion distance;
  a 200-trial equivalence test confirms output-identical to the whole-frame path.
- ~~`cellpose/divergence_maps.py:108-110` — `np.gradient` crashes on a 1-pixel Y/X
  axis.~~ FIXED: a singleton axis contributes zero divergence and is skipped.
- ~~`aggregate_quantification/dynamics/store.py:236` — `_read_table` returns columns
  in HDF5 iteration (alphabetical) order, not declared order.~~ FIXED: authored
  order persisted in a `column_order` attr and restored on read (legacy fallback).
- ~~`aggregate_quantification/catalog.py:179-183` — discovery-only metadata
  fallbacks (`id=stem`, `experiment_id=date`) can collide or block loads.~~
  NON-ISSUE: `load_catalog` already calls `_check_unique_identity`, which raises a
  clear named-duplicate error on collision (never silent pooling); `id` is a
  required column so the stem fallback rarely fires; `experiment_id=date` is the
  documented single-replicate default.
- ~~`napari/correction_widget.py:432-499` — activate clears all
  `viewer.mouse_drag_callbacks` and restores on deactivate; overlapping owners
  could leave the saved list stale~~ WON'T FIX (by design): the single-active-mode
  UI gate prevents overlapping owners, as the review itself notes; no reachable
  defect without a live multi-owner viewer state.

## Low

- ~~`napari/main_widget.py:431-456` — config save/load failures only `print()` to
  the console, no GUI feedback.~~ FIXED: save/load now report through napari
  notifications (`show_info`/`show_warning`/`show_error`).
- ~~`tracking_ultrack/validation_state.py:84-100` — `read_validated_tracks`
  re-parses both JSON files on every call (hot in overlay loops).~~ FIXED:
  `read_corrections` and the legacy reader memoize on the file's (mtime_ns, size)
  signature and return fresh copies; a write invalidates the entry.
- ~~`segmentation/nucleus_segmentation.py:91` — `if not coords` is a dead guard on a
  tuple of arrays; only `coords[0].size == 0` actually fires.~~ FIXED: dead clause
  removed; emptiness checked via the coordinate array length.
- ~~`segmentation/cell_label_icm.py:463-489` — unary cache read swallows errors
  silently; a corrupt cache is indistinguishable from a cold one.~~ FIXED: an
  unreadable (present) cache now logs a warning before recomputing; a true cold
  miss stays silent.
- Code duplication: `_cellflow_version`, `_report_progress`, `_columns_from_rows`
  (aggregate) and `create_engine(...)` boilerplate (tracking, with inconsistent
  `check_same_thread`) are copy-pasted across modules.
  PARTIALLY FIXED: the byte-identical `_cellflow_version` and `_report_progress`
  (3 copies each) are consolidated into `aggregate_quantification/_provenance.py`.
  Left as-is by design: `_columns_from_rows` is same-name/different-logic (contacts
  has object/bool dtype special-casing — not true duplication); the tracking
  `create_engine` calls differ intentionally (`check_same_thread=False` only on the
  three worker-thread read paths), so a blanket helper would alter threading
  semantics for low value.

## Cross-cutting / packaging / docs

- ~~**License mismatch (High, pre-release).** Umbrella `cellflow` is AGPL-3.0
  (LICENSE + pyproject + README); all five extracted sub-packages declare
  GPL-3.0.~~ FIXED: AGPL-3.0 is authoritative everywhere (sub-package pyprojects,
  CITATION.cff, packaging test). Each sub-package also now carries a `LICENSE`
  copy of the AGPL text, so a standalone wheel/sdist ships the license in
  `dist-info/licenses/` (the umbrella already did) — verified by a test build.
- **Doc drift.** README documents `4_contact_analysis/`, but the code
  (`napari/main_widget.py:472`) and `napari/_paths.py` use
  `aggregate_quantification/`.
- **README Python version.** Says "requires Python 3.9 or newer";
  `requires-python = ">=3.10"`.
- **Robustness theme.** ~45 silent `except …: pass` blocks; broad
  `except Exception` concentrated in napari widgets and
  `tracking_ultrack/db_query.py` (9 in one module). Several mask real failures
  (#7 is the worst).
  PARTIALLY ADDRESSED: #7 (export) fixed; the two `db_query.py` optional-import
  probes for `OverlapDB` narrowed to `(ImportError, AttributeError)` so a runtime
  error can't masquerade as "no overlaps" (same class as #7). Left by design: the
  remaining `db_query` swallows are legitimate "DB-not-ready → safe default"
  query guards (narrowing them would break the graceful pre-build UI path), and
  the napari widget guards are defensive event-handler catches — both are
  observability/style cleanups, not correctness masks, and are deferred.
- **Test gap.** `segmentation` is comparatively under-tested relative to the rest.
  PARTIALLY ADDRESSED: this review pass added segmentation regression tests
  (`test_nucleus_segmentation.py`, plus `_argmin_init`, dtype-write, unary-cache,
  and per-label fill coverage in `test_cell_label_icm.py`).

## Fixed

Critical/High items resolved so far, each with a regression test:
- #1 `segmentation/cell_label_icm.py::commit_labels` — dtype-aware label write.
- #2 `correction/labels.py::_free_label` — overflow guard.
- #3 `cellpose/divergence_maps.py::build_divergence_maps` — empty-stack guard.
- #4 `core/label_store.py::write_full_tracked_stack` — O(T) batched save replaces
  the per-frame loop in the napari correction widget.
- #7 `tracking_ultrack/export.py` — capability detection narrowed; runtime export
  failures propagate instead of silently degrading.
- #5 `segmentation/cell_label_icm.py` — unreached foreground pixels stay background
  instead of collapsing onto `label_ids[0]`.
- #9 `core/label_store.py` — explicit written-frame sidecar; existence no longer
  inferred from `.any()` content.
- #8 `aggregate_quantification/curation.py` — NaN-frame-tolerant frame match.
- #6 `tracking_ultrack/_node_geometry.py` — centroid-distance prefilter replaces
  bbox containment.
- #10 `cellpose/divergence_maps.py` — actionable error on `do_3d` 3D-flow input.

All Critical and High items are resolved. Medium tier resolved (or judged
by-design):
- reduce.py `_is_numeric` strictness + ordered-filter no-op on non-numeric.
- pipeline.py per-record build-param gate (per-position pixel size now builds).
- seed_prior.py bulk update + finite zero-base prob; db_build.py single dispose.
- nucleus_segmentation.py reproducible `default_rng(run_index)`.
- clean_stranded_pixels padded-bbox crop (equivalence-tested).
- divergence_2d singleton-axis tolerance; dynamics/store column-order preserved.
- corrections.py anchor chaining, catalog.py id fallback, correction_widget mouse
  callbacks — judged by-design / already-guarded (see notes above).

Remaining: the Low tier and the cross-cutting items (license mismatch + its failing
`test_packaging_metadata` case, README doc/version drift, the broad-`except` theme).
