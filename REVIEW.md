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
   considered. Fix: prefilter by bbox intersection, not centroid containment.
   *Open.*
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
    option silently produces unusable output. Fix: guard/document the path, or
    accept the 3-component layout. *Open.*

## Medium

- `aggregate_quantification/reduce.py:271-274` — `_is_numeric` uses `.any()`, so a
  categorical column with one parseable number gets averaged. Require all non-null
  values numeric.
- `aggregate_quantification/reduce.py:129,142` — a `<`/`>` filter on an
  existing-but-non-numeric column drops all rows silently instead of no-op.
- `aggregate_quantification/pipeline.py:129` / `quantifier.py:124-128` —
  `missing_build_params` only consults the `params` dict, so pixel size set
  per-record (not under `[params]`) silently skips shape/dynamics builds.
- `tracking_ultrack/seed_prior.py:144-148` — per-node `UPDATE` (O(n) statements);
  use `bulk_update_mappings`. Also `normalized_base ** quality_exponent` can
  NaN/inf at 0.
- `tracking_ultrack/db_build.py:404-406` — `engine.dispose()` inside the per-frame
  loop; dispose once after.
- `tracking_ultrack/corrections.py:233-234` — anchor chaining assumes `t+1` is the
  next anchor; non-consecutive anchors leave the gap unbridged.
- `segmentation/nucleus_segmentation.py:129-130` — `np.random.normal` uses the
  global RNG; `run_index` is dead → non-reproducible segmentation. Use
  `default_rng(run_index)`.
- `correction/labels.py:937` — `clean_stranded_pixels` runs `expand_labels` over
  the whole frame per fragment (≈O(n_fragments × frame)). Restrict to a padded
  bbox.
- `cellpose/divergence_maps.py:108-110` — `np.gradient` crashes on a 1-pixel Y/X
  axis.
- `aggregate_quantification/dynamics/store.py:236` — `_read_table` returns columns
  in HDF5 iteration (alphabetical) order, not declared order.
- `aggregate_quantification/catalog.py:179-183` — discovery-only metadata
  fallbacks (`id=stem`, `experiment_id=date`) can collide or block loads.
- `napari/correction_widget.py:432-499` — activate clears all
  `viewer.mouse_drag_callbacks` and restores on deactivate; overlapping owners
  could leave the saved list stale (the UI gate should prevent this).

## Low

- `napari/main_widget.py:431-456` — config save/load failures only `print()` to
  the console, no GUI feedback.
- `tracking_ultrack/validation_state.py:84-100` — `read_validated_tracks`
  re-parses both JSON files on every call (hot in overlay loops).
- `segmentation/nucleus_segmentation.py:91` — `if not coords` is a dead guard on a
  tuple of arrays; only `coords[0].size == 0` actually fires.
- `segmentation/cell_label_icm.py:463-489` — unary cache read swallows errors
  silently; a corrupt cache is indistinguishable from a cold one.
- Code duplication: `_cellflow_version`, `_report_progress`, `_columns_from_rows`
  (aggregate) and `create_engine(...)` boilerplate (tracking, with inconsistent
  `check_same_thread`) are copy-pasted across modules.

## Cross-cutting / packaging / docs

- **License mismatch (High, pre-release).** Umbrella `cellflow` is AGPL-3.0
  (LICENSE + pyproject + README); all five extracted sub-packages declare
  GPL-3.0. Resolve deliberately before any public/JOSS release.
- **Doc drift.** README documents `4_contact_analysis/`, but the code
  (`napari/main_widget.py:472`) and `napari/_paths.py` use
  `aggregate_quantification/`.
- **README Python version.** Says "requires Python 3.9 or newer";
  `requires-python = ">=3.10"`.
- **Robustness theme.** ~45 silent `except …: pass` blocks; broad
  `except Exception` concentrated in napari widgets and
  `tracking_ultrack/db_query.py` (9 in one module). Several mask real failures
  (#7 is the worst).
- **Test gap.** `segmentation` is comparatively under-tested relative to the rest.

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

Suggested next: #6 (centroid-in-bbox prefilter drops valid track matches — needs
a correctness/perf decision, see note) and #10 (cellpose `do_3d` output shape).
