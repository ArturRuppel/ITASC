# Graph Cut Cell Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `scripts/experiment_cell_2d_graphcut.py` — a 2D per-frame α-expansion graph cut experiment that assigns cell labels using a Potts smoothness term derived from contour maps, with hard-pinned centroid seeds.

**Architecture:** For each frame independently, build a binary PyMaxflow graph per label (α-expansion), run max-flow, and update label assignments. Repeat for up to 5 rounds or until convergence. Stack 50 frames into (T, Y, X) output. Sweep `--smoothness-weight` values.

**Tech Stack:** PyMaxflow 1.3.2, numpy, tifffile, scikit-image (`centroid_markers_from_labels`), scipy (`distance_transform_edt` for initialization).

---

## File Structure

- **Create:** `scripts/experiment_cell_2d_graphcut.py` — full experiment script
- **Create:** `tests/segmentation/test_graphcut.py` — unit tests for `_run_alpha_expansion`

---

### Task 1: Write failing tests for `_run_alpha_expansion`

**Files:**
- Create: `tests/segmentation/test_graphcut.py`

- [ ] **Step 1: Write the test file**

```python
# tests/segmentation/test_graphcut.py
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Load the script module without executing main()
_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "experiment_cell_2d_graphcut.py"


@pytest.fixture(scope="module")
def graphcut_module():
    spec = importlib.util.spec_from_file_location("experiment_cell_2d_graphcut", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_alpha_expansion_hard_pins_seed_pixels(graphcut_module):
    """Seed pixels must retain their label after expansion."""
    contours = np.zeros((6, 6), dtype=np.float32)
    foreground = np.ones((6, 6), dtype=bool)
    seeds = np.zeros((6, 6), dtype=np.uint32)
    seeds[0, 0] = 1
    seeds[5, 5] = 2

    labels = graphcut_module._run_alpha_expansion(
        contours, foreground, seeds, smoothness_weight=10.0, max_rounds=3
    )

    assert labels[0, 0] == 1
    assert labels[5, 5] == 2
    assert np.all(labels[foreground] > 0), "all foreground pixels must be labeled"


def test_alpha_expansion_background_pixels_stay_zero(graphcut_module):
    """Pixels outside the foreground mask must stay 0."""
    contours = np.zeros((4, 4), dtype=np.float32)
    foreground = np.ones((4, 4), dtype=bool)
    foreground[3, :] = False  # bottom row is background
    seeds = np.zeros((4, 4), dtype=np.uint32)
    seeds[0, 0] = 1
    seeds[1, 3] = 2

    labels = graphcut_module._run_alpha_expansion(
        contours, foreground, seeds, smoothness_weight=5.0, max_rounds=3
    )

    assert np.all(labels[~foreground] == 0)
    assert labels[0, 0] == 1
    assert labels[1, 3] == 2


def test_alpha_expansion_contour_barrier_splits_labels(graphcut_module):
    """A perfect contour barrier (value=1.0) makes crossing it free, concentrating
    the boundary there rather than elsewhere."""
    # 1-row, 8-column strip. Strong contour barrier between columns 3 and 4.
    contours = np.zeros((1, 8), dtype=np.float32)
    contours[0, 3] = 1.0
    contours[0, 4] = 1.0
    foreground = np.ones((1, 8), dtype=bool)
    seeds = np.zeros((1, 8), dtype=np.uint32)
    seeds[0, 0] = 1
    seeds[0, 7] = 2

    labels = graphcut_module._run_alpha_expansion(
        contours, foreground, seeds, smoothness_weight=50.0, max_rounds=5
    )

    assert labels[0, 0] == 1
    assert labels[0, 7] == 2
    # Left side of barrier should be label 1, right side label 2
    assert labels[0, 1] == 1
    assert labels[0, 2] == 1
    assert labels[0, 5] == 2
    assert labels[0, 6] == 2
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
conda run -n cellflow pytest tests/segmentation/test_graphcut.py -v
```

Expected: `ModuleNotFoundError` or `FileNotFoundError` — the script doesn't exist yet.

---

### Task 2: Implement `_run_alpha_expansion` and helper utilities

**Files:**
- Create: `scripts/experiment_cell_2d_graphcut.py`

- [ ] **Step 1: Create the script with imports, utilities, and `_run_alpha_expansion`**

```python
"""Run a 2D per-frame α-expansion graph cut experiment for cell labels.

Each frame is segmented independently. Centroid seeds from curated nuclear
labels are hard-pinned. A Potts smoothness term derived from the contour
probability map penalizes label disagreements across low-contour edges.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import maxflow
import numpy as np
import tifffile
from cellflow.segmentation import centroid_markers_from_labels
from scipy.ndimage import distance_transform_edt


DEFAULT_POS_DIR = Path(
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/"
    "analysis/pos00"
)
DEFAULT_CONTOUR_DIR = (
    DEFAULT_POS_DIR
    / "3_cell"
    / "contour_experiment"
    / "20260503-232245-thr-8-to-0-maxfg"
)
DEFAULT_SMOOTHNESS_WEIGHT = [5.0, 20.0, 50.0, 100.0]
_INF = 1e10


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")


def _format_float(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text or "0"


def _run_alpha_expansion(
    contours: np.ndarray,
    foreground: np.ndarray,
    seeds: np.ndarray,
    smoothness_weight: float,
    max_rounds: int = 5,
) -> np.ndarray:
    """2D α-expansion graph cut with hard seed constraints and Potts smoothness.

    contours:  (Y, X) float32 in [0, 1]; high = likely boundary
    foreground: (Y, X) bool; pixels outside are fixed to label 0
    seeds:     (Y, X) uint32; nonzero pixels are hard-pinned to their label
    Returns:   (Y, X) uint32 label assignments; background pixels are 0
    """
    height, width = contours.shape
    fg_mask = np.asarray(foreground, dtype=bool)

    # Map foreground pixels to compact node indices
    fg_yx = np.argwhere(fg_mask)          # (N, 2)
    n_nodes = len(fg_yx)
    if n_nodes == 0:
        return np.zeros((height, width), dtype=np.uint32)

    pixel_to_node = np.full((height, width), -1, dtype=np.int32)
    pixel_to_node[fg_yx[:, 0], fg_yx[:, 1]] = np.arange(n_nodes, dtype=np.int32)

    # Precompute horizontal and vertical n-link weights (vectorized)
    c = np.asarray(contours, dtype=np.float32)
    # horizontal: edge between (y, x) and (y, x+1)
    h_left_yx = fg_yx[(fg_yx[:, 1] + 1 < width) & (pixel_to_node[fg_yx[:, 0], np.minimum(fg_yx[:, 1] + 1, width - 1)] >= 0)]
    # vertical: edge between (y, x) and (y+1, x)
    v_top_yx = fg_yx[(fg_yx[:, 0] + 1 < height) & (pixel_to_node[np.minimum(fg_yx[:, 0] + 1, height - 1), fg_yx[:, 1]] >= 0)]

    def _edge_weight(y0: int, x0: int, y1: int, x1: int) -> float:
        avg = 0.5 * (float(c[y0, x0]) + float(c[y1, x1]))
        return float(smoothness_weight) * max(0.0, 1.0 - avg)

    # Build edge list once (reused across all α rounds)
    edges: list[tuple[int, int, float]] = []
    for y, x in fg_yx:
        node_i = int(pixel_to_node[y, x])
        for dy, dx in ((0, 1), (1, 0)):
            ny, nx = y + dy, x + dx
            if ny >= height or nx >= width:
                continue
            node_j = int(pixel_to_node[ny, nx])
            if node_j < 0:
                continue
            w = _edge_weight(y, x, ny, nx)
            edges.append((node_i, node_j, w))

    # Initialize: each foreground pixel gets the label of its nearest seed
    seed_pixels = seeds > 0
    if seed_pixels.any():
        _, nearest = distance_transform_edt(~seed_pixels, return_indices=True)
        init_labels = seeds[nearest[0], nearest[1]]
    else:
        init_labels = np.zeros((height, width), dtype=np.uint32)
    current_labels = np.where(fg_mask, init_labels, 0).astype(np.uint32)

    label_ids = np.unique(seeds)
    label_ids = label_ids[label_ids != 0]

    for _round in range(max_rounds):
        changed = False
        for alpha in label_ids:
            g = maxflow.Graph[float](n_nodes, len(edges))
            g.add_nodes(n_nodes)

            # Data term: hard-pin seed pixels
            seed_vals = seeds[fg_yx[:, 0], fg_yx[:, 1]]
            for node_id in range(n_nodes):
                sv = int(seed_vals[node_id])
                if sv == int(alpha):
                    g.add_tedge(node_id, _INF, 0.0)
                elif sv != 0:
                    g.add_tedge(node_id, 0.0, _INF)
                # else: free pixel — no terminal edge (0, 0)

            # Smoothness term: 4-connected n-links
            for node_i, node_j, w in edges:
                g.add_edge(node_i, node_j, w, w)

            g.maxflow()

            # Update: source partition (segment=0) → label alpha
            for node_id in range(n_nodes):
                if g.get_segment(node_id) == 0:
                    y, x = int(fg_yx[node_id, 0]), int(fg_yx[node_id, 1])
                    if current_labels[y, x] != alpha:
                        current_labels[y, x] = alpha
                        changed = True

        if not changed:
            break

    return current_labels
```

- [ ] **Step 2: Run tests to confirm `_run_alpha_expansion` passes**

```bash
conda run -n cellflow pytest tests/segmentation/test_graphcut.py -v
```

Expected:
```
tests/segmentation/test_graphcut.py::test_alpha_expansion_hard_pins_seed_pixels PASSED
tests/segmentation/test_graphcut.py::test_alpha_expansion_background_pixels_stay_zero PASSED
tests/segmentation/test_graphcut.py::test_alpha_expansion_contour_barrier_splits_labels PASSED
```

- [ ] **Step 3: Commit**

```bash
git add scripts/experiment_cell_2d_graphcut.py tests/segmentation/test_graphcut.py
git commit -m "feat(graphcut): add alpha-expansion core and passing tests"
```

---

### Task 3: Complete the experiment script (I/O, main loop, summaries)

**Files:**
- Modify: `scripts/experiment_cell_2d_graphcut.py`

- [ ] **Step 1: Add `_parse_args`, `_load_inputs`, `_summarize_labels`, and `main` to the script**

Append the following to `scripts/experiment_cell_2d_graphcut.py` after the `_run_alpha_expansion` function:

```python
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2D per-frame α-expansion graph cut for tracked cell segmentation."
    )
    parser.add_argument("--pos-dir", type=Path, default=DEFAULT_POS_DIR)
    parser.add_argument(
        "--contours",
        type=Path,
        default=DEFAULT_CONTOUR_DIR / "contours.tif",
        help="Contour probability volume, expected shape (T, Y, X).",
    )
    parser.add_argument(
        "--foreground-mask",
        type=Path,
        default=DEFAULT_CONTOUR_DIR / "foreground_masks.tif",
        help="Binary foreground domain mask, expected shape (T, Y, X).",
    )
    parser.add_argument(
        "--markers",
        type=Path,
        default=DEFAULT_POS_DIR / "2_nucleus" / "tracked_labels.tif",
        help="Curated tracked nuclear labels, expected shape (T, Y, X).",
    )
    parser.add_argument(
        "--smoothness-weight",
        type=float,
        nargs="+",
        default=DEFAULT_SMOOTHNESS_WEIGHT,
        help="Potts smoothness weight; higher = stronger contour barriers.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum α-expansion rounds per frame.",
    )
    parser.add_argument(
        "--timestamp",
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Checkpoint directory name under 3_cell/graphcut_experiment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing timestamped checkpoint directory.",
    )
    parser.add_argument(
        "--crop",
        type=int,
        nargs=6,
        metavar=("T0", "T1", "Y0", "Y1", "X0", "X1"),
        help="Optional crop for pilot runs, using half-open ranges.",
    )
    parser.add_argument(
        "--contours-source",
        type=Path,
        default=None,
        help="Override contours path (e.g. use mean-z contours).",
    )
    return parser.parse_args()


def _load_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contour_path = args.contours_source if args.contours_source is not None else args.contours
    contours = np.asarray(tifffile.imread(contour_path), dtype=np.float32)
    foreground = np.asarray(tifffile.imread(args.foreground_mask))
    markers = np.asarray(tifffile.imread(args.markers), dtype=np.uint32)

    for arr, name in ((contours, "contours"), (foreground, "foreground"), (markers, "markers")):
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim != 3:
            raise ValueError(f"Expected {name} as (T, Y, X), got {arr.shape}")

    # Re-extract after potential squeeze
    if contours.ndim == 4 and contours.shape[1] == 1:
        contours = contours[:, 0]
    if foreground.ndim == 4 and foreground.shape[1] == 1:
        foreground = foreground[:, 0]
    if markers.ndim == 4 and markers.shape[1] == 1:
        markers = markers[:, 0]

    if foreground.shape != contours.shape:
        raise ValueError(
            f"Foreground shape {foreground.shape} does not match contours {contours.shape}"
        )
    if markers.shape != contours.shape:
        raise ValueError(
            f"Markers shape {markers.shape} does not match contours {contours.shape}"
        )

    if args.crop is not None:
        t0, t1, y0, y1, x0, x1 = args.crop
        crop = (slice(t0, t1), slice(y0, y1), slice(x0, x1))
        contours = contours[crop]
        foreground = foreground[crop]
        markers = markers[crop]

    return contours, foreground, markers


def _summarize_labels(
    labels: np.ndarray, markers: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    marker_ids = np.unique(markers)
    marker_ids = marker_ids[marker_ids != 0]
    label_ids = np.unique(labels)
    label_ids = label_ids[label_ids != 0]
    missing_ids = np.setdiff1d(marker_ids, label_ids)
    extra_ids = np.setdiff1d(label_ids, marker_ids)
    unlabeled_foreground = mask & (labels == 0)
    return {
        "n_marker_ids": int(marker_ids.size),
        "n_output_ids": int(label_ids.size),
        "missing_marker_ids": [int(v) for v in missing_ids[:50]],
        "n_missing_marker_ids": int(missing_ids.size),
        "extra_output_ids": [int(v) for v in extra_ids[:50]],
        "n_extra_output_ids": int(extra_ids.size),
        "foreground_voxels": int(np.count_nonzero(mask)),
        "labeled_voxels": int(np.count_nonzero(labels)),
        "unlabeled_foreground_voxels": int(np.count_nonzero(unlabeled_foreground)),
        "max_label": int(labels.max()) if labels.size else 0,
    }


def main() -> None:
    args = _parse_args()
    output_dir = args.pos_dir / "3_cell" / "graphcut_experiment" / args.timestamp
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{output_dir} exists; pass --overwrite or use a new --timestamp")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading inputs...", flush=True)
    contours, foreground_raw, full_markers = _load_inputs(args)
    seeds = centroid_markers_from_labels(full_markers)
    fg_mask = foreground_raw > 0

    params: dict[str, Any] = {
        "script": str(Path(__file__).resolve()),
        "pos_dir": args.pos_dir,
        "contours": args.contours,
        "contours_source": args.contours_source,
        "foreground_mask": args.foreground_mask,
        "markers": args.markers,
        "smoothness_weight": [float(v) for v in args.smoothness_weight],
        "max_rounds": int(args.max_rounds),
        "shape": tuple(int(v) for v in contours.shape),
        "crop": args.crop,
        "seed_mode": "centroid",
        "seed_marker_voxels": int(np.count_nonzero(seeds)),
        "foreground_voxels": int(np.count_nonzero(fg_mask)),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": output_dir,
    }
    _write_json(output_dir / "parameters.json", params)
    tifffile.imwrite(output_dir / "seed_markers.tif", seeds, compression="zlib")

    n_frames = contours.shape[0]
    summaries: list[dict[str, Any]] = []

    for smoothness_weight in args.smoothness_weight:
        suffix = f"sw_{_format_float(float(smoothness_weight))}"
        label_path = output_dir / f"tracked_labels_{suffix}.tif"
        print(f"Running graph cut {suffix} ({n_frames} frames)...", flush=True)
        t0 = perf_counter()

        all_frames: list[np.ndarray] = []
        for t in range(n_frames):
            frame_labels = _run_alpha_expansion(
                contours[t],
                fg_mask[t],
                seeds[t],
                smoothness_weight=float(smoothness_weight),
                max_rounds=int(args.max_rounds),
            )
            all_frames.append(frame_labels)
            if (t + 1) % 10 == 0 or t + 1 == n_frames:
                print(f"  frame {t + 1}/{n_frames}", flush=True)

        labels = np.stack(all_frames, axis=0).astype(np.uint32)
        elapsed_s = perf_counter() - t0

        tifffile.imwrite(label_path, labels, compression="zlib")
        summary = {
            "smoothness_weight": float(smoothness_weight),
            "elapsed_s": round(elapsed_s, 3),
            "path": label_path,
            **_summarize_labels(labels, seeds, fg_mask),
        }
        summaries.append(summary)
        _write_json(output_dir / f"summary_{suffix}.json", summary)
        print(
            f"  wrote {label_path.name}: {summary['n_output_ids']} IDs, "
            f"{summary['n_missing_marker_ids']} missing, "
            f"{summary['unlabeled_foreground_voxels']} unlabeled fg, "
            f"{elapsed_s:.1f}s",
            flush=True,
        )

    params["finished_at"] = datetime.now().isoformat(timespec="seconds")
    params["summaries"] = summaries
    _write_json(output_dir / "parameters.json", params)
    _write_json(output_dir / "summaries.json", {"summaries": summaries})
    print("Done.", flush=True)
    print(f"Output directory: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the tests again to make sure nothing broke**

```bash
conda run -n cellflow pytest tests/segmentation/test_graphcut.py -v
```

Expected: 3 tests PASSED.

- [ ] **Step 3: Commit**

```bash
git add scripts/experiment_cell_2d_graphcut.py
git commit -m "feat(graphcut): add CLI, I/O, and main experiment loop"
```

---

### Task 4: Pilot run on a small crop to verify correctness

**Files:**
- Read-only verification step

- [ ] **Step 1: Run a 3-frame, 128×128 crop with one smoothness weight**

```bash
conda run -n cellflow python scripts/experiment_cell_2d_graphcut.py \
  --timestamp 20260504-graphcut-pilot \
  --contours /home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/3_cell/contour_experiment/20260504-contours-meanz-thr-m5-to-5/contours.tif \
  --foreground-mask /home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/3_cell/contour_experiment/20260503-232245-thr-8-to-0-maxfg/foreground_masks.tif \
  --smoothness-weight 20 \
  --max-rounds 3 \
  --crop 0 3 192 320 192 320 \
  --overwrite
```

- [ ] **Step 2: Verify pilot output**

```bash
conda run -n cellflow python -c "
import json, tifffile, numpy as np
from pathlib import Path
p = Path('/home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/3_cell/graphcut_experiment/20260504-graphcut-pilot')
summaries = json.loads((p / 'summaries.json').read_text())
for s in summaries['summaries']:
    print(f\"sw={s['smoothness_weight']} ids={s['n_output_ids']} missing={s['n_missing_marker_ids']} unlabeled={s['unlabeled_foreground_voxels']} elapsed={s['elapsed_s']:.1f}s\")
labels = tifffile.imread(p / 'tracked_labels_sw_20.tif')
print('labels shape:', labels.shape, 'dtype:', labels.dtype, 'unique:', len(np.unique(labels)))
"
```

Expected: no errors, labels shape (3, 128, 128), some nonzero IDs, elapsed < 120s.

---

### Task 5: Full experiment run

**Files:**
- Read-only verification step

- [ ] **Step 1: Run the full dataset with 4 smoothness weights**

Use the mean-z contours (same as best geodesic run):

```bash
conda run -n cellflow python scripts/experiment_cell_2d_graphcut.py \
  --timestamp 20260504-graphcut-meanzcontours-centroidseeds \
  --contours /home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/3_cell/contour_experiment/20260504-contours-meanz-thr-m5-to-5/contours.tif \
  --foreground-mask /home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/3_cell/contour_experiment/20260503-232245-thr-8-to-0-maxfg/foreground_masks.tif \
  --smoothness-weight 5 20 50 100 \
  --max-rounds 5 \
  --overwrite
```

- [ ] **Step 2: Verify outputs**

```bash
conda run -n cellflow python -c "
import json, tifffile, numpy as np
from pathlib import Path
p = Path('/home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/analysis/pos00/3_cell/graphcut_experiment/20260504-graphcut-meanzcontours-centroidseeds')
summaries = json.loads((p / 'summaries.json').read_text())
for s in summaries['summaries']:
    print(f\"sw={s['smoothness_weight']:>6} ids={s['n_output_ids']} missing={s['n_missing_marker_ids']} unlabeled={s['unlabeled_foreground_voxels']} elapsed={s['elapsed_s']:.1f}s\")
for tif in sorted(p.glob('tracked_labels_*.tif')):
    arr = tifffile.imread(tif)
    print(tif.name, arr.shape, arr.dtype, 'ids:', len(np.unique(arr)) - 1)
"
```

Expected: 4 TIF outputs, shape (50, 512, 512), 0 missing marker IDs, unlabeled_foreground ≈ 8949 (same seedless islands as geodesic).

- [ ] **Step 3: Commit**

```bash
git add scripts/experiment_cell_2d_graphcut.py tests/segmentation/test_graphcut.py
git commit -m "feat(graphcut): complete 2D alpha-expansion cell segmentation experiment"
```

---

## Notes

**Performance:** With 130 labels, 50 frames, 5 rounds, and ~262K foreground pixels per frame, each smoothness-weight run takes ~10–20 minutes. The `--crop` flag allows quick pilot runs. If full runs are too slow, reduce `--max-rounds` to 2.

**α-expansion simplification:** For pixel pairs where both have different non-α labels (Kolmogorov-Zabih case 3), we use a plain n-link instead of an auxiliary node. This omits a constant energy term that does not affect which assignment is optimal, so the label assignments are still energy-minimizing.

**Comparison baseline:** The best previous result is `tracked_labels_bw_128_alpha_0p25_tspace_10.tif` from the geodesic experiment. Load both in napari to compare elongated cell boundaries.
