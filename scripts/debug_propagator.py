"""Diagnostic script for propagate_next on frame 0.

Prints per-nucleus candidate counts and hypothesis frame counts, then opens
napari showing a random nucleus with its top-5 and worst-5 scoring candidates.
"""
import numpy as np
import napari
import tifffile
from scipy.spatial import KDTree

from cellflow.database.hypotheses import read_hypothesis_labels, list_hypotheses
from cellflow.database.tracked import read_tracked_frame
from cellflow.tracking.propagator import _label_stats, _nucleus_pixels, _iou_direct

DATA = (
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/v2/pos00"
)
HYPOTHESES_H5 = f"{DATA}/2_nucleus/hypotheses.h5"
TRACKED_TIF   = f"{DATA}/2_nucleus/tracked_labels.tif"
RAW_TIF       = f"{DATA}/0_input/nucleus_3dt.tif"

T_CURRENT     = 0
ION_THRESHOLD = 0.3
MAX_DIST_PX   = 50.0
VELOCITY_SIGMA_PX = 25.0

rng = np.random.default_rng(42)

# ── Load current frame ────────────────────────────────────────────────────────

current_labels = read_tracked_frame(TRACKED_TIF, T_CURRENT)
cur_areas, cur_centroids = _label_stats(current_labels)
cur_ids = sorted(cur_centroids.keys())
cur_pixels = _nucleus_pixels(current_labels)
H, W = current_labels.shape

print(f"Frame t={T_CURRENT}: {len(cur_ids)} tracked nuclei, image shape={H}x{W}")

# ── Load all hypothesis frames for t_next ─────────────────────────────────────

t_next = T_CURRENT + 1
n_p, params_by_p = list_hypotheses(HYPOTHESES_H5)
print(f"Hypotheses DB: {n_p} parameter sets")

entries: list[tuple[int, int, np.ndarray]] = []
for p in params_by_p.keys():
    try:
        volume = read_hypothesis_labels(HYPOTHESES_H5, t_next, p)  # (Z, Y, X)
    except (KeyError, ValueError):
        continue
    for z in range(volume.shape[0]):
        entries.append((p, z, volume[z]))

print(f"Total hypothesis frames for t={t_next}: {len(entries)}")

# ── Build flat candidate list ─────────────────────────────────────────────────

flat_cands: list[tuple[int, int, np.ndarray, int, np.ndarray]] = []
for entry_idx, (p, z, frame) in enumerate(entries):
    c_areas, c_centroids = _label_stats(frame)
    c_pixels = _nucleus_pixels(frame)
    for cid, centroid in c_centroids.items():
        cys, cxs = c_pixels[cid]
        flat_idx = cys * W + cxs
        flat_cands.append((entry_idx, int(cid), centroid, int(c_areas[cid]), flat_idx))

print(f"Total individual nucleus candidates: {len(flat_cands)}\n")

cand_centroids_arr = np.vstack([c[2] for c in flat_cands])
tree = KDTree(cand_centroids_arr)

# ── Per-nucleus diagnostics ───────────────────────────────────────────────────

print(f"{'nuc_id':>8}  {'centroid':>18}  {'dist_cands':>10}  {'iou_pass':>8}  {'best_score':>10}  {'best_entry':>10}")
print("-" * 80)

n_matched = 0
all_scored: dict[int, list[tuple[float, float, float, int, int, np.ndarray, int]]] = {}

for cur_id in cur_ids:
    cur_centroid = cur_centroids[cur_id]
    cur_area     = int(cur_areas[cur_id])
    cur_ys, cur_xs = cur_pixels[cur_id]

    nearby_ks = tree.query_ball_point(cur_centroid, MAX_DIST_PX)

    scored: list[tuple[float, float, float, int, int, np.ndarray, int]] = []
    for k in nearby_ks:
        entry_idx, cand_id, cand_centroid, cand_area, cand_flat_idx = flat_cands[k]
        iou = _iou_direct(cur_ys, cur_xs, cur_area, cand_flat_idx, cand_area, W)
        if iou < ION_THRESHOLD:
            continue
        area_ratio = min(cur_area, cand_area) / max(cur_area, cand_area)
        score = iou * area_ratio
        scored.append((score, iou, area_ratio, entry_idx, cand_id, cand_centroid, cand_area))

    scored.sort(key=lambda x: x[0], reverse=True)
    all_scored[cur_id] = scored

    best_score = scored[0][0] if scored else -1.0
    best_entry = scored[0][3] if scored else -1
    if scored:
        n_matched += 1

    centroid_str = f"({cur_centroid[0]:.1f},{cur_centroid[1]:.1f})"
    print(f"{cur_id:>8}  {centroid_str:>18}  {len(nearby_ks):>10}  {len(scored):>8}  "
          f"{best_score:>10.4f}  {best_entry:>10}")

print("-" * 80)
print(f"\nSummary: {n_matched}/{len(cur_ids)} nuclei matched at least one candidate "
      f"(iou_thresh={ION_THRESHOLD}, max_dist={MAX_DIST_PX}px)\n")

dist_counts  = [len(tree.query_ball_point(cur_centroids[cid], MAX_DIST_PX)) for cid in cur_ids]
iou_counts   = [len(all_scored[cid]) for cid in cur_ids]
print(f"Dist-gate candidates — min={min(dist_counts)}, max={max(dist_counts)}, mean={np.mean(dist_counts):.1f}")
print(f"IoU-pass candidates  — min={min(iou_counts)},  max={max(iou_counts)},  mean={np.mean(iou_counts):.1f}")

# ── Pick a random nucleus with enough candidates to show top/worst 5 ──────────

eligible = [cid for cid in cur_ids if len(all_scored[cid]) >= 10]
chosen_id = int(rng.choice(eligible))
scored = all_scored[chosen_id]
top5   = scored[:5]
worst5 = scored[-5:]

print(f"\nChosen nucleus: {chosen_id}  centroid={cur_centroids[chosen_id]}  "
      f"area={int(cur_areas[chosen_id])}  total_iou_pass={len(scored)}")
print(f"  Top-5   scores: {[round(s[0],4) for s in top5]}")
print(f"  Worst-5 scores: {[round(s[0],4) for s in worst5]}")

# ── Build display arrays ──────────────────────────────────────────────────────

def _crop_around(arr2d, centroid, pad=60):
    """Return a square crop and the (row_off, col_off) offset."""
    r, c = int(round(centroid[0])), int(round(centroid[1]))
    r0, r1 = max(0, r - pad), min(arr2d.shape[0], r + pad)
    c0, c1 = max(0, c - pad), min(arr2d.shape[1], c + pad)
    return arr2d[r0:r1, c0:c1], r0, c0


def _candidate_mask(entry_idx, cand_id):
    """Return a (H, W) bool mask for one candidate nucleus."""
    p, z, frame = entries[entry_idx]
    return (frame == cand_id).astype(np.uint8)


centroid = cur_centroids[chosen_id]
cur_mask_full = (current_labels == chosen_id).astype(np.uint8)

# Build stacks: (10, H, W) — top5 first, then worst5
n_show = 5
stack_labels = np.zeros((2 * n_show, H, W), dtype=np.uint8)
labels_list = []

for i, (score, iou, area_ratio, entry_idx, cand_id, cand_centroid, cand_area) in enumerate(top5):
    mask = _candidate_mask(entry_idx, cand_id)
    stack_labels[i] = mask
    p, z, _ = entries[entry_idx]
    labels_list.append(f"top{i+1} score={score:.4f} iou={iou:.3f} ar={area_ratio:.3f} p={p}")

for i, (score, iou, area_ratio, entry_idx, cand_id, cand_centroid, cand_area) in enumerate(worst5):
    mask = _candidate_mask(entry_idx, cand_id)
    stack_labels[n_show + i] = mask
    p, z, _ = entries[entry_idx]
    labels_list.append(f"worst{i+1} score={score:.4f} iou={iou:.3f} ar={area_ratio:.3f} p={p}")

# ── Load raw image if available ───────────────────────────────────────────────

raw_t0 = None
raw_t1 = None
try:
    raw_stack = tifffile.imread(RAW_TIF)  # (T, Z, Y, X)
    raw_t0 = raw_stack[T_CURRENT].mean(axis=0).astype(np.float32)
    raw_t1 = raw_stack[t_next].mean(axis=0).astype(np.float32)
    print("Raw image loaded.")
except Exception as e:
    print(f"Could not load raw image: {e}")

# ── Napari ────────────────────────────────────────────────────────────────────

viewer = napari.Viewer(title=f"Nucleus {chosen_id} — top5 vs worst5 candidates (t={T_CURRENT}→{t_next})")

if raw_t0 is not None:
    viewer.add_image(raw_t0, name="raw t0", colormap="gray", opacity=0.6)
    viewer.add_image(raw_t1, name="raw t1", colormap="gray", opacity=0.6, visible=False)

viewer.add_labels(current_labels, name="tracked t0", visible=True, opacity=0.4)

# Current nucleus highlighted
cur_highlight = np.zeros_like(current_labels)
cur_highlight[current_labels == chosen_id] = chosen_id
viewer.add_labels(cur_highlight, name=f"nucleus {chosen_id} (current)", opacity=0.8)

# Top 5 candidates (green-ish colormaps, one layer each)
colors_top   = ["green",  "lime",   "cyan",    "teal",   "chartreuse"]
colors_worst = ["red",    "orange", "yellow",  "coral",  "magenta"]

for i, (score, iou, area_ratio, entry_idx, cand_id, cand_centroid, cand_area) in enumerate(top5):
    mask = stack_labels[i]
    layer = viewer.add_labels(mask.astype(np.uint32), name=labels_list[i], opacity=0.6, visible=True)

for i, (score, iou, area_ratio, entry_idx, cand_id, cand_centroid, cand_area) in enumerate(worst5):
    mask = stack_labels[n_show + i]
    layer = viewer.add_labels(mask.astype(np.uint32), name=labels_list[n_show + i], opacity=0.6, visible=True)

# Zoom to the nucleus
pad = 80
r, c = int(round(centroid[0])), int(round(centroid[1]))
viewer.camera.center = (r, c)
viewer.camera.zoom = max(H, W) / (2 * pad)

napari.run()
