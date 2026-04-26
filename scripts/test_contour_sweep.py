"""Prototype: unseeded watershed sweep on consensus boundary from Cellpose cellprob sweep.

Builds the consensus boundary for frame 0 by sweeping cellprob_threshold over
[-3.0, -2.0, -1.0, 0.0] and averaging find_boundaries over all (threshold × z).

Then runs an unseeded watershed sweep over seed_distance in [8, 10, 12, 14],
using peak_local_max on the foreground probability to place markers.
"""
import numpy as np
import tifffile
import napari
import torch
from cellpose.dynamics import compute_masks
from scipy.ndimage import label as nd_label
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries, watershed

DATA = (
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/v2/pos00"
)
CELLPOSE_DIR = f"{DATA}/1_cellpose"

THRESHOLDS     = np.arange(-3.0, 1.0, 1.0)   # -3.0, -2.0, -1.0, 0.0
SEED_DISTANCES = range(8, 15, 2)              # 8, 10, 12, 14

T = 0

prob_stack = tifffile.imread(f"{CELLPOSE_DIR}/nucleus_prob_3dt.tif")  # (T, Z, Y, X)
dp_stack   = tifffile.imread(f"{CELLPOSE_DIR}/nucleus_dp_3dt.tif")    # (T, Z, 2, Y, X)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Consensus boundary ────────────────────────────────────────────────────────

prob = prob_stack[T]  # (Z, Y, X)
dp   = dp_stack[T]    # (Z, 2, Y, X)
n_z  = prob.shape[0]

print(f"\nBuilding consensus boundary for frame {T} "
      f"({len(THRESHOLDS)} thresholds × {n_z} z-slices)...")

accum   = np.zeros(prob.shape[1:], dtype=np.float32)
n_total = 0
for thresh in THRESHOLDS:
    print(f"  thresh={thresh:.1f}", flush=True)
    for z in range(n_z):
        result = compute_masks(
            dp[z], prob[z],
            cellprob_threshold=float(thresh),
            flow_threshold=0.0,
            niter=200,
            do_3D=False,
            device=device,
        )
        masks = result[0] if isinstance(result, tuple) else result
        accum += find_boundaries(np.asarray(masks), mode="inner").astype(np.float32)
        n_total += 1

boundary = accum / n_total
print(f"Consensus boundary range=[{boundary.min():.4f}, {boundary.max():.4f}]")

# Foreground: sigmoid of z-averaged prob logits
foreground = 1.0 / (1.0 + np.exp(-prob.mean(axis=0)))
fg_mask    = foreground > 0.5

# ── Unseeded watershed sweep ──────────────────────────────────────────────────

results = {}
for d in SEED_DISTANCES:
    coords  = peak_local_max(foreground, min_distance=d, threshold_abs=0.5, exclude_border=False)
    mask_pts = np.zeros(foreground.shape, dtype=bool)
    mask_pts[coords[:, 0], coords[:, 1]] = True
    markers, _ = nd_label(mask_pts)

    labels = watershed(boundary, markers=markers, mask=fg_mask, watershed_line=True)
    n_cells = int(labels.max())
    print(f"  seed_distance={d:2d}  seeds={markers.max():3d}  cells={n_cells:3d}")
    results[d] = labels

# ── Napari ────────────────────────────────────────────────────────────────────

raw_nuc = tifffile.imread(f"{DATA}/0_input/nucleus_3dt.tif")  # (T, Z, Y, X)

viewer = napari.Viewer(title=f"Unseeded watershed sweep — frame {T}")
viewer.add_image(raw_nuc[T].mean(axis=0), name="nucleus z-avg", colormap="gray")
viewer.add_image(boundary,   name="consensus boundary", colormap="magma")
viewer.add_image(foreground, name="foreground prob",    colormap="inferno", visible=False)

for d, labels in results.items():
    viewer.add_labels(labels, name=f"watershed seed_dist={d}")

napari.run()
