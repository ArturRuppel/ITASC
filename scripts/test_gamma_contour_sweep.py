"""Prototype: unseeded watershed sweep on consensus boundary with gamma correction on cellprob.

Builds the consensus boundary for frame 0 by sweeping cellprob_threshold over
[-3.0, -2.0, -1.0, 0.0] and averaging find_boundaries over all (threshold × z).
This is repeated for different gamma values applied to the cellprob logits.

Then runs an unseeded watershed using peak_local_max on the (gamma-corrected) 
foreground probability to place markers.
"""
import numpy as np
import tifffile
import napari
import torch
from cellpose.dynamics import compute_masks
from scipy.ndimage import label as nd_label
from skimage.feature import peak_local_max
from skimage.segmentation import find_boundaries, watershed

def apply_gamma(logits, gamma):
    """Applies gamma correction to logits by converting to probability space and back."""
    if gamma == 1.0:
        return logits
    # Sigmoid to get probabilities
    probs = 1.0 / (1.0 + np.exp(-logits))
    # Apply gamma
    probs_gamma = np.power(probs, gamma)
    # avoid log(0) and log(inf)
    probs_gamma = np.clip(probs_gamma, 1e-7, 1 - 1e-7)
    # Logit to get back to logit space
    logits_gamma = np.log(probs_gamma / (1.0 - probs_gamma))
    return logits_gamma

DATA = (
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/v2/pos00"
)
CELLPOSE_DIR = f"{DATA}/1_cellpose"

THRESHOLDS     = np.arange(-3.0, 1.0, 1.0)   # -3.0, -2.0, -1.0, 0.0
GAMMAS         = [0.2, 0.5, 1.0, 2.0]
SEED_DISTANCE  = 12

T = 0

print(f"Reading data from {CELLPOSE_DIR}...")
prob_stack = tifffile.imread(f"{CELLPOSE_DIR}/nucleus_prob_3dt.tif")  # (T, Z, Y, X)
dp_stack   = tifffile.imread(f"{CELLPOSE_DIR}/nucleus_dp_3dt.tif")    # (T, Z, 2, Y, X)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

prob = prob_stack[T]  # (Z, Y, X)
dp   = dp_stack[T]    # (Z, 2, Y, X)
n_z  = prob.shape[0]

viewer = napari.Viewer(title=f"Gamma sweep — frame {T}")
raw_nuc = tifffile.imread(f"{DATA}/0_input/nucleus_3dt.tif")  # (T, Z, Y, X)
viewer.add_image(raw_nuc[T].mean(axis=0), name="nucleus z-avg", colormap="gray")

for gamma in GAMMAS:
    print(f"\nProcessing gamma={gamma}...")
    
    # Apply gamma to the whole stack
    prob_g = apply_gamma(prob, gamma)
    
    # ── Consensus boundary ────────────────────────────────────────────────────────
    print(f"  Building consensus boundary ({len(THRESHOLDS)} thresholds × {n_z} z-slices)...")
    accum   = np.zeros(prob.shape[1:], dtype=np.float32)
    n_total = 0
    for thresh in THRESHOLDS:
        print(f"    thresh={thresh:.1f}", flush=True)
        for z in range(n_z):
            result = compute_masks(
                dp[z], prob_g[z],
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
    print(f"  Consensus boundary range=[{boundary.min():.4f}, {boundary.max():.4f}]")
    
    # Foreground: sigmoid of z-averaged prob logits (using gamma-corrected ones)
    foreground = 1.0 / (1.0 + np.exp(-prob_g.mean(axis=0)))
    fg_mask    = foreground > 0.5
    
    # ── Watershed ────────────────────────────────────────────────────────────────
    coords  = peak_local_max(foreground, min_distance=SEED_DISTANCE, threshold_abs=0.5, exclude_border=False)
    mask_pts = np.zeros(foreground.shape, dtype=bool)
    mask_pts[coords[:, 0], coords[:, 1]] = True
    markers, _ = nd_label(mask_pts)

    labels = watershed(boundary, markers=markers, mask=fg_mask, watershed_line=True)
    n_cells = int(labels.max())
    print(f"  gamma={gamma:.1f}  seeds={markers.max():3d}  cells={n_cells:3d}")

    # ── Add to Napari ────────────────────────────────────────────────────────────
    viewer.add_image(boundary,   name=f"boundary g={gamma}", colormap="magma", visible=(gamma==1.0))
    viewer.add_image(foreground, name=f"foreground g={gamma}", colormap="inferno", visible=False)
    viewer.add_labels(labels,    name=f"watershed g={gamma}", visible=(gamma==1.0))

print("\nLaunching Napari...")
napari.run()
