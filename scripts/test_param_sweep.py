"""Sweep gamma (0.2–1.0), cellprob threshold (-5.0–5.0), and boundary blur (0.0–5.0)
independently over the first frame, displaying all results in napari.

Each sweep holds the other two parameters at their defaults:
  DEFAULT_GAMMA=0.5, DEFAULT_THRESHOLD=0.0, DEFAULT_BLUR=0.0
"""
import numpy as np
import tifffile
import napari
from scipy.ndimage import gaussian_filter

from cellflow.segmentation import build_consensus_boundary

DATA = (
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/v2/pos00"
)
CELLPOSE_DIR = f"{DATA}/1_cellpose"

T = 0

GAMMAS     = np.linspace(0.2, 1.0, 5)   # [0.20, 0.40, 0.60, 0.80, 1.00]
THRESHOLDS = np.linspace(-5.0, 5.0, 5)  # [-5.0, -2.5,  0.0,  2.5,  5.0]
BLURS      = np.linspace(0.0, 5.0, 5)   # [0.00, 1.25,  2.50, 3.75, 5.00]

DEFAULT_GAMMA     = 0.5
DEFAULT_THRESHOLD = 0.0
DEFAULT_BLUR      = 0.0
SEED_DISTANCE     = 12


def build_and_collect(
    prob_3d: np.ndarray,
    dp_3d: np.ndarray,
    gamma: float,
    threshold: float,
    blur_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (boundary, foreground, source_masks_zyx)."""
    collected: list[np.ndarray] = []

    def _cb(masks_zyx: np.ndarray, _thresh_idx: int) -> None:
        collected.append(masks_zyx)

    boundary, foreground = build_consensus_boundary(
        prob_3d, dp_3d, [threshold], gamma=gamma, mask_callback=_cb
    )
    if blur_sigma > 0:
        boundary = gaussian_filter(boundary, sigma=float(blur_sigma))
    # collected[0] is (Z, Y, X) uint32 for the single threshold
    source_masks = collected[0] if collected else np.zeros_like(prob_3d, dtype=np.uint32)
    return boundary, foreground, source_masks


print(f"Reading data from {CELLPOSE_DIR}...")
prob_stack = tifffile.imread(f"{CELLPOSE_DIR}/nucleus_prob_3dt.tif")
dp_stack   = tifffile.imread(f"{CELLPOSE_DIR}/nucleus_dp_3dt.tif")

prob = np.asarray(prob_stack[T], dtype=np.float32)
dp   = np.asarray(dp_stack[T],   dtype=np.float32)
print(f"Frame {T}: prob={prob.shape}, dp={dp.shape}")

raw_nuc = tifffile.imread(f"{DATA}/0_input/nucleus_3dt.tif")

viewer = napari.Viewer(title=f"Parameter sweep — frame {T}")
viewer.add_image(
    raw_nuc[T].mean(axis=0).astype(np.float32),
    name="nucleus z-avg",
    colormap="gray",
)

# ── Gamma sweep (threshold=DEFAULT_THRESHOLD, blur=DEFAULT_BLUR) ──────────────
print(f"\nGamma sweep (threshold={DEFAULT_THRESHOLD}, blur={DEFAULT_BLUR})...")
for gamma in GAMMAS:
    print(f"  gamma={gamma:.2f}", flush=True)
    boundary, _, source = build_and_collect(prob, dp, gamma, DEFAULT_THRESHOLD, DEFAULT_BLUR)
    viewer.add_image(boundary, name=f"boundary  g={gamma:.2f}", colormap="magma", visible=False)
    viewer.add_labels(source,  name=f"masks     g={gamma:.2f}",                   visible=False)

# ── Threshold sweep (gamma=DEFAULT_GAMMA, blur=DEFAULT_BLUR) ─────────────────
print(f"\nThreshold sweep (gamma={DEFAULT_GAMMA}, blur={DEFAULT_BLUR})...")
for thresh in THRESHOLDS:
    print(f"  thresh={thresh:.1f}", flush=True)
    boundary, _, source = build_and_collect(prob, dp, DEFAULT_GAMMA, thresh, DEFAULT_BLUR)
    viewer.add_image(boundary, name=f"boundary  thr={thresh:+.1f}", colormap="magma", visible=False)
    viewer.add_labels(source,  name=f"masks     thr={thresh:+.1f}",                   visible=False)

# ── Blur sweep (gamma=DEFAULT_GAMMA, threshold=DEFAULT_THRESHOLD) ─────────────
print(f"\nBlur sweep (gamma={DEFAULT_GAMMA}, threshold={DEFAULT_THRESHOLD})...")
for blur in BLURS:
    print(f"  blur={blur:.2f}", flush=True)
    boundary, _, source = build_and_collect(prob, dp, DEFAULT_GAMMA, DEFAULT_THRESHOLD, blur)
    viewer.add_image(boundary, name=f"boundary  blur={blur:.2f}", colormap="magma", visible=False)
    viewer.add_labels(source,  name=f"masks     blur={blur:.2f}",                   visible=False)

print("\nLaunching napari...")
napari.run()
