"""Compare watershed input images: cellpose probability vs flow magnitude.

Tests 4 variants on pos02 frame 45, all with compactness=0:
  1. prob,     no smoothing  (sigma=0)
  2. prob,     medium smoothing (sigma=2)
  3. flow mag, no smoothing  (sigma=0)
  4. flow mag, medium smoothing (sigma=2)

Usage:
    conda run -n cellflow python scripts/compare_watershed_inputs.py
"""

from __future__ import annotations

import time

import numpy as np
import tifffile
from scipy import ndimage
from skimage.segmentation import watershed

DATA_ROOT = (
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/analysis/pos02"
)
FRAME = 45
PROB_THRESHOLD = 0.0
SMOOTHING_SIGMA = 2.0


def load_frame(frame: int):
    print(f"Loading frame {frame} from {DATA_ROOT}…")
    nuc_stack  = tifffile.imread(f"{DATA_ROOT}/3_correction/nuclear_labels_corrected.tif")
    dp_stack   = tifffile.imread(f"{DATA_ROOT}/1_cellpose/cell_dp.tif")
    prob_stack = tifffile.imread(f"{DATA_ROOT}/1_cellpose/cell_prob.tif")

    print(f"  nuclear_labels_corrected: {nuc_stack.shape} {nuc_stack.dtype}")
    print(f"  cell_dp:  {dp_stack.shape}  range [{dp_stack.min():.2f}, {dp_stack.max():.2f}]")
    print(f"  cell_prob: {prob_stack.shape} range [{prob_stack.min():.2f}, {prob_stack.max():.2f}]")

    nuc  = nuc_stack[frame].astype(np.int32)
    dp   = dp_stack[frame]   # (2, H, W)
    prob = prob_stack[frame]  # (H, W) raw logits

    flow = np.transpose(dp, (1, 2, 0)).astype(np.float32)  # (H, W, 2)
    prob_sigmoid = (1.0 / (1.0 + np.exp(-prob))).astype(np.float32)
    flow_mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)

    print(f"  prob_sigmoid: range [{prob_sigmoid.min():.3f}, {prob_sigmoid.max():.3f}]")
    print(f"  flow_mag:     range [{flow_mag.min():.3f}, {flow_mag.max():.3f}]")

    return nuc, flow, prob_sigmoid, flow_mag


def run_watershed(nuc, basin_image, prob_threshold=PROB_THRESHOLD, sigma=0.0, label=""):
    img = basin_image.copy()
    if sigma > 0.0:
        img = ndimage.gaussian_filter(img, sigma=sigma)
    mask = (img >= prob_threshold) | (nuc > 0)
    t0 = time.perf_counter()
    result = watershed(-img, markers=nuc.astype(np.int32), mask=mask, compactness=0.0)
    elapsed = time.perf_counter() - t0
    n_cells = np.unique(result[result > 0]).size
    print(f"  {label:45s}  {elapsed:.2f}s  |  cells: {n_cells}")
    return result.astype(np.int32)


def main():
    nuc, flow, prob, flow_mag = load_frame(FRAME)

    print(f"\nRunning watershed variants (frame {FRAME}, compactness=0)…")
    ws_prob_raw    = run_watershed(nuc, prob,     sigma=0.0,            label=f"prob,     sigma=0")
    ws_prob_smooth = run_watershed(nuc, prob,     sigma=SMOOTHING_SIGMA, label=f"prob,     sigma={SMOOTHING_SIGMA}")
    ws_mag_raw     = run_watershed(nuc, flow_mag, sigma=0.0,            label=f"flow mag, sigma=0")
    ws_mag_smooth  = run_watershed(nuc, flow_mag, sigma=SMOOTHING_SIGMA, label=f"flow mag, sigma={SMOOTHING_SIGMA}")

    print("\nLaunching napari…")
    import napari

    viewer = napari.Viewer(title=f"Watershed input comparison — pos02 frame {FRAME}")

    viewer.add_labels(nuc,            name="nuclear seeds",              opacity=0.7)
    viewer.add_labels(ws_prob_raw,    name="prob | sigma=0",             opacity=0.5)
    viewer.add_labels(ws_prob_smooth, name=f"prob | sigma={SMOOTHING_SIGMA}", opacity=0.5, visible=False)
    viewer.add_labels(ws_mag_raw,     name="flow mag | sigma=0",         opacity=0.5, visible=False)
    viewer.add_labels(ws_mag_smooth,  name=f"flow mag | sigma={SMOOTHING_SIGMA}", opacity=0.5, visible=False)

    viewer.add_image(prob,                          name="cell prob (sigmoid)",       colormap="gray",  opacity=0.5, visible=False)
    viewer.add_image(ndimage.gaussian_filter(prob,  sigma=SMOOTHING_SIGMA),
                                                    name=f"cell prob smoothed σ={SMOOTHING_SIGMA}", colormap="gray",  opacity=0.5, visible=False)
    viewer.add_image(flow_mag,                      name="flow magnitude",            colormap="magma", opacity=0.5, visible=False)
    viewer.add_image(ndimage.gaussian_filter(flow_mag, sigma=SMOOTHING_SIGMA),
                                                    name=f"flow mag smoothed σ={SMOOTHING_SIGMA}",  colormap="magma", opacity=0.5, visible=False)

    napari.run()


if __name__ == "__main__":
    main()
