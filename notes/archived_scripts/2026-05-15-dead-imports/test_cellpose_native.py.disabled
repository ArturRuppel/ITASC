"""Quick test of Cellpose native flow-based segmentation on one z-slice."""
import numpy as np
import tifffile
import napari

from cellflow.segmentation import CellposeFlowHypothesisParams, compute_cellpose_flow_hypothesis

DATA = "/home/aruppel/Data/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/v2/pos00/1_cellpose"

T, Z = 0, 3

prob_stack = tifffile.imread(f"{DATA}/nucleus_prob_3dt.tif")  # (T, Z, Y, X)
dp_stack   = tifffile.imread(f"{DATA}/nucleus_dp_3dt.tif")    # (T, Z, 2, Y, X)

prob_slice = prob_stack[T, Z:Z+1]   # (1, Y, X)
dp_slice   = dp_stack[T, Z:Z+1]     # (1, 2, Y, X)

print(f"prob  shape={prob_slice.shape}  range=[{prob_slice.min():.2f}, {prob_slice.max():.2f}]")
print(f"dp    shape={dp_slice.shape}    range=[{dp_slice.min():.2f}, {dp_slice.max():.2f}]")
print(f"foreground pixels (prob>0): {int((prob_slice > 0).sum())}")

params = CellposeFlowHypothesisParams(
    cellprob_threshold=0.0,
    flow_threshold=0.0,
    min_size=15,
    niter=200,
)

print("Running compute_cellpose_flow_hypothesis...")
labels = compute_cellpose_flow_hypothesis(prob_slice, dp_slice, params)  # (1, Y, X)
print(f"labels shape={labels.shape}  n_cells={int(labels.max())}")

flow_mag = np.sqrt(dp_slice[0, 0] ** 2 + dp_slice[0, 1] ** 2)  # (Y, X)

viewer = napari.Viewer(title=f"Cellpose native — t={T} z={Z}")
viewer.add_image(prob_slice[0],  name="cell prob",   colormap="inferno")
viewer.add_image(flow_mag,       name="flow mag",    colormap="viridis",  blending="additive")
viewer.add_labels(labels[0],     name="masks")
napari.run()
