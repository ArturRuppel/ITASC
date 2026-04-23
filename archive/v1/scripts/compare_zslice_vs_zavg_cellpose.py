"""Compare per-z-slice vs z-averaged Cellpose on pos01, frame 0.

Runs Cellpose on each individual z-slice of the cell channel, then compares:
  1. Flow magnitude per z-slice
  2. Mean of per-slice flow magnitudes
  3. Flow magnitude when Cellpose is applied to the already z-averaged image

Usage:
    conda run -n cellflow python scripts/compare_zslice_vs_zavg_cellpose.py
"""

from __future__ import annotations

import numpy as np
import tifffile
from ndtiff import Dataset
from skimage.transform import downscale_local_mean

DATA_ROOT = (
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/analysis/pos01"
)
NDTIFF_PATH = (
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/_1"
)
POS = 1
FRAME = 0
XY_DOWNSAMPLE = 2
GAMMA = 0.5

# Channel indices in the NDTiff dataset
# ChNames: ['CSUTRANS', 'CSU405 ', 'CSU488', 'CSU561']
CH_488 = 2   # membrane/cell marker
CH_405 = 1   # nuclear marker


def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma-correct each channel (H, W, C) in place, returning float32."""
    out = img.astype(np.float32)
    for c in range(out.shape[2]):
        ch = out[:, :, c]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            ch_norm = (ch - ch_min) / (ch_max - ch_min)
            out[:, :, c] = (ch_norm ** gamma) * (ch_max - ch_min) + ch_min
    return out


def _flow_mag(dp: np.ndarray) -> np.ndarray:
    """dp: (2, H, W) → flow_mag: (H, W) float32."""
    flow = np.transpose(dp, (1, 2, 0)).astype(np.float32)  # (H, W, 2)
    return np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)


def load_cell_zstack(ds: Dataset, pos: int, t: int, z_indices: list[int]) -> np.ndarray:
    """Load 488-channel z-stack from NDTiff, xy-downsampled → (Z, H, W) uint16."""
    slices = []
    for z in z_indices:
        img = ds.read_image(position=pos, time=t, channel=CH_488, z=z)
        if img is None:
            img = np.zeros((ds.image_height, ds.image_width), dtype=np.uint16)
        slices.append(img)
    volume = np.stack(slices, axis=0)  # (Z, H_raw, W_raw)
    if XY_DOWNSAMPLE > 1:
        volume = downscale_local_mean(volume, (1, XY_DOWNSAMPLE, XY_DOWNSAMPLE)).astype(np.uint16)
    return volume


def load_nucleus_zstack(t: int) -> np.ndarray:
    """Load nucleus_4d.tif and return frame *t* as a (Z, H, W) array."""
    path = f"{DATA_ROOT}/0_input/nucleus_4d.tif"
    return tifffile.imread(path)[t]  # (Z, H, W)


def run_cellpose_slice(model, cell_slice: np.ndarray, nuc_slice: np.ndarray, label: str):
    """Run Cellpose on one (H, W) pair → returns (dp, prob, flow_mag)."""
    img = np.stack([cell_slice, nuc_slice], axis=-1).astype(np.float32)  # (H, W, 2)
    if GAMMA is not None and GAMMA != 1.0:
        img = _apply_gamma(img, GAMMA)
    print(f"  {label} ...", end="", flush=True)
    _, flows, _ = model.eval(img, diameter=None, min_size=0)
    dp = flows[1].astype(np.float32)   # (2, H, W)
    prob = flows[2].astype(np.float32)  # (H, W)
    mag = _flow_mag(dp)
    print("  done", flush=True)
    return dp, prob, mag


def main():
    from cellpose.models import CellposeModel

    import torch
    gpu = torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0) if gpu else 'none'}")

    model = CellposeModel(gpu=gpu, pretrained_model="cpsam")
    print("Model 'cpsam' loaded.")

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"\nLoading pos{POS:02d} frame {FRAME} z-stack from NDTiff…")
    ds = Dataset(NDTIFF_PATH)
    z_indices = sorted(ds.axes.get("z", [0]))
    print(f"  z-indices: {z_indices}")

    cell_z = load_cell_zstack(ds, POS, FRAME, z_indices)   # (Z, H, W)
    nuc_z  = load_nucleus_zstack(FRAME)                     # (Z, H, W)
    print(f"  cell z-stack: {cell_z.shape} {cell_z.dtype}")
    print(f"  nuc  z-stack: {nuc_z.shape} {nuc_z.dtype}")

    cell_zavg = cell_z.mean(axis=0).astype(np.uint16)   # (H, W)
    nuc_zavg  = nuc_z.mean(axis=0).astype(np.uint16)    # (H, W)

    # ── Per-slice Cellpose ────────────────────────────────────────────────────
    Z = cell_z.shape[0]
    slice_dp_list:  list[np.ndarray] = []
    slice_mag_list: list[np.ndarray] = []
    slice_prob_list: list[np.ndarray] = []

    print(f"\nRunning Cellpose on {Z} z-slices…")
    for z in range(Z):
        dp, prob, mag = run_cellpose_slice(
            model, cell_z[z], nuc_z[z], label=f"z{z:02d}"
        )
        slice_dp_list.append(dp)
        slice_mag_list.append(mag)
        slice_prob_list.append(prob)

    slice_dp   = np.stack(slice_dp_list,  axis=0)   # (Z, 2, H, W)
    slice_mag  = np.stack(slice_mag_list, axis=0)   # (Z, H, W)
    slice_prob = np.stack(slice_prob_list, axis=0)  # (Z, H, W)
    mean_mag   = slice_mag.mean(axis=0)              # (H, W)

    print(f"\nPer-slice flow_mag range: [{slice_mag.min():.3f}, {slice_mag.max():.3f}]")
    print(f"Mean flow_mag range:      [{mean_mag.min():.3f}, {mean_mag.max():.3f}]")

    # ── Z-averaged Cellpose ───────────────────────────────────────────────────
    print("\nRunning Cellpose on z-averaged image…")
    dp_zavg, prob_zavg, mag_zavg = run_cellpose_slice(
        model, cell_zavg, nuc_zavg, label="zavg"
    )
    print(f"zavg flow_mag range: [{mag_zavg.min():.3f}, {mag_zavg.max():.3f}]")

    del model
    if gpu:
        torch.cuda.empty_cache()

    # ── Napari display ────────────────────────────────────────────────────────
    print("\nLaunching napari…")
    import napari

    viewer = napari.Viewer(
        title=f"Cellpose per-slice vs z-avg — pos{POS:02d} frame {FRAME}"
    )

    # Raw images for context
    viewer.add_image(
        cell_z.astype(np.float32),
        name="cell z-stack (488)",
        colormap="gray",
        visible=True,
    )
    viewer.add_image(
        nuc_z.astype(np.float32),
        name="nuc z-stack (405)",
        colormap="cyan",
        visible=False,
    )

    # Per-slice flow magnitudes (Z, H, W) — browsable as a stack
    viewer.add_image(
        slice_mag,
        name="flow mag per z-slice",
        colormap="magma",
        visible=True,
    )

    # Average of per-slice flow magnitudes
    viewer.add_image(
        mean_mag,
        name="mean of per-slice flow mags",
        colormap="magma",
        visible=True,
    )

    # Flow magnitude from z-averaged input
    viewer.add_image(
        mag_zavg,
        name="flow mag (z-avg input)",
        colormap="magma",
        visible=True,
    )

    # Difference: mean-per-slice vs z-avg
    diff = mean_mag - mag_zavg
    viewer.add_image(
        diff,
        name="diff: mean-per-slice minus z-avg",
        colormap="RdBu",
        visible=False,
    )

    # Cell probability maps for reference
    viewer.add_image(
        slice_prob,
        name="cell prob per z-slice",
        colormap="gray",
        visible=False,
    )
    viewer.add_image(
        prob_zavg,
        name="cell prob (z-avg input)",
        colormap="gray",
        visible=False,
    )

    napari.run()


if __name__ == "__main__":
    main()
