"""Experiment: derive nucleus foreground + contour maps directly from Cellpose
prob/dp without the (threshold x z) mask sweep.

Hypotheses being tested
-----------------------
1. ``foreground_scores`` (currently the mean of ``masks > 0`` over thresholds and
   z) is essentially ``sigmoid(prob)`` reduced over z. If that holds, we can
   skip the sweep entirely for the foreground signal.

2. The contour map (currently the mean of ``find_boundaries(masks)`` over the
   same sweep) can be replaced by the **positive part of the divergence of the
   Cellpose flow field**. Inside a cell the dp field converges toward the
   center -> div < 0. Across the boundary between two touching cells the field
   flips direction -> a "source" with div > 0. Numerically that should mark the
   ridge that separates cells.

Outputs (next to the existing per-position 2_nucleus/ artifacts):
    .../2_nucleus/divergence_experiment/
        foreground_sigmoid.tif        T x Y x X float32
        contour_divergence.tif        T x Y x X float32
        panels/frame_{t:03d}.png      side-by-side comparison
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

POS_DIR = Path(
    "/home/aruppel/Data/"
    "2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/"
    "analysis/pos00"
)
CELLPOSE_DIR = POS_DIR / "1_cellpose"
NUC_DIR = POS_DIR / "2_nucleus"
OUT_DIR = NUC_DIR / "divergence_experiment"
PANEL_DIR = OUT_DIR / "panels"

# Reduction across z for the foreground signal: "mean" or "max".
Z_REDUCTION_FG = "mean"
# Contour: compute both max-z and mean-z divergence so they can be compared.

# Smoothing applied to dp before taking the divergence (pixels). Cellpose flows
# are noisy in the background; a small Gaussian blur stabilizes the numerical
# derivative without much effect on real ridges.
DP_SMOOTH_SIGMA = 1.0

# Frames to render side-by-side panels for.
PANEL_FRAMES = (0, 12, 24, 36, 47)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def foreground_from_prob(prob_tzyx: np.ndarray, reduction: str) -> np.ndarray:
    """sigmoid(prob) reduced across z. Shape (T, Y, X)."""
    p = _sigmoid(prob_tzyx.astype(np.float32))
    if reduction == "max":
        return p.max(axis=1)
    return p.mean(axis=1)


def divergence_2d(flow_yx: np.ndarray) -> np.ndarray:
    """Divergence of a (2, Y, X) flow field with channels [dy, dx]."""
    dy = flow_yx[0]
    dx = flow_yx[1]
    d_dy = np.gradient(dy, axis=0)
    d_dx = np.gradient(dx, axis=1)
    return (d_dy + d_dx).astype(np.float32)


def contour_from_dp(
    dp_tzcyx: np.ndarray,
    *,
    smooth_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per (t, z) positive divergence of the flow, reduced across z.

    Returns (max_z, mean_z) stacks, each (T, Y, X) float32.
    """
    from scipy.ndimage import gaussian_filter

    n_t, n_z, _, n_y, n_x = dp_tzcyx.shape
    out_max = np.zeros((n_t, n_y, n_x), dtype=np.float32)
    out_mean = np.zeros((n_t, n_y, n_x), dtype=np.float32)
    for t in range(n_t):
        per_z = np.empty((n_z, n_y, n_x), dtype=np.float32)
        for z in range(n_z):
            flow = dp_tzcyx[t, z].astype(np.float32)
            if smooth_sigma > 0:
                flow = np.stack(
                    [gaussian_filter(flow[0], smooth_sigma),
                     gaussian_filter(flow[1], smooth_sigma)],
                    axis=0,
                )
            div = divergence_2d(flow)
            per_z[z] = np.clip(div, 0.0, None)
        out_max[t] = per_z.max(axis=0)
        out_mean[t] = per_z.mean(axis=0)
    return out_max, out_mean


def _norm01(arr: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    lo = np.percentile(arr, lo_pct)
    hi = np.percentile(arr, hi_pct)
    if hi <= lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def render_panel(
    t: int,
    fg_existing: np.ndarray,
    fg_new: np.ndarray,
    contour_existing: np.ndarray,
    contour_max: np.ndarray,
    contour_mean: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(fg_existing, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("foreground_scores (existing)")
    axes[0, 1].imshow(_norm01(fg_new), cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title(f"sigmoid(prob).{Z_REDUCTION_FG}(z) (proposed)")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(contour_existing, cmap="magma", vmin=0, vmax=1)
    axes[1, 0].set_title("contour_maps (existing)")
    axes[1, 1].imshow(_norm01(contour_max), cmap="magma", vmin=0, vmax=1)
    axes[1, 1].set_title("div(dp)+ * fg, max(z)")
    axes[1, 2].imshow(_norm01(contour_mean), cmap="magma", vmin=0, vmax=1)
    axes[1, 2].set_title("div(dp)+ * fg, mean(z)")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"frame {t}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading prob and dp stacks...")
    prob_stack = tifffile.imread(str(CELLPOSE_DIR / "nucleus_prob_3dt.tif"))  # (T,Z,Y,X)
    dp_stack = tifffile.imread(str(CELLPOSE_DIR / "nucleus_dp_3dt.tif"))      # (T,Z,2,Y,X)
    print(f"  prob {prob_stack.shape} {prob_stack.dtype}")
    print(f"  dp   {dp_stack.shape} {dp_stack.dtype}")

    fg_existing_stack = tifffile.imread(str(NUC_DIR / "foreground_scores.tif"))
    contour_existing_stack = tifffile.imread(str(NUC_DIR / "contour_maps.tif"))

    print("Computing proposed foreground (sigmoid + z-reduce)...")
    fg_new_stack = foreground_from_prob(prob_stack, reduction=Z_REDUCTION_FG)
    print("Computing proposed contour (positive divergence of dp, max & mean over z)...")
    contour_max_raw, contour_mean_raw = contour_from_dp(
        dp_stack, smooth_sigma=DP_SMOOTH_SIGMA
    )
    # Background regions have noisy flow vectors that produce spurious
    # divergence; suppress them by gating on the foreground signal itself.
    contour_max_stack = contour_max_raw * fg_new_stack
    contour_mean_stack = contour_mean_raw * fg_new_stack

    print("Writing TIFF stacks...")
    tifffile.imwrite(
        str(OUT_DIR / "foreground_sigmoid.tif"),
        fg_new_stack.astype(np.float32),
        compression="zlib",
    )
    tifffile.imwrite(
        str(OUT_DIR / "contour_divergence_max.tif"),
        contour_max_stack.astype(np.float32),
        compression="zlib",
    )
    tifffile.imwrite(
        str(OUT_DIR / "contour_divergence_mean.tif"),
        contour_mean_stack.astype(np.float32),
        compression="zlib",
    )

    print("Correlations vs existing maps:")
    for name, a, b in [
        ("foreground", fg_existing_stack, fg_new_stack),
        ("contour (max-z)", contour_existing_stack, contour_max_stack),
        ("contour (mean-z)", contour_existing_stack, contour_mean_stack),
    ]:
        a_flat = a.reshape(-1).astype(np.float64)
        b_flat = b.reshape(-1).astype(np.float64)
        r = np.corrcoef(a_flat, b_flat)[0, 1]
        print(f"  {name}: pearson r = {r:.4f}")

    print("Rendering panels for frames:", PANEL_FRAMES)
    for t in PANEL_FRAMES:
        render_panel(
            t,
            fg_existing_stack[t],
            fg_new_stack[t],
            contour_existing_stack[t],
            contour_max_stack[t],
            contour_mean_stack[t],
            PANEL_DIR / f"frame_{t:03d}.png",
        )
    print(f"Done. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
