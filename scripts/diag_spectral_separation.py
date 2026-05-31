#!/usr/bin/env python
"""Diagnostic: does nucleus-vs-background separation live in the spectrum?

We keep failing to separate faint nuclei from background with intensity-based
methods because faint nuclei and bright background overlap in absolute value.
The eye succeeds because it reads *local structure at a characteristic scale*
and is invariant to contrast. This script tests, before we commit to any method,
whether that separation is actually recoverable from spectral / phase features.

It does three things, on both the contour map and the foreground map:

1. Radially-averaged power spectrum of nucleus patches vs background patches.
   If nuclei dominate some frequency band, a scale-tuned filter will help; if
   the curves overlap everywhere, a magnitude band-pass cannot separate them.

2. Builds candidate feature maps:
     - raw                  (baseline)
     - loggabor_energy      multi-scale log-Gabor band energy (contrast-dependent)
     - phase_symmetry       monogenic phase symmetry, amplitude-normalized with a
                            noise floor -> contrast-INVARIANT blob/ridge response
   shown in napari for visual inspection.

3. Separability: for each feature, the AUC of separating inside-nucleus pixels
   from background pixels (with a margin ring excluded), reported overall and for
   faint nuclei only. AUC=1 perfect, 0.5 useless. The feature with the best
   *faint* AUC is the one a by-eye-like method should be built on.

Usage
-----
    python scripts/diag_spectral_separation.py            # compute + save plot
    python scripts/diag_spectral_separation.py --gui      # also open napari
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

DATA = Path(
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/pos00"
)
CONTOUR_PATH = DATA / "1_cellpose" / "nucleus_contours.tif"
FG_PATH = DATA / "1_cellpose" / "nucleus_foreground.tif"
GT_PATH = DATA / "2_nucleus" / "tracked_labels.tif"

FAINT_PCT = 15
PATCH = 32  # window for per-nucleus power spectrum


# --------------------------------------------------------------------------- #
# log-Gabor / monogenic signal (frequency domain, no external dependency)
# --------------------------------------------------------------------------- #
def _freq_grids(shape):
    v = np.fft.fftfreq(shape[0])
    u = np.fft.fftfreq(shape[1])
    U, V = np.meshgrid(u, v)
    radius = np.sqrt(U ** 2 + V ** 2)
    radius[0, 0] = 1.0
    return U, V, radius


def loggabor_radial(shape, wavelength, sigma_onf=0.55):
    _, _, radius = _freq_grids(shape)
    f0 = 1.0 / wavelength
    lg = np.exp(-(np.log(radius / f0) ** 2) / (2 * np.log(sigma_onf) ** 2))
    lg[0, 0] = 0.0
    return lg


def monogenic_features(img, wavelengths, sigma_onf=0.55, k_noise=2.0):
    """Return (band_energy, phase_symmetry) summed across scales.

    band_energy   = sum of log-Gabor local amplitudes (contrast-dependent).
    phase_symmetry= sum_s max(0, |even| - |odd| - T) / (sum_s amp + eps).
                    Large where the local signal is phase-symmetric (blob/ridge
                    centre). Amplitude normalisation makes it contrast-invariant;
                    the noise floor T suppresses flat background.
    """
    F = np.fft.fft2(img)
    U, V, radius = _freq_grids(img.shape)
    riesz_u = 1j * U / radius
    riesz_v = 1j * V / radius

    energy = np.zeros(img.shape, dtype=np.float64)
    sym_num = np.zeros(img.shape, dtype=np.float64)
    amp_sum = np.zeros(img.shape, dtype=np.float64)
    for wl in wavelengths:
        lg = loggabor_radial(img.shape, wl, sigma_onf)
        Flg = F * lg
        even = np.real(np.fft.ifft2(Flg))
        odd_u = np.real(np.fft.ifft2(Flg * riesz_u))
        odd_v = np.real(np.fft.ifft2(Flg * riesz_v))
        odd = np.sqrt(odd_u ** 2 + odd_v ** 2)
        amp = np.sqrt(even ** 2 + odd ** 2)
        # noise floor from the smallest-scale amplitude median (robust)
        T = k_noise * np.median(amp)
        energy += amp
        sym_num += np.maximum(0.0, np.abs(even) - np.abs(odd) - T)
        amp_sum += amp
    phase_symmetry = sym_num / (amp_sum + 1e-6)
    return energy, phase_symmetry


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
def radial_power(patch):
    p = patch - patch.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(p))) ** 2
    cy, cx = (np.array(P.shape) - 1) / 2.0
    y, x = np.indices(P.shape)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), P.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def collect_patches(img, gt, faint_flags, n_bg_per_frame=80, rng=None):
    rng = rng or np.random.default_rng(0)
    nuc, faint, bg = [], [], []
    half = PATCH // 2
    H, W = img.shape[1:]
    for t in range(img.shape[0]):
        lab = gt[t]
        # nucleus patches centred on centroids
        for rid in np.unique(lab):
            if rid == 0:
                continue
            cy, cx = ndi.center_of_mass(lab == rid)
            cy, cx = int(round(cy)), int(round(cx))
            if half <= cy < H - half and half <= cx < W - half:
                patch = img[t, cy - half:cy + half, cx - half:cx + half]
                nuc.append(patch)
                if faint_flags.get((t, int(rid)), False):
                    faint.append(patch)
        # background patches: random windows with no nucleus pixel
        any_nuc = lab > 0
        tries = 0
        got = 0
        while got < n_bg_per_frame and tries < n_bg_per_frame * 10:
            tries += 1
            cy = rng.integers(half, H - half)
            cx = rng.integers(half, W - half)
            win = any_nuc[cy - half:cy + half, cx - half:cx + half]
            if not win.any():
                bg.append(img[t, cy - half:cy + half, cx - half:cx + half])
                got += 1
    return nuc, faint, bg


def auc(pos, neg, max_n=200000):
    """Mann-Whitney AUC of pos vs neg (P(pos > neg))."""
    rng = np.random.default_rng(0)
    if len(pos) > max_n:
        pos = rng.choice(pos, max_n, replace=False)
    if len(neg) > max_n:
        neg = rng.choice(neg, max_n, replace=False)
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(np.float64) + 1
    rp = ranks[:len(pos)].sum()
    n1, n2 = len(pos), len(neg)
    u = rp - n1 * (n1 + 1) / 2.0
    return u / (n1 * n2)


def feature_separability(feat, gt, faint_flags, margin=3):
    """AUC separating inside-nucleus from background pixels for one feature map."""
    struct = ndi.generate_binary_structure(2, 2)
    pos, posf, neg = [], [], []
    for t in range(feat.shape[0]):
        lab = gt[t]
        any_nuc = lab > 0
        inside = ndi.binary_erosion(any_nuc, struct, iterations=1)
        far_bg = ~ndi.binary_dilation(any_nuc, struct, iterations=margin)
        pos.append(feat[t][inside])
        neg.append(feat[t][far_bg])
        # faint nuclei pixels
        faint_mask = np.zeros_like(any_nuc)
        for rid in np.unique(lab):
            if rid != 0 and faint_flags.get((t, int(rid)), False):
                faint_mask |= lab == rid
        posf.append(feat[t][faint_mask])
    pos = np.concatenate(pos)
    posf = np.concatenate(posf)
    neg = np.concatenate(neg)
    return auc(pos, neg), (auc(posf, neg) if len(posf) else float("nan"))


def build_faint_flags(gt, fg):
    means = []
    keys = []
    for t in range(gt.shape[0]):
        lab = gt[t]
        for rid in np.unique(lab):
            if rid == 0:
                continue
            means.append(float(fg[t][lab == rid].mean()))
            keys.append((t, int(rid)))
    cut = np.percentile(means, FAINT_PCT)
    return {k: (m <= cut) for k, m in zip(keys, means)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--wavelengths", type=float, nargs="+",
                    default=[8, 12, 18, 28],
                    help="log-Gabor centre wavelengths (px)")
    ap.add_argument("--out", default="scripts/diag_spectral_separation.png")
    args = ap.parse_args()

    contour = tifffile.imread(CONTOUR_PATH).astype(np.float32)[:args.frames]
    fg = tifffile.imread(FG_PATH).astype(np.float32)[:args.frames]
    gt = tifffile.imread(GT_PATH)[:args.frames]
    print(f"frames={contour.shape[0]}  wavelengths={args.wavelengths}")

    faint_flags = build_faint_flags(gt, fg)
    n_faint = sum(faint_flags.values())
    print(f"GT nuclei={len(faint_flags)}  faint={n_faint}")

    inputs = {"contour": contour, "fg": fg}

    # ---- feature maps -----------------------------------------------------
    feats = {}
    for name, img in inputs.items():
        energy = np.zeros_like(img)
        psym = np.zeros_like(img)
        for t in range(img.shape[0]):
            e, ps = monogenic_features(img[t], args.wavelengths)
            energy[t] = e
            psym[t] = ps
        feats[f"{name}_raw"] = img
        feats[f"{name}_loggabor_energy"] = energy
        feats[f"{name}_phase_symmetry"] = psym

    # ---- separability -----------------------------------------------------
    print("\nseparability (AUC: inside-nucleus vs background pixels)")
    print(f"{'feature':<28}{'all':>8}{'faint':>8}")
    seps = {}
    for name, feat in feats.items():
        a, af = feature_separability(feat, gt, faint_flags)
        seps[name] = (a, af)
        print(f"{name:<28}{a:>8.3f}{af:>8.3f}")

    # ---- radial power spectra --------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(inputs), figsize=(6 * len(inputs), 4.5))
    if len(inputs) == 1:
        axes = [axes]
    for ax, (name, img) in zip(axes, inputs.items()):
        nuc, faint, bg = collect_patches(img, gt, faint_flags)
        def avg_rp(patches):
            if not patches:
                return None
            return np.mean([radial_power(p) for p in patches], axis=0)
        rp_n, rp_f, rp_b = avg_rp(nuc), avg_rp(faint), avg_rp(bg)
        freq = np.arange(len(rp_n)) / PATCH  # cycles/px
        ax.loglog(freq[1:], rp_n[1:], label=f"nucleus (n={len(nuc)})", lw=2)
        if rp_f is not None:
            ax.loglog(freq[1:], rp_f[1:], label=f"faint (n={len(faint)})", lw=2)
        ax.loglog(freq[1:], rp_b[1:], label=f"background (n={len(bg)})", lw=2)
        for wl in args.wavelengths:
            ax.axvline(1.0 / wl, color="gray", ls=":", lw=0.8)
        ax.set_title(f"{name}: radial power spectrum")
        ax.set_xlabel("spatial frequency (cycles/px)")
        ax.set_ylabel("power")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"\nsaved power-spectrum figure -> {args.out}")

    if not args.gui:
        print("\n(pass --gui to inspect feature maps in napari)")
        return

    import napari
    v = napari.Viewer()
    v.add_image(fg, name="fg_raw", colormap="gray")
    v.add_image(contour, name="contour_raw", colormap="inferno",
                blending="additive", visible=False)
    for name, feat in feats.items():
        if name.endswith("_raw"):
            continue
        v.add_image(feat, name=name, colormap="magma", blending="additive",
                    visible=False)
    v.add_labels(gt, name="GT", opacity=0.3, visible=False)
    napari.run()


if __name__ == "__main__":
    main()
