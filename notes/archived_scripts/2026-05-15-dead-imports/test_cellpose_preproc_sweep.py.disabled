"""
Cellpose preprocessing sweep — feasibility test.

Bypasses the contour-averaging step entirely. For each combination of
(gamma, blur_sigma, cellprob_threshold), preprocesses the raw prob/dp maps
and runs Cellpose compute_masks directly. Each z-slice at each combo is
stored as an independent hypothesis in the existing hypotheses.h5 database.

Run on two time frames with large steps to get a broad picture quickly.
After writing, opens a napari viewer showing a sampled grid of results.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

from cellflow.database.hypotheses import HypothesisRecord, write_hypothesis_sweep_h5
from cellflow.segmentation import apply_gamma

# ── Configuration ─────────────────────────────────────────────────────────────

POS_DIR = Path(
    "/home/aruppel/Data"
    "/2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/v2/pos00"
)
FRAMES = [0, 1]

GAMMA_VALS         = [0.1, 0.5, 1.0, 1.5, 2.0]
BLUR_SIGMA_VALS    = [0.0, 2.5, 5.0, 7.5, 10.0]
CELLPROB_THRESH_VALS = [-5.0, -2.5, 0.0, 2.5, 5.0]

# ── Params dataclass ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CellposePreProcSweepParams:
    """Parameters for one preprocessing-sweep hypothesis."""

    gamma: float = 1.0
    blur_sigma: float = 0.0
    cellprob_threshold: float = 0.0
    z_slice: int = 0

    def to_dict(self) -> dict:
        return {"method": "cellpose_preproc_sweep", **asdict(self)}


# ── Core sweep logic ──────────────────────────────────────────────────────────


def sweep_frame(
    prob_3d: np.ndarray,  # (Z, Y, X) float32 logits
    dp_3d: np.ndarray,    # (Z, 2, Y, X) float32 flows
    t: int,
    *,
    gamma_vals: list[float],
    blur_sigma_vals: list[float],
    cellprob_thresh_vals: list[float],
) -> list[HypothesisRecord]:
    try:
        import torch
        from cellpose.dynamics import compute_masks
    except ImportError as exc:
        raise ImportError("cellpose and torch must be installed") from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_z = prob_3d.shape[0]
    combos = list(itertools.product(gamma_vals, blur_sigma_vals, cellprob_thresh_vals))
    n_hyp = len(combos) * n_z
    print(f"  t={t}: {len(combos)} combos × {n_z} z-slices = {n_hyp} hypotheses")

    records: list[HypothesisRecord] = []
    t0 = time.time()

    for i, (gamma, sigma, thresh) in enumerate(combos):
        prob_g = apply_gamma(prob_3d, gamma)

        if sigma > 0:
            prob_pre = np.stack([gaussian_filter(prob_g[z], sigma=sigma) for z in range(n_z)])
            dp_pre = np.stack([
                np.stack([gaussian_filter(dp_3d[z, c], sigma=sigma) for c in range(2)])
                for z in range(n_z)
            ])
        else:
            prob_pre = prob_g
            dp_pre = dp_3d

        for z in range(n_z):
            try:
                result = compute_masks(
                    dp_pre[z],
                    prob_pre[z],
                    cellprob_threshold=float(thresh),
                    flow_threshold=0.0,
                    niter=200,
                    do_3D=False,
                    device=device,
                )
                masks = result[0] if isinstance(result, tuple) else result
            except Exception as e:
                print(f"    WARN combo {i+1} z={z}: {e} — storing empty mask")
                masks = np.zeros(prob_pre.shape[1:], dtype=np.uint32)

            labels_3d = np.asarray(masks, dtype=np.uint32)[np.newaxis]  # (1, Y, X)
            params = CellposePreProcSweepParams(
                gamma=gamma,
                blur_sigma=sigma,
                cellprob_threshold=thresh,
                z_slice=z,
            )
            records.append(HypothesisRecord(t=t, p=0, labels=labels_3d, params=params))

        if (i + 1) % 25 == 0 or (i + 1) == len(combos):
            elapsed = time.time() - t0
            per_combo = elapsed / (i + 1)
            remaining = per_combo * (len(combos) - i - 1)
            print(
                f"    {i+1}/{len(combos)} combos  "
                f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)  "
                f"γ={gamma} σ={sigma} thr={thresh}"
            )

    return records


# ── Napari viewer ─────────────────────────────────────────────────────────────


def view_sample(
    prob_3d: np.ndarray,
    records: list[HypothesisRecord],
    t: int,
    n_sample: int = 16,
) -> None:
    import napari

    t_records = [r for r in records if r.t == t]
    if not t_records:
        print("No records to visualise.")
        return

    step = max(1, len(t_records) // n_sample)
    sample = t_records[::step][:n_sample]

    viewer = napari.Viewer(title=f"Cellpose preproc sweep — t={t} ({len(sample)} samples)")
    viewer.add_image(prob_3d.mean(axis=0), name="prob z-avg", colormap="inferno")

    for rec in sample:
        p = rec.params
        assert isinstance(p, CellposePreProcSweepParams)
        name = f"γ={p.gamma} σ={p.blur_sigma} thr={p.cellprob_threshold} z={p.z_slice}"
        viewer.add_labels(rec.labels[0], name=name, visible=False)

    # Make the first sample visible so napari opens with something to see.
    if viewer.layers:
        viewer.layers[-len(sample)].visible = True

    print(f"\nNapari viewer open — {len(sample)} hypothesis layers for t={t}.")
    print("Toggle layers in the layer list to compare hypotheses.")
    napari.run()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    prob_path = POS_DIR / "1_cellpose" / "nucleus_prob_3dt.tif"
    dp_path   = POS_DIR / "1_cellpose" / "nucleus_dp_3dt.tif"
    out_path  = POS_DIR / "2_nucleus"  / "hypotheses.h5"

    print("Loading prob/dp stacks…")
    prob_full = np.asarray(tifffile.imread(str(prob_path)), dtype=np.float32)
    dp_full   = np.asarray(tifffile.imread(str(dp_path)),   dtype=np.float32)

    if prob_full.ndim == 3:
        prob_full = prob_full[np.newaxis]
    if dp_full.ndim == 4:
        dp_full = dp_full[np.newaxis]

    print(f"prob {prob_full.shape}  range=[{prob_full.min():.2f}, {prob_full.max():.2f}]")
    print(f"dp   {dp_full.shape}")
    print(
        f"\nSweep: {len(GAMMA_VALS)} gamma × {len(BLUR_SIGMA_VALS)} sigma "
        f"× {len(CELLPROB_THRESH_VALS)} threshold = "
        f"{len(GAMMA_VALS)*len(BLUR_SIGMA_VALS)*len(CELLPROB_THRESH_VALS)} combos"
    )

    all_records: list[HypothesisRecord] = []
    for t in FRAMES:
        if t >= prob_full.shape[0]:
            print(f"Skipping t={t}: only {prob_full.shape[0]} frames available.")
            continue
        print(f"\nProcessing t={t}…")
        records = sweep_frame(
            prob_full[t], dp_full[t], t,
            gamma_vals=GAMMA_VALS,
            blur_sigma_vals=BLUR_SIGMA_VALS,
            cellprob_thresh_vals=CELLPROB_THRESH_VALS,
        )
        all_records.extend(records)
        print(f"  t={t}: {len(records)} records")

    print(f"\nWriting {len(all_records)} records → {out_path} (append, no overwrite)…")
    t_write = time.time()
    write_hypothesis_sweep_h5(out_path, iter(all_records), overwrite=False)
    print(f"Done in {time.time() - t_write:.1f}s.")

    view_sample(prob_full[FRAMES[0]], all_records, t=FRAMES[0])


if __name__ == "__main__":
    main()
