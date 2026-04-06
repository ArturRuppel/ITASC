"""
Core pipeline functions operating on numpy arrays (no file I/O, no globals).

All processing parameters are passed explicitly so the same functions work
both from the CLI and from the napari widget.
"""

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from laptrack import LapTrack


# ── Image helpers ──────────────────────────────────────────────────────

def downscale(img, factor):
    """Block-average downsample by integer factor."""
    h, w = img.shape
    h2, w2 = h // factor, w // factor
    return img[:h2*factor, :w2*factor].reshape(h2, factor, w2, factor).mean(axis=(1, 3))


# ── Cellpose ───────────────────────────────────────────────────────────

def make_cp_model(model_type, custom_model_path=None, gpu=True):
    """Create and return a CellposeModel.

    Parameters
    ----------
    model_type        : "cpsam" | "custom"  (cellpose ≥ 4.0: all standard models use cpsam)
    custom_model_path : path to .pt file when model_type == "custom"
    gpu               : use GPU if available
    """
    import logging as _logging
    from cellpose.models import CellposeModel

    # Verify CUDA availability; warn and fall back instead of silently using CPU.
    if gpu:
        try:
            import torch
            if not torch.cuda.is_available():
                _logging.getLogger(__name__).warning(
                    "GPU requested but CUDA is not available; using CPU."
                )
                gpu = False
        except ImportError:
            _logging.getLogger(__name__).warning(
                "GPU requested but torch is not importable; using CPU."
            )
            gpu = False

    if model_type == "custom":
        if not custom_model_path:
            raise ValueError("custom_model_path is required when model_type='custom'")
        return CellposeModel(gpu=gpu, pretrained_model=custom_model_path)

    # In cellpose ≥ 4.0, "cpsam" is the only bundled model; it is auto-downloaded.
    _BUNDLED = {"cyto", "cyto2", "cyto3", "nuclei", "bact_omni", "cyto2_omni", "cpsam"}
    if model_type not in _BUNDLED:
        # Use cellpose's own model-directory lookup so we honour whatever path
        # cellpose itself would use, then verify the file actually exists.
        from pathlib import Path
        try:
            from cellpose import models as _cp_models
            _model_dir = Path(getattr(_cp_models, "model_dir",
                                      Path.home() / ".cellpose" / "models"))
        except Exception:
            _model_dir = Path.home() / ".cellpose" / "models"
        model_path = _model_dir / model_type
        if not model_path.exists():
            raise FileNotFoundError(
                f"Cellpose model '{model_type}' not found at {model_path}.\n"
                "Download it first or choose a different model (cyto3, nuclei, …)."
            )

    return CellposeModel(gpu=gpu, pretrained_model=model_type)


def run_cp(img, model, diameter, flow_threshold, cellprob_threshold, min_size,
           do_3D=False, stitch_threshold=None):
    """Run cellpose on a 2D grayscale image or a (Z, H, W) volume."""
    kwargs = dict(
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
    )
    if do_3D:
        kwargs["do_3D"] = True
        if img.ndim >= 3:
            kwargs["z_axis"] = 0
    elif stitch_threshold is not None:
        kwargs["stitch_threshold"] = stitch_threshold
        if img.ndim >= 3:
            kwargs["z_axis"] = 0
    masks, _, _ = model.eval(img, **kwargs)
    return masks


def run_cp_two_channel(img_primary, img_secondary, model, diameter,
                        flow_threshold, cellprob_threshold, min_size,
                        do_3D=False, stitch_threshold=None):
    """Run cellpose in two-channel mode (cell body + nucleus).

    img_primary   : (H, W) or (Z, H, W) array – cell / cytoplasm channel
    img_secondary : (H, W) or (Z, H, W) array – nuclear / helper channel

    Channels are stacked on the last axis → (H, W, 2) for 2D or (Z, H, W, 2)
    for 3D, which matches Cellpose v4+ convention.
    For 4D inputs Cellpose requires explicit z_axis and channel_axis.
    """
    img_2ch = np.stack([img_primary, img_secondary], axis=-1)
    kwargs = dict(
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
    )
    if do_3D:
        kwargs["do_3D"] = True
        if img_2ch.ndim == 4:
            kwargs["z_axis"] = 0
            kwargs["channel_axis"] = -1
    elif stitch_threshold is not None:
        kwargs["stitch_threshold"] = stitch_threshold
        if img_2ch.ndim == 4:
            kwargs["z_axis"] = 0
            kwargs["channel_axis"] = -1
    masks, _, _ = model.eval(img_2ch, **kwargs)
    return masks


# ── Tracking ───────────────────────────────────────────────────────────

def _iou_cost_matrix(f0, f1, centroids0, centroids1, max_link_dist,
                     iou_weight):
    """Compute the combined (centroid + IoU) cost matrix for one frame pair.

    Only computes mask IoU for pairs whose centroids are within max_link_dist,
    avoiding O(n²·H·W) work for distant cell pairs.

    Returns an (n0 × n1) float32 array in pixel units.
    """
    labels0 = np.array(sorted(set(np.unique(f0)) - {0}), dtype=np.int32)
    labels1 = np.array(sorted(set(np.unique(f1)) - {0}), dtype=np.int32)
    n0, n1  = len(labels0), len(labels1)
    if n0 == 0 or n1 == 0:
        return np.full((n0, n1), np.inf, dtype=np.float32)

    # Vectorised centroid distance matrix (n0 × n1)
    c0 = np.array([centroids0[l] for l in labels0])   # (n0, 2)
    c1 = np.array([centroids1[l] for l in labels1])   # (n1, 2)
    diff = c0[:, None, :] - c1[None, :, :]            # (n0, n1, 2)
    d    = np.sqrt((diff ** 2).sum(axis=2))            # (n0, n1)

    if iou_weight == 0.0:
        return d.astype(np.float32)

    # Only compute mask IoU for candidate pairs (centroid within max_link_dist)
    iou_mat  = np.zeros((n0, n1), dtype=np.float32)
    areas0   = {l: int((f0 == l).sum()) for l in labels0}
    masks1   = {l: (f1 == l)            for l in labels1}
    areas1   = {l: int(m.sum())         for l, m in masks1.items()}

    rows0, cols1 = np.where(d <= max_link_dist)
    for r, c in zip(rows0, cols1):
        la, lb   = int(labels0[r]), int(labels1[c])
        mask_a   = f0 == la
        inter    = int(np.sum(mask_a & masks1[lb]))
        if inter > 0:
            union = areas0[la] + areas1[lb] - inter
            iou_mat[r, c] = inter / union

    cost = ((1.0 - iou_weight) * d
            + iou_weight * max_link_dist * (1.0 - iou_mat))
    return cost.astype(np.float32)


def _make_matrix_metric(cost_mat, labels_row, labels_col):
    """Return a cdist callable that looks up a precomputed cost matrix.

    Coordinate vector: [y, x, label]
    Labels are used as keys into per-row/col index maps.
    """
    row_idx = {int(l): i for i, l in enumerate(labels_row)}
    col_idx = {int(l): i for i, l in enumerate(labels_col)}

    def _metric(u, v):
        r = row_idx.get(int(round(u[2])))
        c = col_idx.get(int(round(v[2])))
        if r is None or c is None:
            return np.inf
        return float(cost_mat[r, c])
    return _metric


def _gap_closing_metric(u, v):
    """Plain euclidean on the centroid coords; ignores the label tail."""
    return np.sqrt((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2)


def track_nuclei_laptrack(nuc_raw, max_link_dist, max_gap_dist,
                           gap_closing_max_frame_count,
                           metric="euclidean", gap_closing_metric="euclidean",
                           track_start_cost=None, track_end_cost=None,
                           alternative_cost_factor=1.05,
                           alternative_cost_percentile=90,
                           iou_weight=0.0,
                           progress_cb=None):
    """
    Track nuclei across frames using LapTrack (centroid-distance LAP).

    Parameters
    ----------
    iou_weight : float in [0, 1]
        Blend between pure centroid distance (0.0) and IoU-informed cost (1.0).
        Cost = (1-w)*d_euclidean + w*max_link_dist*(1-IoU).
        Gap closing always uses plain euclidean distance.
    progress_cb : callable(str) or None
        Called with progress messages during IoU precomputation so the caller
        can forward them to a log widget.

    Returns
    -------
    tracked_nuc : list of (H,W) uint16 arrays with consistent track IDs (1-based)
    track_df    : LapTrack output dataframe
    """
    def _log(msg):
        if progress_cb is not None:
            progress_cb(msg)

    records = []
    for t, nuc in enumerate(nuc_raw):
        if nuc.max() == 0:
            continue
        props = regionprops_table(nuc, properties=["label", "centroid"])
        df_t  = pd.DataFrame(props)
        df_t.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)
        df_t["frame"] = t
        records.append(df_t)

    if not records:
        return [np.zeros_like(n) for n in nuc_raw], pd.DataFrame()

    det_df = pd.concat(records, ignore_index=True)
    det_df["frame"] = det_df["frame"].astype(int)

    use_iou = iou_weight > 0.0
    if use_iou:
        # Build centroid lookup: frame -> {label: (y, x)}
        centroids = {}
        for t, nuc in enumerate(nuc_raw):
            if nuc.max() == 0:
                continue
            props = regionprops_table(nuc, properties=["label", "centroid"])
            centroids[t] = {
                int(l): (cy, cx)
                for l, cy, cx in zip(props["label"],
                                     props["centroid-0"],
                                     props["centroid-1"])
            }

        # Precompute one cost matrix per consecutive frame pair.
        # Each is an (n0 × n1) float32 array; labels are stored alongside for
        # index mapping.  Progress is reported so callers can log each frame.
        precomputed = {}   # t -> (cost_mat, {label: row}, sorted_col_labels, {label: col})
        n_pairs = len(nuc_raw) - 1
        for t in range(n_pairs):
            if nuc_raw[t].max() == 0 or nuc_raw[t + 1].max() == 0:
                _log(f"  IoU precompute frame {t + 1}/{n_pairs}")
                continue
            labels0 = np.array(sorted(set(np.unique(nuc_raw[t]))     - {0}), dtype=np.int32)
            labels1 = np.array(sorted(set(np.unique(nuc_raw[t + 1])) - {0}), dtype=np.int32)
            cost_mat = _iou_cost_matrix(
                nuc_raw[t], nuc_raw[t + 1],
                centroids[t], centroids[t + 1],
                float(max_link_dist), iou_weight,
            )
            row_idx = {int(l): i for i, l in enumerate(labels0)}
            col_idx = {int(l): i for i, l in enumerate(labels1)}
            precomputed[t] = (cost_mat, row_idx, col_idx)
            _log(f"  IoU precompute frame {t + 1}/{n_pairs}")

        def link_metric(u, v):
            # coord vector: [y, x, frame, label]
            t  = int(round(min(u[2], v[2])))
            la = int(round(u[3])) if u[2] <= v[2] else int(round(v[3]))
            lb = int(round(v[3])) if u[2] <= v[2] else int(round(u[3]))
            entry = precomputed.get(t)
            if entry is None:
                return float(max_link_dist) + 1.0
            r = entry[1].get(la)
            c = entry[2].get(lb)
            if r is None or c is None:
                return float(max_link_dist) + 1.0
            return float(entry[0][r, c])

        gap_metric = _gap_closing_metric
        coord_cols = ["y", "x", "frame", "label"]
    else:
        link_metric = metric
        gap_metric  = gap_closing_metric
        coord_cols  = ["y", "x"]

    tracker = LapTrack(
        metric=link_metric,
        cutoff=float(max_link_dist),
        gap_closing_metric=gap_metric,
        gap_closing_cutoff=float(max_gap_dist),
        gap_closing_max_frame_count=gap_closing_max_frame_count,
        splitting_cutoff=False,
        merging_cutoff=False,
        track_start_cost=track_start_cost,
        track_end_cost=track_end_cost,
        alternative_cost_factor=alternative_cost_factor,
        alternative_cost_percentile=alternative_cost_percentile,
    )
    track_df, _, _ = tracker.predict_dataframe(
        det_df, coordinate_cols=coord_cols, frame_col="frame"
    )
    track_df = track_df.copy()
    track_df["track_id"] = track_df["track_id"] + 1  # 1-based (0 = background)

    # Remap masks with consistent track IDs
    tracked_nuc = [np.zeros_like(nuc) for nuc in nuc_raw]
    for _, row in track_df.iterrows():
        t        = int(row["frame"])
        orig_lbl = int(row["label"])
        new_lbl  = int(row["track_id"])
        tracked_nuc[t][nuc_raw[t] == orig_lbl] = new_lbl

    # Fill gap frames: shift-copy nuclear shape to interpolated centroid
    for tid in sorted(track_df["track_id"].unique()):
        det_frames = sorted(track_df[track_df["track_id"] == tid]["frame"].tolist())
        for k in range(len(det_frames) - 1):
            t0, t1 = det_frames[k], det_frames[k + 1]
            if t1 - t0 <= 1:
                continue
            ys0, xs0 = np.where(tracked_nuc[t0] == tid)
            if len(ys0) == 0:
                continue
            c0      = np.array([ys0.mean(), xs0.mean()])
            c1_arr  = np.where(tracked_nuc[t1] == tid)
            if len(c1_arr[0]) == 0:
                continue
            c1 = np.array([c1_arr[0].mean(), c1_arr[1].mean()])
            H, W = tracked_nuc[t0].shape
            for gap_t in range(t0 + 1, t1):
                alpha  = (gap_t - t0) / (t1 - t0)
                c_pred = c0 * (1 - alpha) + c1 * alpha
                dy     = int(round(c_pred[0] - c0[0]))
                dx     = int(round(c_pred[1] - c0[1]))
                new_ys = np.clip(ys0 + dy, 0, H - 1)
                new_xs = np.clip(xs0 + dx, 0, W - 1)
                free   = tracked_nuc[gap_t][new_ys, new_xs] == 0
                tracked_nuc[gap_t][new_ys[free], new_xs[free]] = tid

    return tracked_nuc, track_df
