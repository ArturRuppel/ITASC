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


# ── Guided segmentation ────────────────────────────────────────────────

def run_cp_get_probability_map(img, model, diameter, flow_threshold,
                                cellprob_threshold, min_size):
    """Run cellpose on a 2D image and return the cell probability map.

    Unlike run_cp(), discards masks and returns flows[1] — the per-pixel
    cell probability (values roughly in [0, 1]) produced by the neural
    network before thresholding.  Used by guided segmentation: watershed
    uses this map as the elevation surface while nuclear track positions
    supply the seeds, so cell identity is fully determined by tracking
    and cellpose only contributes boundary information.

    Parameters
    ----------
    img : (H, W) array

    Returns
    -------
    prob_map : (H, W) float32 array
    """
    _, flows, _ = model.eval(
        img,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
    )
    # cellpose eval returns flows = [dx_to_circ(dP), dP, cellprob]:
    #   flows[0] = HSV visualization
    #   flows[1] = dP gradient maps  (2, H, W) — NOT the probability
    #   flows[2] = cellprob          (H, W)    — what we actually need
    return np.asarray(flows[2], dtype=np.float32)


def make_nuclear_seeds(track_df, frame, shape):
    """Build a (H, W) seed image from nuclear track positions for one frame.

    Parameters
    ----------
    track_df : DataFrame with columns (frame, track_id, y, x)
    frame    : int
    shape    : (H, W)

    Returns
    -------
    seeds : (H, W) int32; pixel value = track_id, 0 = background
    """
    seeds = np.zeros(shape, dtype=np.int32)
    H, W = shape
    for _, row in track_df[track_df["frame"] == frame].iterrows():
        yi = int(round(float(row["y"])))
        xi = int(round(float(row["x"])))
        if 0 <= yi < H and 0 <= xi < W:
            seeds[yi, xi] = int(row["track_id"])
    return seeds


def run_guided_segmentation(membrane_stack, track_df, model,
                             diameter, flow_threshold, cellprob_threshold,
                             min_size, progress_cb=None):
    """Segment cells using cellpose probability maps seeded by nuclear tracks.

    For each frame: run the cellpose network to get a probability map,
    plant one seed per tracked nucleus, then run watershed.  Output pixel
    values are track IDs — no post-hoc relabelling is needed.

    Parameters
    ----------
    membrane_stack : (T, H, W) or (T, Z, H, W) array
        For 4-D input the Z-axis is max-projected before running cellpose.
    track_df       : DataFrame with columns (frame, track_id, y, x)
                     as returned by track_nuclei_laptrack.
    model          : CellposeModel
    progress_cb    : callable(str) or None

    Returns
    -------
    label_stack : uint16 array with the same shape as membrane_stack.
        For (T, H, W) input: (T, H, W).
        For (T, Z, H, W) input: (T, Z, H, W) — each Z-plane carries the
        same 2-D segmentation (the per-frame max-projection result).
    """
    from skimage.segmentation import watershed

    membrane_stack = np.asarray(membrane_stack)
    input_shape = membrane_stack.shape

    if membrane_stack.ndim == 3:
        T, H, W = input_shape
        proj_stack = membrane_stack          # already (T, H, W)
    elif membrane_stack.ndim == 4:
        T, Z, H, W = input_shape
        proj_stack = membrane_stack.max(axis=1)   # max-project Z → (T, H, W)
    else:
        raise ValueError(
            f"membrane_stack must be (T, H, W) or (T, Z, H, W), "
            f"got shape {input_shape}"
        )

    label_stack = np.zeros(input_shape, dtype=np.uint16)

    for t in range(T):
        if progress_cb:
            progress_cb(f"Frame {t + 1}/{T}")

        prob_map = run_cp_get_probability_map(
            proj_stack[t], model,
            diameter, flow_threshold, cellprob_threshold, min_size,
        )
        seeds = make_nuclear_seeds(track_df, t, (H, W))

        if seeds.max() == 0:
            continue

        cell_mask = prob_map > cellprob_threshold
        seg2d = watershed(
            -prob_map, markers=seeds, mask=cell_mask,
        ).astype(np.uint16)

        if membrane_stack.ndim == 4:
            label_stack[t] = seg2d[np.newaxis]   # broadcast 2-D result to all Z-planes
        else:
            label_stack[t] = seg2d

    return label_stack


def flatten_nuclear_labels(labels, method="max", hole_fill_radius=0,
                           min_size=0, split_touching=False):
    """Flatten a (T, Z, H, W) or (T, H, W) nuclear label stack to (T, H, W).

    Parameters
    ----------
    labels           : (T, Z, H, W) or (T, H, W) integer array
    method           : "max" | "mean" | "sum" — Z-projection method for 4-D input
    hole_fill_radius : int; fills holes up to this radius inside each label
    min_size         : int; remove labeled regions smaller than this (px²)
    split_touching   : bool; watershed on distance map to separate merged nuclei

    Returns
    -------
    out : (T, H, W) uint16 array
    """
    from skimage.morphology import remove_small_objects, disk, binary_closing
    from skimage.measure import label as sk_label, regionprops

    labels = np.asarray(labels)
    if labels.ndim == 4:
        if method == "max":
            flat = labels.max(axis=1)
        elif method == "mean":
            flat = labels.mean(axis=1).astype(labels.dtype)
        else:
            flat = labels.sum(axis=1).astype(labels.dtype)
    elif labels.ndim == 3:
        flat = labels.copy()
    else:
        raise ValueError(
            f"Expected (T, H, W) or (T, Z, H, W), got shape {labels.shape}"
        )

    T, H, W = flat.shape
    out = np.zeros((T, H, W), dtype=np.uint16)

    for t in range(T):
        frame = flat[t].astype(np.int32)
        if frame.max() == 0:
            continue

        mask = frame > 0

        # Hole filling: close holes within labeled regions
        if hole_fill_radius > 0:
            from scipy.ndimage import binary_fill_holes
            filled_mask = binary_fill_holes(mask)
            # only fill in pixels that were holes (background surrounded by labels)
            newly_filled = filled_mask & ~mask
            if newly_filled.any():
                from scipy.ndimage import label as ndi_label
                filled_struct, _ = ndi_label(newly_filled)
                for hole_lbl in np.unique(filled_struct[filled_struct > 0]):
                    hole_pix = filled_struct == hole_lbl
                    # find which nucleus label surrounds this hole
                    border = np.zeros_like(mask, dtype=bool)
                    from scipy.ndimage import binary_dilation
                    dilated = binary_dilation(hole_pix, iterations=1)
                    border = dilated & mask
                    neighbors = frame[border]
                    if neighbors.size > 0:
                        fill_val = int(np.bincount(neighbors[neighbors > 0]).argmax())
                        frame[hole_pix] = fill_val

        # Min size filtering
        if min_size > 0:
            for region_lbl in np.unique(frame[frame > 0]):
                if int((frame == region_lbl).sum()) < min_size:
                    frame[frame == region_lbl] = 0

        # Split touching nuclei via watershed on distance map
        if split_touching:
            from scipy.ndimage import distance_transform_edt, binary_dilation
            from skimage.segmentation import watershed
            from skimage.feature import peak_local_max
            current_mask = frame > 0
            dist = distance_transform_edt(current_mask)
            coords = peak_local_max(dist, footprint=np.ones((3, 3)),
                                    labels=current_mask)
            seeds = np.zeros_like(current_mask, dtype=np.int32)
            seeds[tuple(coords.T)] = 1
            from scipy.ndimage import label as ndi_label
            seeds, _ = ndi_label(seeds)
            frame = watershed(-dist, seeds, mask=current_mask).astype(np.int32)

        out[t] = np.clip(frame, 0, 65535).astype(np.uint16)

    return out


def run_guided_segmentation_from_labels(membrane_stack, nuclear_labels_stack,
                                         model, diameter, flow_threshold,
                                         cellprob_threshold, min_size,
                                         progress_cb=None):
    """Segment cells using cellpose probability maps seeded by nuclear labels.

    Uses nuclear label regions directly as watershed seeds (rather than point
    centroids), which produces more stable boundaries and does not require a
    prior tracking step.  Cell IDs in the output match nuclear label IDs —
    if the nuclear labels have been tracked, the cell segmentation is
    automatically tracked too.

    Parameters
    ----------
    membrane_stack      : (T, H, W) or (T, Z, H, W) array
    nuclear_labels_stack: (T, H, W) uint integer array — 2D nuclear labels,
                          one per time frame. Each unique nonzero value is one seed.
    model               : CellposeModel
    progress_cb         : callable(str) or None

    Returns
    -------
    label_stack : uint16 (T, H, W) array; pixel values = nuclear label IDs.
    """
    from skimage.segmentation import watershed

    membrane_stack       = np.asarray(membrane_stack)
    nuclear_labels_stack = np.asarray(nuclear_labels_stack)

    if membrane_stack.ndim == 4:
        proj_stack = membrane_stack.max(axis=1)   # (T, H, W)
    elif membrane_stack.ndim == 3:
        proj_stack = membrane_stack
    else:
        raise ValueError(
            f"membrane_stack must be (T, H, W) or (T, Z, H, W), "
            f"got shape {membrane_stack.shape}"
        )

    T, H, W = proj_stack.shape
    label_stack = np.zeros((T, H, W), dtype=np.uint16)

    if nuclear_labels_stack.ndim != 3 or nuclear_labels_stack.shape[0] != T:
        raise ValueError(
            f"nuclear_labels_stack must be (T, H, W) with T={T}, "
            f"got shape {nuclear_labels_stack.shape}"
        )

    for t in range(T):
        if progress_cb:
            progress_cb(f"Frame {t + 1}/{T}")

        prob_map = run_cp_get_probability_map(
            proj_stack[t], model,
            diameter, flow_threshold, cellprob_threshold, min_size,
        )

        seeds = nuclear_labels_stack[t].astype(np.int32)
        if seeds.max() == 0:
            continue

        cell_mask = prob_map > cellprob_threshold
        seg2d = watershed(-prob_map, markers=seeds, mask=cell_mask).astype(np.uint16)
        label_stack[t] = seg2d

    return label_stack


def track_nuclei_3d_laptrack(nuclear_volumes, z_scale,
                              max_link_dist, max_gap_dist,
                              gap_closing_max_frame_count,
                              metric="euclidean",
                              gap_closing_metric="euclidean",
                              track_start_cost=None,
                              track_end_cost=None,
                              alternative_cost_factor=1.05,
                              alternative_cost_percentile=90,
                              progress_cb=None):
    """Track nuclei across time points using 3D centroids.

    Parameters
    ----------
    nuclear_volumes : list of (Z, H, W) integer arrays, one per time point.
                      Each array is one time point's nuclear label volume.
    z_scale         : float
                      z-spacing in xy-pixel units (z_spacing_um / xy_pixel_um).
                      z centroids are multiplied by this so that max_link_dist
                      can be specified in xy-pixel units throughout.
    max_link_dist   : float, max linking distance in xy-pixel units
    progress_cb     : callable(str) or None

    Returns
    -------
    track_df : DataFrame with columns (frame, track_id, label, z, y, x).
               (y, x) are the 2D centroid — ready for use as watershed seeds.
    """
    def _log(msg):
        if progress_cb is not None:
            progress_cb(msg)

    records = []
    for t, vol in enumerate(nuclear_volumes):
        if vol.ndim == 2:
            # Treat a 2D (H,W) frame as a single-slice volume
            vol = vol[np.newaxis]
        if vol.max() == 0:
            _log(f"  Frame {t}: no nuclei")
            continue
        props = regionprops_table(vol, properties=["label", "centroid"])
        df_t = pd.DataFrame(props)
        df_t.rename(columns={
            "centroid-0": "z",
            "centroid-1": "y",
            "centroid-2": "x",
        }, inplace=True)
        df_t["frame"] = t
        # Scale z so distances are comparable to xy-pixel distances
        df_t["z_s"] = df_t["z"] * z_scale
        records.append(df_t)
        _log(f"  Frame {t}: {len(df_t)} nuclei detected")

    if not records:
        return pd.DataFrame(columns=["frame", "track_id", "label", "z", "y", "x"])

    det_df = pd.concat(records, ignore_index=True)
    det_df["frame"] = det_df["frame"].astype(int)

    tracker = LapTrack(
        metric=metric,
        cutoff=float(max_link_dist),
        gap_closing_metric=gap_closing_metric,
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
        det_df, coordinate_cols=["z_s", "y", "x"], frame_col="frame",
    )
    track_df = track_df.reset_index(drop=True).copy()
    track_df["track_id"] = track_df["track_id"] + 1  # 1-based

    return track_df[["frame", "track_id", "label", "z", "y", "x"]].copy()


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
