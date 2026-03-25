"""
Core pipeline functions operating on numpy arrays (no file I/O, no globals).

All processing parameters are passed explicitly so the same functions work
both from the CLI and from the napari widget.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.spatial import Voronoi, cKDTree
from skimage.filters import gaussian
from skimage.measure import regionprops, regionprops_table
from skimage.morphology import disk, binary_closing
from skimage.segmentation import expand_labels
from laptrack import LapTrack


# ── Image helpers ──────────────────────────────────────────────────────

def downscale(img, factor):
    """Block-average downsample by integer factor."""
    h, w = img.shape
    h2, w2 = h // factor, w // factor
    return img[:h2*factor, :w2*factor].reshape(h2, factor, w2, factor).mean(axis=(1, 3))


def smooth_labels(masks, sigma, thresh):
    """Gaussian-smooth each label contour; resolve overlaps by winner-takes-all."""
    labels   = np.unique(masks[masks > 0])
    best_val = np.zeros(masks.shape, dtype=np.float32)
    result   = np.zeros(masks.shape, dtype=masks.dtype)
    for lbl in labels:
        blob    = binary_closing(masks == lbl, disk(2)).astype(np.float32)
        blurred = gaussian(blob, sigma=sigma)
        win     = blurred > best_val
        result[win]   = lbl
        best_val[win] = blurred[win]
    result[best_val < thresh] = 0
    return result


def expand_voronoi(nuclei, max_expand):
    """Expand nuclear seeds via Voronoi tessellation with distance limit.

    Each background pixel is assigned to the nearest nucleus label.
    Growth is capped at max_expand pixels from the nearest nucleus boundary,
    preventing edge cells from growing without bound.
    """
    return expand_labels(nuclei.astype(np.int32), distance=max_expand).astype(np.int32)


def _polygon_centroid(vertices):
    """Centroid of a polygon using the shoelace formula. Returns (y, x) or None."""
    n = len(vertices)
    if n < 3:
        return None
    y = vertices[:, 0]
    x = vertices[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    signed_area = np.sum(cross)
    if abs(signed_area) < 1e-12:
        return np.mean(vertices, axis=0)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (3 * signed_area)
    cx = np.sum((x + np.roll(x, -1)) * cross) / (3 * signed_area)
    return np.array([cy, cx])


def _lloyd_relaxation(positions, image_shape, n_iterations=10, tol=0.1):
    """Lloyd's algorithm: iteratively move seeds to Voronoi polygon centroids.

    Uses mirror points at image boundaries for a bounded tessellation.
    Stops early when max centroid displacement < tol.
    """
    H, W = image_shape
    pts = positions.copy()

    for _ in range(n_iterations):
        all_pts = np.vstack([
            pts,
            np.column_stack((-pts[:, 0],          pts[:, 1])),        # top mirror
            np.column_stack((2 * H - pts[:, 0],   pts[:, 1])),        # bottom mirror
            np.column_stack((pts[:, 0],           -pts[:, 1])),        # left mirror
            np.column_stack((pts[:, 0],  2 * W - pts[:, 1])),         # right mirror
        ])
        vor = Voronoi(all_pts)
        n_real = len(pts)
        new_pts = pts.copy()

        for i in range(n_real):
            region = vor.regions[vor.point_region[i]]
            if -1 in region or len(region) < 3:
                continue
            verts = np.clip(vor.vertices[region], [0, 0], [H, W])
            c = _polygon_centroid(verts)
            if c is not None:
                new_pts[i] = c

        max_disp = np.max(np.linalg.norm(new_pts - pts, axis=1))
        pts = new_pts
        if max_disp < tol:
            break

    return pts


def expand_voronoi_lloyd(nuclei, max_expand, lloyd_iterations=10, lloyd_tol=0.1):
    """Expand nuclear masks via Lloyd's relaxation (centroidal Voronoi tessellation).

    Extracts nuclear centroids, runs Lloyd's algorithm to produce more
    regular (hexagonal-like) cell shapes, then rasterizes via nearest-neighbour
    and applies the max_expand distance constraint.
    """
    H, W = nuclei.shape
    props = regionprops(nuclei.astype(np.int32))
    if not props:
        return np.zeros_like(nuclei, dtype=np.int32)

    labels    = np.array([p.label    for p in props])
    positions = np.array([p.centroid for p in props])   # (N, 2) as (y, x)

    relaxed = _lloyd_relaxation(positions, (H, W), lloyd_iterations, lloyd_tol)

    # Rasterize: assign each pixel to its nearest relaxed centroid
    tree = cKDTree(relaxed)
    yy, xx = np.mgrid[0:H, 0:W]
    _, nearest_idx = tree.query(np.column_stack([yy.ravel(), xx.ravel()]))
    result = labels[nearest_idx].reshape(H, W).astype(np.int32)

    # Apply max_expand distance constraint (mirrors expand_voronoi behaviour)
    dist_from_nuc = distance_transform_edt(nuclei == 0)
    result[dist_from_nuc > max_expand] = 0

    return result


# ── Cellpose ───────────────────────────────────────────────────────────

def make_cp_model(model_type, custom_model_path=None, gpu=True):
    """Create and return a CellposeModel.

    Parameters
    ----------
    model_type        : "cyto3" | "nuclei" | "cpsam" | "custom"
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

    # Models bundled with / auto-downloaded by cellpose itself.
    _BUNDLED = {"cyto", "cyto2", "cyto3", "nuclei", "bact_omni", "cyto2_omni"}
    if model_type not in _BUNDLED:
        from pathlib import Path
        model_path = Path.home() / ".cellpose" / "models" / model_type
        if not model_path.exists():
            raise FileNotFoundError(
                f"Cellpose model '{model_type}' not found at {model_path}.\n"
                "Download it first or choose a different model (cyto3, nuclei, …)."
            )

    return CellposeModel(gpu=gpu, pretrained_model=model_type)


def run_cp(img, model, diameter, flow_threshold, cellprob_threshold, min_size):
    """Run cellpose on a single 2D grayscale image."""
    masks, _, _ = model.eval(
        img,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
    )
    return masks


def run_cp_two_channel(img_primary, img_secondary, model, diameter,
                        flow_threshold, cellprob_threshold, min_size):
    """Run cellpose in two-channel mode (cell body + nucleus).

    img_primary   : (H, W) array – cell / cytoplasm channel
    img_secondary : (H, W) array – nuclear / helper channel

    Cellpose v4+: channels are inferred from array shape. Stack as (H, W, 2)
    with cell channel first, nuclear channel second (same convention as the
    old channels=[1, 2] but without the deprecated kwarg).
    """
    img_2ch = np.stack([img_primary, img_secondary], axis=-1)  # (H, W, 2)
    masks, _, _ = model.eval(
        img_2ch,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
    )
    return masks


# ── Tracking ───────────────────────────────────────────────────────────

def track_nuclei_laptrack(nuc_raw, max_link_dist, max_gap_dist,
                           gap_closing_max_frame_count):
    """
    Track nuclei across frames using LapTrack (centroid-distance LAP).

    Returns
    -------
    tracked_nuc : list of (H,W) uint16 arrays with consistent track IDs (1-based)
    track_df    : LapTrack output dataframe
    """
    records = []
    for t, nuc in enumerate(nuc_raw):
        if nuc.max() == 0:
            continue
        props = regionprops_table(nuc, properties=["label", "centroid", "area"])
        df_t  = pd.DataFrame(props)
        df_t.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)
        df_t["frame"] = t
        records.append(df_t)

    if not records:
        return [np.zeros_like(n) for n in nuc_raw], pd.DataFrame()

    det_df = pd.concat(records, ignore_index=True)
    det_df["frame"] = det_df["frame"].astype(int)

    tracker = LapTrack(
        metric="euclidean",
        cutoff=float(max_link_dist),
        gap_closing_metric="euclidean",
        gap_closing_cutoff=float(max_gap_dist),
        gap_closing_max_frame_count=gap_closing_max_frame_count,
        splitting_cutoff=False,
        merging_cutoff=False,
    )
    track_df, _, _ = tracker.predict_dataframe(
        det_df, coordinate_cols=["y", "x"], frame_col="frame"
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


# ── Cell bodies ────────────────────────────────────────────────────────

def _centroid(frame, lbl):
    ys, xs = np.where(frame == lbl)
    return np.array([ys.mean(), xs.mean()]) if len(ys) > 0 else None


def cells_from_nuclei(tracked_nuc, max_expand, smooth_sigma, smooth_thresh):
    """Derive smoothed cell body masks from tracked nuclei via Voronoi expansion."""
    return [
        smooth_labels(
            expand_voronoi(nuc, max_expand),
            smooth_sigma, smooth_thresh
        )
        for nuc in tracked_nuc
    ]


# ── Temporal correction ────────────────────────────────────────────────

def correct_false_splits(tracked_nuc, dapi_imgs, model, params, log=None):
    """
    Detect false nuclear splits and correct them in-place.

    Returns corrected nuclear mask list (copy of input).
    """
    max_split_radius = params["max_expand"] * params["max_split_radius_factor"]
    division_confirm = params["division_confirm_frames"]
    division_sep_min = params["division_sep_min_px"]

    N        = len(tracked_nuc)
    corr_nuc = [f.copy() for f in tracked_nuc]
    checked  = set()
    candidates = []

    for t in range(1, N):
        ids_prev = set(np.unique(corr_nuc[t-1])) - {0}
        ids_curr = set(np.unique(corr_nuc[t]))   - {0}
        new_ids  = ids_curr - ids_prev

        for new_id in new_ids:
            c_new = _centroid(corr_nuc[t], new_id)
            if c_new is None:
                continue

            partner, partner_dist = None, float("inf")
            for eid in (ids_curr - new_ids):
                c = _centroid(corr_nuc[t], eid)
                if c is None:
                    continue
                d = np.linalg.norm(c_new - c)
                if d < partner_dist:
                    partner_dist, partner = d, eid

            if partner is None or partner_dist > max_split_radius:
                continue

            pair = (min(new_id, partner), max(new_id, partner))
            if pair in checked:
                continue
            checked.add(pair)

            area_prev  = int((corr_nuc[t-1] == partner).sum())
            area_now   = int((corr_nuc[t]   == partner).sum())
            area_new   = int((corr_nuc[t]   == new_id).sum())
            area_ratio = (area_now + area_new) / (area_prev + 1e-3)
            if not (0.65 < area_ratio < 1.5):
                continue

            dists = []
            for obs_t in range(t, min(t + division_confirm, N)):
                ca = _centroid(corr_nuc[obs_t], partner)
                cb = _centroid(corr_nuc[obs_t], new_id)
                if ca is None or cb is None:
                    break
                dists.append(np.linalg.norm(ca - cb))

            if len(dists) < 2 or (dists[-1] - dists[0]) < division_sep_min:
                candidates.append((t, new_id, partner, c_new))

    if log:
        log(f"  {len(candidates)} false-split candidate(s) found")

    n_fixed = 0
    for idx, (t, new_id, partner, c_new) in enumerate(candidates):
        if log:
            log(f"  [{idx+1}/{len(candidates)}] t={t}: nucleus {new_id} → merge into {partner}")

        merged = False
        for cp_strictness in [1.0, 2.0]:
            new_nuc = run_cp(
                dapi_imgs[t], model,
                diameter=params["diameter"],
                min_size=params["min_size"],
                cellprob_threshold=params["cellprob_threshold"] + cp_strictness,
                flow_threshold=max(0.2, params["flow_threshold"] - 0.1),
            )
            H, W   = new_nuc.shape
            c_part = _centroid(corr_nuc[t], partner)
            if c_part is None:
                break
            cy_p, cx_p = int(round(c_part[0])), int(round(c_part[1]))
            cy_n, cx_n = int(round(c_new[0])),  int(round(c_new[1]))
            lbl_p = new_nuc[cy_p, cx_p] if (0 <= cy_p < H and 0 <= cx_p < W) else 0
            lbl_n = new_nuc[cy_n, cx_n] if (0 <= cy_n < H and 0 <= cx_n < W) else 0

            if lbl_p != 0 and lbl_p == lbl_n:
                corr_nuc[t][corr_nuc[t] == partner] = 0
                corr_nuc[t][corr_nuc[t] == new_id]  = 0
                corr_nuc[t][new_nuc == lbl_p]        = partner
                merged = True
                if log:
                    log(f"    confirmed by re-segmentation")
                break

        if not merged:
            corr_nuc[t][corr_nuc[t] == new_id] = partner
            if log:
                log(f"    merged by mask union")

        n_fixed += 1

    if log:
        log(f"  Fixed {n_fixed} false split(s)")

    return corr_nuc


# ── Full pipeline ──────────────────────────────────────────────────────

def run_pipeline(dapi_imgs, params, log=None, skip_temporal=False):
    """
    Run the full segmentation + tracking pipeline on arrays.

    Parameters
    ----------
    dapi_imgs    : list of (H,W) 2D arrays (nuclear channel)
    params       : dict of pipeline parameters (see config.yaml for keys)
    log          : optional callable(str) for progress messages
    skip_temporal: if True, skip false-split correction step

    Returns
    -------
    dict with keys:
        'tracked_nuc'  : list of (H,W) uint16 label arrays
        'tracked_cell' : list of (H,W) uint16 label arrays
        'corr_nuc'     : (only if not skip_temporal) list of (H,W) uint16 arrays
        'corr_cell'    : (only if not skip_temporal) list of (H,W) uint16 arrays
    """
    def _log(msg):
        if log:
            log(msg)

    _log("Loading Cellpose model...")
    model = make_cp_model(params["model"], gpu=params.get("gpu", True))

    _log("Running Cellpose on DAPI frames...")
    nuc_raw = []
    for i, dapi in enumerate(dapi_imgs):
        nuc = run_cp(
            dapi, model,
            diameter=params["diameter"],
            flow_threshold=params["flow_threshold"],
            cellprob_threshold=params["cellprob_threshold"],
            min_size=params["min_size"],
        )
        nuc_raw.append(nuc)
        n = len(np.unique(nuc[nuc > 0]))
        _log(f"  Frame {i}: {n} nucleus/nuclei detected")

    _log("Tracking nuclei with LapTrack...")
    tracked_nuc, track_df = track_nuclei_laptrack(
        nuc_raw,
        max_link_dist=params["max_link_dist"],
        max_gap_dist=params["max_gap_dist"],
        gap_closing_max_frame_count=params["gap_closing_max_frame_count"],
    )
    n_tracks = track_df["track_id"].nunique() if len(track_df) > 0 else 0
    _log(f"  {n_tracks} track(s) found")

    _log("Expanding nuclei to cell bodies (Voronoi + smoothing)...")
    tracked_cell = cells_from_nuclei(
        tracked_nuc,
        max_expand=params["max_expand"],
        smooth_sigma=params["smooth_sigma"],
        smooth_thresh=params["smooth_thresh"],
    )

    result = {
        "tracked_nuc":  tracked_nuc,
        "tracked_cell": tracked_cell,
    }

    if not skip_temporal:
        _log("Detecting and correcting false splits...")
        corr_nuc = correct_false_splits(tracked_nuc, dapi_imgs, model, params, log=_log)

        _log("Deriving corrected cell bodies...")
        corr_cell = cells_from_nuclei(
            corr_nuc,
            max_expand=params["max_expand"],
            smooth_sigma=params["smooth_sigma"],
            smooth_thresh=params["smooth_thresh"],
        )
        result["corr_nuc"]  = corr_nuc
        result["corr_cell"] = corr_cell

    _log("Pipeline complete!")
    return result


# ── File-based auto-run (mirrors run_pipeline.py) ──────────────────────

def load_from_files(raw_dir, position, downscale_factor):
    """
    Discover and load raw TIF files matching the original naming convention.

        raw_dir/<experiment>_w2FLUO DAPI_s{pos}_t{t}.TIF
        raw_dir/<experiment>_w1TRANS_s{pos}_t{t}.TIF

    Returns
    -------
    dapi_imgs  : list of (H,W) uint16 arrays (downscaled)
    trans_imgs : list of (H,W) uint16 arrays (downscaled)
    timepoints : list of int timepoint indices
    """
    import glob
    import re
    from tifffile import imread

    dapi_paths = sorted(
        glob.glob(str(raw_dir) + f"/*w2FLUO DAPI*_s{position}_t*.TIF"),
        key=lambda p: int(re.search(r"_t(\d+)\.TIF$", p, re.I).group(1)),
    )
    if not dapi_paths:
        # Try case-insensitive extension
        dapi_paths = sorted(
            glob.glob(str(raw_dir) + f"/*w2FLUO DAPI*_s{position}_t*.tif"),
            key=lambda p: int(re.search(r"_t(\d+)\.tif$", p, re.I).group(1)),
        )
    if not dapi_paths:
        raise FileNotFoundError(
            f"No DAPI images found in {raw_dir} for position {position}.\n"
            "Expected pattern: *w2FLUO DAPI*_s{pos}_t*.TIF"
        )

    trans_paths = [p.replace("w2FLUO DAPI", "w1TRANS") for p in dapi_paths]
    missing = [p for p in trans_paths if not __import__("os").path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing TRANS files: {missing[:3]}")

    timepoints = [int(re.search(r"_t(\d+)\.TIF$", p, re.I).group(1)) for p in dapi_paths]

    dapi_imgs  = [downscale(imread(p), downscale_factor).astype(np.uint16) for p in dapi_paths]
    trans_imgs = [downscale(imread(p), downscale_factor).astype(np.uint16) for p in trans_paths]

    return dapi_imgs, trans_imgs, timepoints


def discover_positions(raw_dir):
    """Return sorted list of stage positions found in raw_dir."""
    import glob
    import re
    paths = glob.glob(str(raw_dir) + "/*w2FLUO DAPI*_s*_t*.TIF")
    paths += glob.glob(str(raw_dir) + "/*w2FLUO DAPI*_s*_t*.tif")
    positions = sorted({int(re.search(r"_s(\d+)_t", p).group(1)) for p in paths})
    return positions
