"""Render two presentation movies for pos08.

Movie 1 — "channels": cell, nucleus, and NLS z-averages composited on a white
background. Each channel contributes a subtractive tint (cells → grey,
nucleus → orange, NLS → blue), so co-localisation darkens correctly.

Movie 2 — "analysis": tracked cell/nucleus labels + cell–cell contact contours
+ nucleus tracks, all coloured by NLS class (orange = ctrl, blue = vimentin_ko)
with teal for heterotypic contacts.

Outputs are written next to this script:
    pos08_channels.mp4
    pos08_analysis.mp4
"""

from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import tifffile
from PIL import Image
from skimage.draw import line_aa

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

POS = Path(
    "/home/aruppel/Data/2026-04-30_U251-WT-NLS-mCherry_U251-VimentinKO_"
    "circle300um_live_spinning-disk/analysis/pos08"
)
CELL_ZAVG = POS / "0_input" / "cell_zavg.tif"
NUC_ZAVG = POS / "0_input" / "nucleus_zavg.tif"
NLS_ZAVG = POS / "0_input" / "NLS_zavg.tif"
NUC_LABELS = POS / "2_nucleus" / "tracked_labels.tif"
CELL_LABELS = POS / "3_cell" / "tracked_labels.tif"
H5 = POS / "4_contact_analysis" / "contact_analysis.h5"

OUT_DIR = Path(__file__).resolve().parent
OUT_CHANNELS = OUT_DIR / "pos08_channels.mp4"
OUT_ANALYSIS = OUT_DIR / "pos08_analysis.mp4"
OUT_ANALYSIS_T1 = OUT_DIR / "pos08_analysis_t1.mp4"

FPS = 10
SUPER = 2  # supersampling factor for the analysis movie (anti-aliasing)
TRACK_TAIL = 40
FADE_ALPHA = 0.20  # opacity of the faded base for the T1 movie
T1_LOSING_COLOR = np.array([1.00, 0.05, 0.10])   # red — edge about to disappear
T1_GAINING_COLOR = np.array([0.05, 0.85, 0.20])  # green — newly formed edge

# Class / tint colours (RGB 0–1)
COL_CTRL = np.array([1.00, 0.55, 0.00])          # orange
COL_VIM = np.array([0.10, 0.45, 0.95])           # blue
COL_GREY = np.array([0.45, 0.45, 0.45])          # cells tint
COL_HOMO_CTRL = np.array([0.85, 0.35, 0.00])     # saturated orange (contour)
COL_HOMO_VIM = np.array([0.00, 0.30, 0.80])      # saturated blue   (contour)
COL_HETERO = np.array([0.00, 0.65, 0.65])        # teal             (contour)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def stretch(
    stack: np.ndarray,
    lo_pct: float = 0.0,
    hi_pct: float = 99.0,
    gain: float = 1.0,
) -> np.ndarray:
    """Percentile-clip → [0, 1]; `lo_pct` removes background, `gain<1` softens contrast."""
    arr = stack.astype(np.float32)
    lo = float(np.percentile(arr, lo_pct)) if lo_pct > 0 else float(arr.min())
    hi = float(np.percentile(arr, hi_pct))
    if hi <= lo:
        return np.zeros_like(arr)
    out = (arr - lo) / (hi - lo)
    return np.clip(out * gain, 0.0, 1.0)


def encode_mp4(frames_rgb_uint8: np.ndarray, out_path: Path, fps: int) -> None:
    """Pipe (T, H, W, 3) uint8 frames to ffmpeg → libx264 mp4."""
    T, H, W, _ = frames_rgb_uint8.shape
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-preset", "slow",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.stdin.write(np.ascontiguousarray(frames_rgb_uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")


def downsample(rgb: np.ndarray, factor: int) -> np.ndarray:
    """Bicubic-downsample one (H, W, 3) float32 frame by `factor`."""
    if factor == 1:
        return rgb
    H, W, _ = rgb.shape
    img = Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))
    img = img.resize((W // factor, H // factor), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Movie 1 — subtractive channel overlay
# ---------------------------------------------------------------------------


def render_channels_movie() -> None:
    # cells: subtract background, full 99-pct stretch
    cells = stretch(tifffile.imread(CELL_ZAVG), lo_pct=15.0, hi_pct=99.0)
    # nucleus / NLS: stronger background subtraction + softer contrast (gain < 1)
    nuc = stretch(tifffile.imread(NUC_ZAVG), lo_pct=30.0, hi_pct=99.0, gain=0.7)
    nls = stretch(tifffile.imread(NLS_ZAVG), lo_pct=30.0, hi_pct=99.0, gain=0.7)

    T, H, W = cells.shape
    # subtractive tints — what each channel removes from white
    tint_cells = 1.0 - COL_GREY
    tint_nuc = 1.0 - COL_CTRL
    tint_nls = 1.0 - COL_VIM

    out = np.empty((T, H, W, 3), dtype=np.uint8)
    for t in range(T):
        sub = (
            cells[t, ..., None] * tint_cells
            + nuc[t, ..., None] * tint_nuc
            + nls[t, ..., None] * tint_nls
        )
        rgb = np.clip(1.0 - sub, 0.0, 1.0)
        out[t] = (rgb * 255.0 + 0.5).astype(np.uint8)
    encode_mp4(out, OUT_CHANNELS, FPS)
    print(f"wrote {OUT_CHANNELS}")


# ---------------------------------------------------------------------------
# Movie 2 — analysis overlay
# ---------------------------------------------------------------------------


def read_h5_tables() -> tuple[dict, dict]:
    with h5py.File(H5, "r") as f:
        cells = {
            "frame": f["cells/table/frame"][:],
            "cell_id": f["cells/table/cell_id"][:],
            "class_label": f["cells/table/class_label"].asstr()[:],
        }
        edges = {
            "frame": f["edges/table/frame"][:],
            "cell_a": f["edges/table/cell_a"][:],
            "cell_b": f["edges/table/cell_b"][:],
            "kind": f["edges/table/kind"].asstr()[:],
            "coord_offset": f["edges/table/coord_offset"][:],
            "coord_count": f["edges/table/coord_count"][:],
            "coord_y": f["edges/coordinates/y"][:],
            "coord_x": f["edges/coordinates/x"][:],
            "t1_event_id": f["edges/table/t1_event_id"][:],
        }
    return cells, edges


def read_t1_events() -> dict:
    with h5py.File(H5, "r") as f:
        return {
            "frame": f["t1_events/table/frame"][:],
            "t1_event_id": f["t1_events/table/t1_event_id"][:],
            "losing_cell_a": f["t1_events/table/losing_cell_a"][:],
            "losing_cell_b": f["t1_events/table/losing_cell_b"][:],
            "gaining_cell_a": f["t1_events/table/gaining_cell_a"][:],
            "gaining_cell_b": f["t1_events/table/gaining_cell_b"][:],
        }


def build_class_map(cells: dict) -> dict[int, str]:
    """One class per cell_id (track) — first non-empty wins."""
    out: dict[int, str] = {}
    for cid, lbl in zip(cells["cell_id"], cells["class_label"]):
        cid = int(cid)
        if lbl and cid not in out:
            out[cid] = str(lbl)
    return out


def class_color(label: str) -> np.ndarray:
    # Class colours are swapped vs the channels-movie palette: ctrl → blue, vimentin_ko → orange.
    return COL_VIM if label == "ctrl" else COL_CTRL if label == "vimentin_ko" else np.array([0.6, 0.6, 0.6])


def contour_color(label_a: str, label_b: str) -> np.ndarray:
    if label_a == label_b == "ctrl":
        return COL_HOMO_VIM
    if label_a == label_b == "vimentin_ko":
        return COL_HOMO_CTRL
    return COL_HETERO


def label_color_image(
    labels: np.ndarray,
    color_lut: np.ndarray,
    alpha_lut: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised lookup: returns (H, W, 3) colour and (H, W) alpha."""
    color = color_lut[labels]
    alpha = alpha_lut[labels]
    return color, alpha


def draw_aa_polyline(
    rgb: np.ndarray,
    alpha: np.ndarray,  # unused; kept for signature compatibility
    ys: np.ndarray,
    xs: np.ndarray,
    color: np.ndarray,
    width: int = 2,
    alpha_scale: float = 1.0,
) -> None:
    """Draw anti-aliased polyline directly over `rgb` (opaque base, no alpha channel)."""
    H, W, _ = rgb.shape
    if len(ys) < 2:
        return
    color = np.asarray(color, dtype=np.float32)
    for i in range(len(ys) - 1):
        y0, x0 = int(round(ys[i])), int(round(xs[i]))
        y1, x1 = int(round(ys[i + 1])), int(round(xs[i + 1]))
        rr, cc, val = line_aa(y0, x0, y1, x1)
        m = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        rr, cc, val = rr[m], cc[m], val[m]
        for dy in range(width):
            for dx in range(width):
                rr2 = rr + dy
                cc2 = cc + dx
                m2 = (rr2 < H) & (cc2 < W)
                rr3, cc3, val3 = rr2[m2], cc2[m2], val[m2]
                src_a = val3.astype(np.float32) * alpha_scale
                if src_a.size == 0:
                    continue
                a = src_a[:, None]
                dst_rgb = rgb[rr3, cc3]
                rgb[rr3, cc3] = color * a + dst_rgb * (1.0 - a)


def nucleus_track_centroids(nuc_labels: np.ndarray) -> dict[int, list[tuple[int, float, float]]]:
    """Per-track centroid trajectory: {cell_id → [(frame, y, x), …]}."""
    T = nuc_labels.shape[0]
    tracks: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for t in range(T):
        frame = nuc_labels[t]
        flat = frame.ravel()
        order = np.argsort(flat, kind="stable")
        sorted_ids = flat[order]
        change = np.empty(len(sorted_ids), dtype=bool)
        change[0] = True
        np.not_equal(sorted_ids[1:], sorted_ids[:-1], out=change[1:])
        boundaries = np.flatnonzero(change)
        rows_all, cols_all = np.divmod(order, frame.shape[1])
        ends = np.empty_like(boundaries)
        ends[:-1] = boundaries[1:]
        ends[-1] = len(sorted_ids)
        for bi in range(len(boundaries)):
            cid = int(sorted_ids[boundaries[bi]])
            if cid == 0:
                continue
            s, e = int(boundaries[bi]), int(ends[bi])
            tracks[cid].append(
                (t, float(rows_all[s:e].mean()), float(cols_all[s:e].mean()))
            )
    return tracks


def composite_over(base_rgb: np.ndarray, layer_rgb: np.ndarray, layer_a: np.ndarray) -> np.ndarray:
    """Composite `layer` over `base` (both float32, 0–1)."""
    a = layer_a[..., None]
    return layer_rgb * a + base_rgb * (1.0 - a)


def render_analysis_movie() -> None:
    cells_tbl, edges_tbl = read_h5_tables()
    class_map = build_class_map(cells_tbl)

    nuc_labels = tifffile.imread(NUC_LABELS)
    cell_labels = tifffile.imread(CELL_LABELS)
    T, H, W = nuc_labels.shape
    Hs, Ws = H * SUPER, W * SUPER

    # ---- label-id → colour LUT
    max_nuc = int(nuc_labels.max())
    max_cell = int(cell_labels.max())
    nuc_color_lut = np.zeros((max_nuc + 1, 3), dtype=np.float32)
    nuc_alpha_lut = np.zeros((max_nuc + 1,), dtype=np.float32)
    for cid, lbl in class_map.items():
        if cid <= max_nuc:
            nuc_color_lut[cid] = class_color(lbl)
            nuc_alpha_lut[cid] = 0.65
    cell_color_lut = np.zeros((max_cell + 1, 3), dtype=np.float32)
    cell_alpha_lut = np.zeros((max_cell + 1,), dtype=np.float32)
    for cid, lbl in class_map.items():
        if cid <= max_cell:
            cell_color_lut[cid] = class_color(lbl)
            cell_alpha_lut[cid] = 0.18

    # ---- track segments by end frame
    centroids = nucleus_track_centroids(nuc_labels)
    seg_by_end: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for cid, rows in centroids.items():
        for prev, curr in zip(rows[:-1], rows[1:]):
            sf, sy, sx = prev
            ef, ey, ex = curr
            if ef != sf + 1:
                continue
            seg_by_end[ef].append((cid, sy, sx, ey, ex))

    # ---- index edges by frame for fast access
    frames_e = edges_tbl["frame"].astype(int)
    edges_by_frame: dict[int, list[int]] = defaultdict(list)
    for idx, f in enumerate(frames_e):
        edges_by_frame[int(f)].append(idx)

    out = np.empty((T, H, W, 3), dtype=np.uint8)

    for t in range(T):
        # --- low-res (1×) base: white + label fills (vectorised LUT)
        base = np.ones((H, W, 3), dtype=np.float32)
        cell_rgb, cell_a = label_color_image(cell_labels[t], cell_color_lut, cell_alpha_lut)
        base = composite_over(base, cell_rgb, cell_a)
        nuc_rgb, nuc_a = label_color_image(nuc_labels[t], nuc_color_lut, nuc_alpha_lut)
        base = composite_over(base, nuc_rgb, nuc_a)

        # --- upsample base to supersampled canvas (nearest is fine — labels are blocky already)
        base_img = Image.fromarray((base * 255.0 + 0.5).astype(np.uint8))
        base_hi = np.asarray(
            base_img.resize((Ws, Hs), Image.BILINEAR), dtype=np.float32
        ) / 255.0

        # --- contours (anti-aliased) at supersampled resolution
        line_alpha = np.zeros((Hs, Ws), dtype=np.float32)
        # we composite contours directly onto base_hi
        kinds = edges_tbl["kind"]
        for idx in edges_by_frame.get(t, []):
            if kinds[idx] != "cell_cell":
                continue  # skip border contours
            ca = int(edges_tbl["cell_a"][idx])
            cb = int(edges_tbl["cell_b"][idx])
            la = class_map.get(ca, "")
            lb = class_map.get(cb, "")
            col = contour_color(la, lb)
            o = int(edges_tbl["coord_offset"][idx])
            n = int(edges_tbl["coord_count"][idx])
            if n < 2:
                continue
            ys = edges_tbl["coord_y"][o:o + n] * SUPER
            xs = edges_tbl["coord_x"][o:o + n] * SUPER
            draw_aa_polyline(base_hi, line_alpha, ys, xs, col, width=2 * SUPER)

        # --- tracks (fade tail) at supersampled resolution
        track_alpha = np.zeros((Hs, Ws), dtype=np.float32)
        start = max(0, t - TRACK_TAIL)
        for end_frame in range(start + 1, t + 1):
            age = t - end_frame
            # linear fade to fully transparent at the tail end
            tail_alpha = max(0.0, 1.0 - age / TRACK_TAIL)
            if tail_alpha <= 0.0:
                continue
            for cid, sy, sx, ey, ex in seg_by_end.get(end_frame, []):
                col = class_color(class_map.get(cid, ""))
                ys = np.array([sy, ey]) * SUPER
                xs = np.array([sx, ex]) * SUPER
                draw_aa_polyline(
                    base_hi,
                    track_alpha,
                    ys,
                    xs,
                    col,
                    width=max(1, SUPER // 2),  # thinner: ~0.5 px at output resolution
                    alpha_scale=tail_alpha,
                )

        # --- downsample back to original resolution
        frame_rgb = downsample(base_hi, SUPER)
        out[t] = (np.clip(frame_rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        if (t + 1) % 5 == 0 or t == T - 1:
            print(f"  analysis frame {t + 1}/{T}")

    encode_mp4(out, OUT_ANALYSIS, FPS)
    print(f"wrote {OUT_ANALYSIS}")


# ---------------------------------------------------------------------------
# Movie 3 — faded analysis with T1 edge highlights
# ---------------------------------------------------------------------------


def render_analysis_t1_movie() -> None:
    cells_tbl, edges_tbl = read_h5_tables()
    t1_tbl = read_t1_events()
    class_map = build_class_map(cells_tbl)

    nuc_labels = tifffile.imread(NUC_LABELS)
    cell_labels = tifffile.imread(CELL_LABELS)
    T, H, W = nuc_labels.shape
    Hs, Ws = H * SUPER, W * SUPER

    max_nuc = int(nuc_labels.max())
    max_cell = int(cell_labels.max())
    nuc_color_lut = np.zeros((max_nuc + 1, 3), dtype=np.float32)
    nuc_alpha_lut = np.zeros((max_nuc + 1,), dtype=np.float32)
    for cid, lbl in class_map.items():
        if cid <= max_nuc:
            nuc_color_lut[cid] = class_color(lbl)
            nuc_alpha_lut[cid] = 0.65
    cell_color_lut = np.zeros((max_cell + 1, 3), dtype=np.float32)
    cell_alpha_lut = np.zeros((max_cell + 1,), dtype=np.float32)
    for cid, lbl in class_map.items():
        if cid <= max_cell:
            cell_color_lut[cid] = class_color(lbl)
            cell_alpha_lut[cid] = 0.18

    centroids = nucleus_track_centroids(nuc_labels)
    seg_by_end: dict[int, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    for cid, rows in centroids.items():
        for prev, curr in zip(rows[:-1], rows[1:]):
            sf, sy, sx = prev
            ef, ey, ex = curr
            if ef != sf + 1:
                continue
            seg_by_end[ef].append((cid, sy, sx, ey, ex))

    frames_e = edges_tbl["frame"].astype(int)
    edges_by_frame: dict[int, list[int]] = defaultdict(list)
    for idx, f in enumerate(frames_e):
        edges_by_frame[int(f)].append(idx)

    # T1 highlights: a transition between frames F and F+1 — at frame F the
    # losing edge has its last appearance; at frame F+1 the gaining edge first
    # appears. Index edges by (frame, sorted cell pair) so we can fetch each.
    edge_a = edges_tbl["cell_a"]
    edge_b = edges_tbl["cell_b"]
    pair_to_edge_idx: dict[tuple[int, int, int], int] = {}
    for idx in range(len(frames_e)):
        a, b = int(edge_a[idx]), int(edge_b[idx])
        if a > b:
            a, b = b, a
        pair_to_edge_idx[(int(frames_e[idx]), a, b)] = idx

    # frame -> [(edge_idx, color), ...]
    t1_highlight: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for k in range(len(t1_tbl["frame"])):
        F = int(t1_tbl["frame"][k])
        la, lb = int(t1_tbl["losing_cell_a"][k]), int(t1_tbl["losing_cell_b"][k])
        ga, gb = int(t1_tbl["gaining_cell_a"][k]), int(t1_tbl["gaining_cell_b"][k])
        if la > lb:
            la, lb = lb, la
        if ga > gb:
            ga, gb = gb, ga
        if 0 <= F < T:
            li = pair_to_edge_idx.get((F, la, lb))
            if li is not None:
                t1_highlight[F].append((li, T1_LOSING_COLOR))
        if 0 <= F + 1 < T:
            gi = pair_to_edge_idx.get((F + 1, ga, gb))
            if gi is not None:
                t1_highlight[F + 1].append((gi, T1_GAINING_COLOR))

    out = np.empty((T, H, W, 3), dtype=np.uint8)

    for t in range(T):
        base = np.ones((H, W, 3), dtype=np.float32)
        cell_rgb, cell_a = label_color_image(cell_labels[t], cell_color_lut, cell_alpha_lut)
        base = composite_over(base, cell_rgb, cell_a)
        nuc_rgb, nuc_a = label_color_image(nuc_labels[t], nuc_color_lut, nuc_alpha_lut)
        base = composite_over(base, nuc_rgb, nuc_a)

        base_img = Image.fromarray((base * 255.0 + 0.5).astype(np.uint8))
        base_hi = np.asarray(
            base_img.resize((Ws, Hs), Image.BILINEAR), dtype=np.float32
        ) / 255.0

        line_alpha = np.zeros((Hs, Ws), dtype=np.float32)
        kinds = edges_tbl["kind"]
        for idx in edges_by_frame.get(t, []):
            if kinds[idx] != "cell_cell":
                continue
            ca = int(edges_tbl["cell_a"][idx])
            cb = int(edges_tbl["cell_b"][idx])
            la = class_map.get(ca, "")
            lb = class_map.get(cb, "")
            col = contour_color(la, lb)
            o = int(edges_tbl["coord_offset"][idx])
            n = int(edges_tbl["coord_count"][idx])
            if n < 2:
                continue
            ys = edges_tbl["coord_y"][o:o + n] * SUPER
            xs = edges_tbl["coord_x"][o:o + n] * SUPER
            draw_aa_polyline(base_hi, line_alpha, ys, xs, col, width=2 * SUPER)

        track_alpha = np.zeros((Hs, Ws), dtype=np.float32)
        start = max(0, t - TRACK_TAIL)
        for end_frame in range(start + 1, t + 1):
            age = t - end_frame
            tail_alpha = max(0.0, 1.0 - age / TRACK_TAIL)
            if tail_alpha <= 0.0:
                continue
            for cid, sy, sx, ey, ex in seg_by_end.get(end_frame, []):
                col = class_color(class_map.get(cid, ""))
                ys = np.array([sy, ey]) * SUPER
                xs = np.array([sx, ex]) * SUPER
                draw_aa_polyline(
                    base_hi,
                    track_alpha,
                    ys,
                    xs,
                    col,
                    width=max(1, SUPER // 2),
                    alpha_scale=tail_alpha,
                )

        # Fade everything to FADE_ALPHA over a white background.
        base_hi = 1.0 - FADE_ALPHA * (1.0 - base_hi)

        # Overlay T1 edges at full opacity: red = losing, green = gaining.
        for idx, col in t1_highlight.get(t, []):
            o = int(edges_tbl["coord_offset"][idx])
            n = int(edges_tbl["coord_count"][idx])
            if n < 2:
                continue
            ys = edges_tbl["coord_y"][o:o + n] * SUPER
            xs = edges_tbl["coord_x"][o:o + n] * SUPER
            draw_aa_polyline(
                base_hi,
                line_alpha,
                ys,
                xs,
                col,
                width=3 * SUPER,
            )

        frame_rgb = downsample(base_hi, SUPER)
        out[t] = (np.clip(frame_rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        if (t + 1) % 5 == 0 or t == T - 1:
            print(f"  analysis-t1 frame {t + 1}/{T}")

    encode_mp4(out, OUT_ANALYSIS_T1, FPS)
    print(f"wrote {OUT_ANALYSIS_T1}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    print("rendering channels movie …")
    render_channels_movie()
    print("rendering analysis movie …")
    render_analysis_movie()
    print("rendering analysis-t1 movie …")
    render_analysis_t1_movie()
