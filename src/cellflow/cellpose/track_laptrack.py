"""Link per-frame native masks across time with laptrack, Qt-free.

The standalone ``cellflow-cellpose`` tool segments each frame independently
(:mod:`cellflow.cellpose.native_masks`), producing labels that are *not*
consistent across time. This module closes that gap: it computes per-frame
centroids, runs a linear-assignment tracker (``laptrack``) to link objects
between consecutive frames, and relabels the stack so a tracked object keeps one
id over its whole lifetime.

``laptrack`` (and ``pandas``) are imported lazily inside :func:`track_masks`, so
importing this module — and the centroid / relabel helpers, which only need
numpy + scikit-image — does not require the optional ``[laptrack]`` extra.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "build_track_dataframe",
    "relabel_by_tracks",
    "stitch_z",
    "track_masks",
    "track_axiswise",
    "COORDINATE_COLS",
]

#: Centroid coordinate columns used for linking; ``z`` is constant for 2D input.
COORDINATE_COLS = ["z", "y", "x"]


def build_track_dataframe(masks_tzyx: np.ndarray):
    """Build a per-object centroid table from a ``(T, Z, Y, X)`` label stack.

    Returns a :class:`pandas.DataFrame` with columns ``frame, label, z, y, x``
    (one row per labelled object per frame). ``z`` is always present and is
    constant ``0`` for single-slice (2D) input, so linking is uniform.
    """
    import pandas as pd
    from skimage.measure import regionprops_table

    if masks_tzyx.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X), got shape {masks_tzyx.shape}")
    frames, labels, zs, ys, xs = [], [], [], [], []
    for t in range(masks_tzyx.shape[0]):
        volume = np.asarray(masks_tzyx[t])
        if not volume.any():
            continue
        table = regionprops_table(volume, properties=("label", "centroid"))
        n = len(table["label"])
        # centroid-0/1/2 -> z/y/x (3D volume always yields three centroid cols).
        frames.append(np.full(n, t, dtype=np.int64))
        labels.append(table["label"])
        zs.append(table.get("centroid-0", np.zeros(n)))
        ys.append(table.get("centroid-1", np.zeros(n)))
        xs.append(table.get("centroid-2", np.zeros(n)))
    if not frames:
        return pd.DataFrame(columns=["frame", "label", "z", "y", "x"])
    return pd.DataFrame(
        {
            "frame": np.concatenate(frames),
            "label": np.concatenate(labels).astype(int),
            "z": np.concatenate(zs).astype(float),
            "y": np.concatenate(ys).astype(float),
            "x": np.concatenate(xs).astype(float),
        }
    )[["frame", "label", "z", "y", "x"]]


def relabel_by_tracks(masks_tzyx: np.ndarray, track_of: dict) -> np.ndarray:
    """Relabel ``(T, Z, Y, X)`` masks by a ``(frame, orig_label) -> track_id`` map.

    Output labels are ``track_id + 1`` (so tracks start at ``1`` and background
    stays ``0``). Original labels absent from ``track_of`` are dropped to ``0``.
    """
    if masks_tzyx.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X), got shape {masks_tzyx.shape}")
    out = np.zeros_like(masks_tzyx, dtype=np.int32)
    for t in range(masks_tzyx.shape[0]):
        frame = np.asarray(masks_tzyx[t])
        max_label = int(frame.max())
        if max_label == 0:
            continue
        lut = np.zeros(max_label + 1, dtype=np.int32)
        for orig_label in range(1, max_label + 1):
            track_id = track_of.get((t, orig_label))
            if track_id is not None:
                lut[orig_label] = int(track_id) + 1
        out[t] = lut[frame]
    return out


def _stitch_volume(vol_zyx: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Merge per-plane labels of one ``(Z, Y, X)`` volume into z-coherent objects.

    Each z-plane is independently labelled (frame-unique ids). Two labels in
    adjacent z-planes whose footprints overlap by ``IoU > iou_threshold`` are the
    same 3-D object; they are unioned and the volume relabelled with compact ids.
    """
    vol = np.asarray(vol_zyx)
    Z = vol.shape[0]
    max_label = int(vol.max())
    if max_label == 0:
        return np.zeros_like(vol, dtype=np.int32)

    counts = np.bincount(vol.reshape(-1), minlength=max_label + 1)
    parent = np.arange(max_label + 1, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    stride = max_label + 1
    for z in range(Z - 1):
        a = vol[z]
        b = vol[z + 1]
        both = (a > 0) & (b > 0)
        if not both.any():
            continue
        av = a[both].astype(np.int64)
        bv = b[both].astype(np.int64)
        keys, inter = np.unique(av * stride + bv, return_counts=True)
        for key, n in zip(keys, inter):
            la = int(key // stride)
            lb = int(key % stride)
            union_area = counts[la] + counts[lb] - n
            if union_area > 0 and (n / union_area) > iou_threshold:
                union(la, lb)

    # Compact the surviving roots to 1..K (background 0 stays 0).
    lut = np.zeros(max_label + 1, dtype=np.int32)
    new_id: dict[int, int] = {}
    for label in range(1, max_label + 1):
        if counts[label] == 0:
            continue
        root = find(label)
        if root not in new_id:
            new_id[root] = len(new_id) + 1
        lut[label] = new_id[root]
    return lut[vol]


def stitch_z(masks_tzyx: np.ndarray, *, iou_threshold: float = 0.25) -> np.ndarray:
    """Stitch per-plane labels into z-coherent 3-D objects, per timepoint.

    Input ``(T, Z, Y, X)`` masks are labelled independently in every z-plane; this
    links labels across adjacent z by IoU so an object spanning several planes
    shares one id within its frame (background ``0``). For ``Z == 1`` it is a
    no-op (returns an int32 copy). Cross-time uniqueness is left to the tracker.
    """
    masks = np.asarray(masks_tzyx)
    if masks.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X), got shape {masks.shape}")
    if masks.shape[1] == 1:
        return masks.astype(np.int32, copy=True)
    out = np.zeros_like(masks, dtype=np.int32)
    for t in range(masks.shape[0]):
        out[t] = _stitch_volume(masks[t], iou_threshold)
    return out


def _track_ids_from_tree(tree) -> dict:
    """Map each ``(frame, index)`` tree node to a track id via connected components.

    Only valid when splitting/merging are disabled (as they always are in
    :func:`_run_laptrack`): every node then has at most one predecessor and one
    successor, so the raw tree's connected components are exactly the tracks —
    matching what ``laptrack``'s own ``tree_id``/``track_id`` computation would
    give here, without going through its per-node assignment (see
    :func:`_run_laptrack` docstring). Guards against silently wrong ids if that
    assumption is ever broken by raising on unexpected branching.
    """
    import networkx as nx

    if any(tree.out_degree(n) > 1 or tree.in_degree(n) > 1 for n in tree.nodes):
        raise AssertionError(
            "track tree has branching (split/merge); _run_laptrack assumes "
            "splitting/merging stay disabled"
        )
    track_id_by_node: dict = {}
    for track_id, nodes in enumerate(nx.connected_components(nx.Graph(tree))):
        for node in nodes:
            track_id_by_node[node] = track_id
    return track_id_by_node


def _run_laptrack(df, *, max_distance: float, max_frame_gap: int):
    """Run laptrack on a centroid dataframe; return it with a ``track_id`` column.

    Calls ``LapTrack.predict()`` (the public method returning the raw tracking
    graph) rather than ``predict_dataframe()``. The latter's own conversion step
    (``laptrack.data_conversion.tree_to_dataframe``) assigns ``tree_id`` and
    ``track_id`` to every node **one at a time** via ``DataFrame.loc[...] =``
    (the library's own source flags this with "XXX there may exist faster
    impl."); on a dense stack (profiled: 150 frames x ~576 cells/frame) those
    per-node pandas writes accounted for >80% of total tracking time, dwarfing
    laptrack's actual linking cost. :func:`_track_ids_from_tree` gets the same
    result via a dict lookup instead.

    Isolated so :func:`track_masks` orchestration can be tested without the
    optional dependency installed (tests monkeypatch this).
    """
    from laptrack import LapTrack
    from laptrack.data_conversion import dataframe_to_coords_frame_index

    cutoff = float(max_distance) ** 2  # sqeuclidean metric
    lt = LapTrack(
        track_dist_metric="sqeuclidean",
        track_cost_cutoff=cutoff,
        gap_closing_dist_metric="sqeuclidean",
        gap_closing_cost_cutoff=cutoff if max_frame_gap > 0 else False,
        gap_closing_max_frame_count=int(max_frame_gap),
        splitting_cost_cutoff=False,
        merging_cost_cutoff=False,
    )
    coords, frame_index = dataframe_to_coords_frame_index(
        df, COORDINATE_COLS, frame_col="frame"
    )
    tree = lt.predict(coords)
    track_id_by_node = _track_ids_from_tree(tree)
    track_df = df.reset_index(drop=True).copy()
    track_df["track_id"] = [track_id_by_node[node] for node in frame_index]
    return track_df


def track_masks(
    masks_tzyx: np.ndarray,
    *,
    max_distance: float = 15.0,
    max_frame_gap: int = 0,
) -> np.ndarray:
    """Link per-frame masks across time and return a track-consistent stack.

    ``max_distance`` is the maximum centroid displacement (in pixels, over
    ``z/y/x``) allowed for a link; ``max_frame_gap`` > 0 enables gap-closing over
    that many missed frames. Output is ``(T, Z, Y, X)`` ``int32`` with labels
    stable across time (background ``0``).
    """
    if masks_tzyx.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X), got shape {masks_tzyx.shape}")
    df = build_track_dataframe(masks_tzyx)
    if df.empty:
        return np.zeros_like(masks_tzyx, dtype=np.int32)
    tracked_df = _run_laptrack(df, max_distance=max_distance, max_frame_gap=max_frame_gap)
    track_of = {
        (int(r.frame), int(r.label)): int(r.track_id)
        for r in tracked_df.itertuples(index=False)
    }
    return relabel_by_tracks(masks_tzyx, track_of)


def track_axiswise(
    masks_tzyx: np.ndarray,
    *,
    max_distance: float = 15.0,
    max_frame_gap: int = 0,
    stitch_iou: float = 0.25,
) -> np.ndarray:
    """Axis-by-axis linking: **stitch z (overlap) then track t (motion)**.

    Per-plane native masks are first stitched through z by IoU
    (:func:`stitch_z`) so an object becomes one 3-D label per frame, then linked
    across time by centroid (:func:`track_masks`). For single-slice (``Z == 1``)
    input the stitch is a no-op and this reduces to plain time tracking.

    Stitching z (where adjacent planes *overlap*) and tracking t (where objects
    *move*) use the right metric for each axis; the shorter leading axis is taken
    as z upstream (see :func:`cellflow.cellpose.shape.to_canonical_tzyx`).
    """
    stitched = stitch_z(masks_tzyx, iou_threshold=stitch_iou)
    return track_masks(
        stitched, max_distance=max_distance, max_frame_gap=max_frame_gap
    )
