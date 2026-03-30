"""
Core consensus segmentation — no Qt, no napari imports.

match_masks_iou  : pairwise IoU matching between two 2-D label arrays
consensus_frame  : aggregate N plane masks into one consensus mask per timepoint
consensus_stack  : apply consensus_frame across all timepoints
consensus_all    : vote across all (config × z-plane) pairs per timepoint
temporal_filter  : prune cells that are temporally inconsistent
"""

import numpy as np
from collections import defaultdict


def match_masks_iou(masks_a, masks_b, iou_threshold):
    """Return {label_in_a: label_in_b} for pairs with IoU > iou_threshold.

    Each label in masks_a is matched to at most one label in masks_b
    (the one with the highest IoU that exceeds the threshold).
    """
    labels_a = np.unique(masks_a[masks_a > 0])
    labels_b = np.unique(masks_b[masks_b > 0])
    if len(labels_a) == 0 or len(labels_b) == 0:
        return {}

    Na = len(labels_a)
    Nb = len(labels_b)

    # Re-index labels to 1..Na and 1..Nb for a compact intersection matrix.
    a_map = np.zeros(int(masks_a.max()) + 1, dtype=np.int32)
    b_map = np.zeros(int(masks_b.max()) + 1, dtype=np.int32)
    for i, la in enumerate(labels_a, 1):
        a_map[la] = i
    for i, lb in enumerate(labels_b, 1):
        b_map[lb] = i

    ma = a_map[masks_a]
    mb = b_map[masks_b]

    # counts[ia, ib] = pixel overlap between re-indexed cell ia (in a) and ib (in b)
    flat = ma.ravel().astype(np.int64) * (Nb + 1) + mb.ravel().astype(np.int64)
    counts = np.bincount(flat, minlength=(Na + 1) * (Nb + 1)).reshape(Na + 1, Nb + 1)

    areas_a = counts[1:, :].sum(axis=1)   # (Na,)
    areas_b = counts[:, 1:].sum(axis=0)   # (Nb,)

    matches = {}
    for ia in range(Na):
        best_iou = 0.0
        best_jb  = -1
        for jb in range(Nb):
            inter = int(counts[ia + 1, jb + 1])
            if inter == 0:
                continue
            union = int(areas_a[ia]) + int(areas_b[jb]) - inter
            iou   = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_jb  = jb
        if best_jb >= 0 and best_iou > iou_threshold:
            matches[int(labels_a[ia])] = int(labels_b[best_jb])

    return matches


def consensus_frame(plane_masks, min_votes, iou_threshold):
    """Aggregate N (H, W) masks into one consensus (H, W) mask.

    Steps:
      1. Match cells across all plane pairs via match_masks_iou.
      2. Group matched cells using union-find (connected components).
      3. Drop groups appearing in fewer distinct planes than min_votes.
      4. For surviving groups, keep the instance with the largest area.
      5. Re-label 1…N.

    Parameters
    ----------
    plane_masks   : list of (H, W) integer label arrays, one per Z-plane
    min_votes     : int  — minimum number of planes a cell must appear in
    iou_threshold : float — IoU threshold for declaring two masks the same cell

    Returns
    -------
    (H, W) int32 label array
    """
    if not plane_masks:
        raise ValueError("plane_masks must be non-empty")

    H, W = plane_masks[0].shape
    N    = len(plane_masks)

    # ── Union-Find ──────────────────────────────────────────────────────
    parent: dict = {}

    def find(x):
        root = x
        while parent.get(root, root) != root:
            root = parent.get(root, root)
        # Path compression
        while parent.get(x, x) != root:
            nxt = parent.get(x, x)
            parent[x] = root
            x = nxt
        return root

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # Register every cell
    for pi, masks in enumerate(plane_masks):
        for lbl in np.unique(masks[masks > 0]):
            node = (pi, int(lbl))
            parent[node] = node

    # Union matched cell pairs across all plane combinations
    for pi in range(N):
        for pj in range(pi + 1, N):
            pairs = match_masks_iou(plane_masks[pi], plane_masks[pj], iou_threshold)
            for la, lb in pairs.items():
                union((pi, la), (pj, lb))

    # Collect groups: root -> list of (plane_idx, label, area)
    groups: dict = defaultdict(list)
    for pi, masks in enumerate(plane_masks):
        for lbl in np.unique(masks[masks > 0]):
            node = (pi, int(lbl))
            root = find(node)
            area = int((masks == lbl).sum())
            groups[root].append((pi, int(lbl), area))

    # Build output: filter by votes, keep largest-area instance per group
    output    = np.zeros((H, W), dtype=np.int32)
    new_label = 0
    for root, members in groups.items():
        planes_present = len({pi for pi, _, _ in members})
        if planes_present < min_votes:
            continue
        best_pi, best_lbl, _ = max(members, key=lambda m: m[2])
        new_label += 1
        output[plane_masks[best_pi] == best_lbl] = new_label

    return output


def consensus_stack(plane_stacks, min_votes, iou_threshold, progress_callback=None):
    """Apply consensus_frame to every timepoint.

    Parameters
    ----------
    plane_stacks      : list of (T, H, W) arrays, one per Z-plane
    min_votes         : int
    iou_threshold     : float
    progress_callback : optional callable(t, T) called after each frame

    Returns
    -------
    (T, H, W) int32 consensus label array
    """
    T       = plane_stacks[0].shape[0]
    results = []
    for t in range(T):
        plane_masks = [stack[t] for stack in plane_stacks]
        results.append(consensus_frame(plane_masks, min_votes, iou_threshold))
        if progress_callback is not None:
            progress_callback(t + 1, T)
    return np.stack(results, axis=0).astype(np.int32)


def consensus_all(plane_stacks_per_config, min_votes, iou_threshold, progress_callback=None):
    """Vote across all (config × z-plane) pairs per timepoint.

    Treats every (config_index, z_index) combination as an independent voter
    for each timepoint and passes the full voter list to consensus_frame.
    With multiple parameter configs, this naturally resolves split/merge
    ambiguities: if most configs agree that two blobs are separate cells,
    the consensus will separate them, and vice versa.

    Parameters
    ----------
    plane_stacks_per_config : list of (T, Z, H, W) int32 arrays, one per config
    min_votes               : int — minimum voters a cell must appear in
    iou_threshold           : float — IoU threshold for mask matching
    progress_callback       : optional callable(t, T)

    Returns
    -------
    (T, H, W) int32 consensus labels
    """
    T = plane_stacks_per_config[0].shape[0]
    results = []
    for t in range(T):
        voters = [
            stack[t, z]
            for stack in plane_stacks_per_config
            for z in range(stack.shape[1])
        ]
        results.append(consensus_frame(voters, min_votes, iou_threshold))
        if progress_callback is not None:
            progress_callback(t + 1, T)
    return np.stack(results, axis=0).astype(np.int32)


def temporal_filter(masks_thw, window, min_votes, iou_threshold, progress_callback=None):
    """Filter (T, H, W) labels by temporal consistency.

    For each cell label at frame t, counts how many frames in the range
    [t − window, t + window] (excluding t itself) contain a cell whose IoU
    with that label exceeds iou_threshold.  Labels supported by fewer than
    min_votes neighboring frames are removed.

    This prunes both phantom splits (a single cell detected as two in a few
    frames — each phantom half fails because it only covers part of the
    consistent cell in other frames) and phantom merges (two cells fused in
    a few frames — the merged blob fails because it poorly matches either
    individual cell in neighboring frames).

    Parameters
    ----------
    masks_thw         : (T, H, W) int32 label array
    window            : int — half-window size in frames (looks ± window)
    min_votes         : int — minimum neighboring frames with a matching cell
    iou_threshold     : float
    progress_callback : optional callable(t, T)

    Returns
    -------
    (T, H, W) int32 filtered label array, relabelled 1…N per frame
    """
    T, H, W = masks_thw.shape
    output  = np.zeros_like(masks_thw)

    for t in range(T):
        frame_t  = masks_thw[t]
        labels_t = np.unique(frame_t[frame_t > 0])

        if len(labels_t) == 0:
            if progress_callback is not None:
                progress_callback(t + 1, T)
            continue

        nb_indices = [
            t2 for t2 in range(max(0, t - window), min(T, t + window + 1))
            if t2 != t
        ]

        if not nb_indices:
            # Single-frame stack: keep everything unchanged
            output[t] = frame_t
            if progress_callback is not None:
                progress_callback(t + 1, T)
            continue

        areas_t = {lbl: int((frame_t == lbl).sum()) for lbl in labels_t}
        keep    = set()

        for lbl in labels_t:
            mask_lbl = frame_t == lbl
            area_lbl = areas_t[lbl]
            votes    = 0
            for t2 in nb_indices:
                frame_nb   = masks_thw[t2]
                # Only evaluate cells that actually overlap the query mask
                candidates = np.unique(frame_nb[mask_lbl])
                candidates = candidates[candidates > 0]
                best_iou   = 0.0
                for nb_lbl in candidates:
                    mask_nb = frame_nb == nb_lbl
                    inter   = int((mask_lbl & mask_nb).sum())
                    union   = area_lbl + int(mask_nb.sum()) - inter
                    if union > 0:
                        best_iou = max(best_iou, inter / union)
                if best_iou >= iou_threshold:
                    votes += 1
            if votes >= min_votes:
                keep.add(lbl)

        frame_out = np.zeros((H, W), dtype=np.int32)
        new_lbl   = 0
        for lbl in sorted(keep):
            new_lbl += 1
            frame_out[frame_t == lbl] = new_lbl
        output[t] = frame_out

        if progress_callback is not None:
            progress_callback(t + 1, T)

    return output
