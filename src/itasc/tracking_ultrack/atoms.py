# src/itasc/tracking_ultrack/atoms.py
"""Atom extraction: residual-conditioned foreground split by contour ridges.

Stage ① of the atom-based candidate pipeline. Pure, deterministic functions
shared by the interactive preview and the full-stack ``atoms.tif`` writer.
"""
from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import numpy as np
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from scipy import ndimage as ndi
from skimage.segmentation import watershed

import tifffile

from itasc.core.imageops import residual

_PARAMS_KEY = "itasc_atom_params"
_FINGERPRINT_KEY = "itasc_atom_fingerprint"


@dataclass(frozen=True)
class AtomParams:
    """The knobs that fully determine an atom segmentation."""
    fg_window: int = 51
    fg_cutoff: float = 0.002
    fg_strength: float = 1.0
    contour_window: int = 51
    contour_floor: float = 0.01
    contour_strength: float = 1.0
    atom_min_area: int = 100


def extract_atoms_frame(
    residual_contour: np.ndarray,
    territory: np.ndarray,
    contour_floor: float,
    atom_min_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split ``territory`` into atoms along the cleaned contour ridge.

    ``ridge`` is where the residual contour exceeds ``contour_floor`` (a noise
    cutoff). Cores (ridge-free territory) seed a watershed that floods the
    residual-contour elevation, so a broken faint ridge still meets at the crest.
    Atoms smaller than ``atom_min_area`` are merged into the neighbouring atom
    they share the longest border with (see ``_merge_small_atoms``); an atom that
    touches no other label is kept as-is, so no territory pixel is ever blanked.

    Returns ``(atoms, ridge)``: ``atoms`` is the ``int32`` label image; ``ridge``
    is the boolean wall ``residual_contour > contour_floor`` returned as ``uint8``
    ``(Y, X)`` — the exact array the watershed carves out of territory, surfaced
    so it can be tuned against directly. Small-atom merging only relabels pixels,
    so it does not move the ridge.
    """
    residual_contour = np.asarray(residual_contour, dtype=np.float32)
    territory = np.asarray(territory, dtype=bool)
    ridge = residual_contour > contour_floor
    cores = territory & ~ridge
    markers, _ = ndi.label(cores)
    atoms = watershed(residual_contour, markers=markers, mask=territory)
    if atom_min_area > 0:
        atoms = _merge_small_atoms(atoms, atom_min_area)
    return atoms.astype(np.int32), ridge.astype(np.uint8)


def _merge_small_atoms(atoms: np.ndarray, atom_min_area: int) -> np.ndarray:
    """Fold each atom below ``atom_min_area`` into the neighbour it shares the
    longest border with, repeating until no undersized atom touches another label.

    Smallest-first: the most undersized atom that has a neighbour is merged into
    its longest-shared-border neighbour (ties broken by larger neighbour area,
    then smaller id), its area and adjacency folded into the survivor, and the
    process re-run. An atom with no neighbouring label is left in place — so an
    isolated small territory keeps its label rather than being removed. Merging
    only relabels pixels, leaving no holes.
    """
    ids, counts = np.unique(atoms, return_counts=True)
    area = {int(i): int(c) for i, c in zip(ids.tolist(), counts.tolist()) if i != 0}
    if len(area) < 2:
        return atoms

    # Shared 4-connected border length between each adjacent pair of labels.
    border: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for a, b in ((atoms[:-1, :], atoms[1:, :]), (atoms[:, :-1], atoms[:, 1:])):
        m = (a != b) & (a != 0) & (b != 0)
        for u, v in zip(a[m].tolist(), b[m].tolist()):
            border[u][v] += 1
            border[v][u] += 1

    remap: dict[int, int] = {}
    while True:
        candidates = [lbl for lbl, ar in area.items()
                      if ar < atom_min_area and border[lbl]]
        if not candidates:
            break
        small = min(candidates, key=lambda l: (area[l], l))
        target = max(border[small],
                     key=lambda n: (border[small][n], area[n], -n))
        remap[small] = target
        area[target] += area.pop(small)
        for n, length in border.pop(small).items():
            border[n].pop(small, None)
            if n != target:
                border[target][n] += length
                border[n][target] += length
        border[target].pop(small, None)

    if not remap:
        return atoms

    def resolve(label: int) -> int:
        while label in remap:
            label = remap[label]
        return label

    lut = np.arange(int(atoms.max()) + 1, dtype=atoms.dtype)
    for label in remap:
        lut[label] = resolve(label)
    return lut[atoms]


def extract_atoms_stack_with_maps(
    fg: np.ndarray, contour: np.ndarray, params: AtomParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(atoms, territory, residual_foreground, residual_contour, ridge) stacks, each (T, Y, X).

    ``ridge`` is the per-frame ``residual_contour > contour_floor`` wall returned
    by ``extract_atoms_frame``, accumulated as a ``uint8`` stack.
    """
    fg = np.asarray(fg, dtype=np.float32)
    contour = np.asarray(contour, dtype=np.float32)
    atoms_out = np.zeros(fg.shape, dtype=np.int32)
    territory_out = np.zeros(fg.shape, dtype=np.uint8)
    rf_out = np.zeros(fg.shape, dtype=np.float32)
    rc_out = np.zeros(fg.shape, dtype=np.float32)
    ridge_out = np.zeros(fg.shape, dtype=np.uint8)
    for t in range(fg.shape[0]):
        rf = residual(fg[t], params.fg_window, params.fg_strength)
        territory = rf > params.fg_cutoff
        rc = residual(contour[t], params.contour_window, params.contour_strength)
        atoms_out[t], ridge_out[t] = extract_atoms_frame(
            rc, territory, params.contour_floor, params.atom_min_area
        )
        territory_out[t] = territory.astype(np.uint8)
        rf_out[t] = rf
        rc_out[t] = rc
    return atoms_out, territory_out, rf_out, rc_out, ridge_out


def extract_atoms_stack(
    fg: np.ndarray, contour: np.ndarray, params: AtomParams
) -> np.ndarray:
    """Atom label stack for a (T, Y, X) foreground + contour pair.

    Per frame: residual the fg map and threshold it into territory, residual the
    contour map into the watershed elevation, then ``extract_atoms_frame``.
    Returns the atom labels only — the residual maps are internal.
    """
    atoms, *_ = extract_atoms_stack_with_maps(fg, contour, params)
    return atoms


def params_fingerprint(params: AtomParams) -> str:
    """Stable SHA-1 of the params, used to detect stale atoms.tif in DB-Gen."""
    payload = json.dumps(asdict(params), sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()


def write_atoms_tif(path, atoms: np.ndarray, params: AtomParams) -> None:
    """Write the atom label stack with the params + fingerprint embedded in the
    TIFF ImageDescription, so DB-Gen (②) can read what produced it."""
    description = json.dumps(
        {_PARAMS_KEY: asdict(params), _FINGERPRINT_KEY: params_fingerprint(params)}
    )
    tifffile.imwrite(
        str(path), np.asarray(atoms, dtype=np.int32), description=description
    )


def atom_adjacency(atoms: np.ndarray) -> dict[int, set[int]]:
    """Region-adjacency graph of a label image: labels sharing a 4-connected border."""
    adj: dict[int, set[int]] = {}
    pairs: set[tuple[int, int]] = set()
    for a, b in (
        (atoms[:-1, :], atoms[1:, :]),
        (atoms[:, :-1], atoms[:, 1:]),
    ):
        m = (a != b) & (a != 0) & (b != 0)
        for u, v in zip(a[m].tolist(), b[m].tolist()):
            pairs.add((u, v) if u < v else (v, u))
    for lbl in np.unique(atoms):
        if lbl != 0:
            adj[int(lbl)] = set()
    for u, v in pairs:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def atom_adjacency_weighted(
    atoms: np.ndarray, residual_contour: np.ndarray
) -> tuple[dict[int, set[int]], dict[tuple[int, int], float]]:
    """Atom RAG plus the mean ridge strength on each shared border.

    Returns ``(adj, weights)`` where ``adj`` is the same region-adjacency graph
    as :func:`atom_adjacency`, and ``weights`` maps each ``(u, v)`` edge
    (``u < v``) to the mean ``residual_contour`` value on the 4-connected wall
    separating the two atoms — the saliency of that wall. This is the per-edge
    ridge weight the merge tree merges across and the branch admission orders by.

    The wall value of a border pixel-pair ``(p ∈ u, q ∈ v)`` is the mean of the
    residual-contour at ``p`` and ``q`` (the elevation the watershed floods to
    join them); the edge weight is the mean over all such pairs.
    """
    residual_contour = np.asarray(residual_contour, dtype=np.float32)
    adj: dict[int, set[int]] = {int(lbl): set() for lbl in np.unique(atoms) if lbl != 0}
    wsum: dict[tuple[int, int], float] = defaultdict(float)
    wcnt: dict[tuple[int, int], int] = defaultdict(int)
    for la, lb, ca, cb in (
        (atoms[:-1, :], atoms[1:, :], residual_contour[:-1, :], residual_contour[1:, :]),
        (atoms[:, :-1], atoms[:, 1:], residual_contour[:, :-1], residual_contour[:, 1:]),
    ):
        m = (la != lb) & (la != 0) & (lb != 0)
        wall = (ca[m] + cb[m]) * 0.5
        for u, v, w in zip(la[m].tolist(), lb[m].tolist(), wall.tolist()):
            key = (u, v) if u < v else (v, u)
            wsum[key] += w
            wcnt[key] += 1
    weights = {k: wsum[k] / wcnt[k] for k in wsum}
    for u, v in weights:
        adj[u].add(v)
        adj[v].add(u)
    return adj, weights


def build_atom_merge_tree(
    adj: dict[int, set[int]],
    weights: dict[tuple[int, int], float],
    areas: dict[int, int],
    *,
    min_area: int,
    max_area: int,
    min_frontier: float,
) -> list[frozenset[int]]:
    """Backbone watershed-style merge tree over the atom RAG (always built).

    A binary partition tree by altitude: atoms are leaves, and edges (walls) are
    consumed in ascending ridge weight — weakest wall first — each merge creating
    an internal node = the union of the two regions it joins. Per connected
    component the leaves are the atoms and the root is the maximal connected
    merge; every node is a candidate. Because the structure is a tree, any two
    nodes are nested or disjoint, so overlaps among backbone nodes are
    ancestor↔descendant only (linear, not quadratic).

    This is the same shape of hierarchy ultrack builds at the pixel level
    (``hg.watershed_hierarchy_by_*`` + ``min_area`` / ``max_area`` /
    ``min_frontier`` pruning), but with atoms — the agreed stage-① primitive — as
    leaves, implemented here without a higra dependency on the candidate path.

    Pruning: singletons (atoms) are always kept; an internal node is kept only
    when ``min_area ≤ area ≤ max_area`` and the wall joining it to its sibling
    (its parent merge's frontier) is ``≥ min_frontier`` — a sub-threshold wall is
    a non-salient split, collapsed into the parent. Returns the kept candidate
    members as frozensets, singletons first.
    """
    kept: list[frozenset[int]] = []
    seen: set[frozenset[int]] = set()
    for a in sorted(areas):
        fs = frozenset((a,))
        seen.add(fs)
        kept.append(fs)

    parent = {a: a for a in areas}
    comp_fs = {a: frozenset((a,)) for a in areas}
    comp_area = {a: areas[a] for a in areas}

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    parent_frontier: dict[frozenset[int], float] = {}
    internal: list[tuple[frozenset[int], int]] = []
    for (u, v), w in sorted(weights.items(), key=lambda kv: (kv[1], kv[0])):
        ru, rv = find(u), find(v)
        if ru == rv:
            continue
        child_u, child_v = comp_fs[ru], comp_fs[rv]
        merged = child_u | child_v
        marea = comp_area[ru] + comp_area[rv]
        parent_frontier[child_u] = float(w)
        parent_frontier[child_v] = float(w)
        parent[rv] = ru
        comp_fs[ru] = merged
        comp_area[ru] = marea
        internal.append((merged, marea))

    inf = float("inf")
    for fs, marea in internal:
        if fs in seen:
            continue
        if marea < min_area or marea > max_area:
            continue
        if parent_frontier.get(fs, inf) < min_frontier:
            continue
        seen.add(fs)
        kept.append(fs)
    return kept


@dataclass(frozen=True)
class BranchReport:
    """How branch admission used the per-frame overlap budget."""
    admitted: int = 0
    skipped: int = 0
    budget_hit: bool = False


def branch_unions(
    adj: dict[int, set[int]],
    weights: dict[tuple[int, int], float],
    areas: dict[int, int],
    backbone: list[frozenset[int]] | set[frozenset[int]],
    *,
    max_area: int,
    overlap_budget: int,
) -> tuple[list[frozenset[int]], BranchReport]:
    """Alternative connected unions the backbone did not take, admitted
    most-ambiguous-first up to the per-frame overlap budget.

    Best-first growth over the atom RAG enumerates connected unions
    (area ≤ ``max_area``) in ascending *bottleneck wall weight* — the strongest
    wall that must be crossed to form the union, i.e. how near-tied its weakest
    internal split is. A near-tie is a coin-flip the tree may have gotten wrong;
    a strong wall is a confident separation. Singletons and backbone nodes are
    skipped (they are already candidates) but still expanded.

    Budget: admitting a candidate over atoms ``S`` adds ``Σ_{a∈S} k_a`` new
    overlap pairs, where ``k_a`` is how many already-kept candidates contain atom
    ``a`` (initialised from the backbone, so branch↔backbone overlaps are
    charged; the backbone↔backbone floor is free). A running total is kept and
    admission stops the moment the next candidate would cross ``overlap_budget``
    — most-ambiguous-first, no overshoot. ``overlap_budget`` → 0 admits nothing
    (pure backbone); ``overlap_budget`` → large exhausts the lattice.

    Returns ``(admitted_members, report)``.
    """
    backbone_set = set(backbone)
    k = {a: 0 for a in areas}
    for fs in backbone_set:
        for a in fs:
            k[a] += 1

    def edge_w(a: int, b: int) -> float | None:
        return weights.get((a, b) if a < b else (b, a))

    neg = float("-inf")
    counter = itertools.count()
    heap: list[tuple[float, int, frozenset[int], int]] = []
    for a in sorted(areas):
        heapq.heappush(heap, (neg, next(counter), frozenset((a,)), areas[a]))

    finalized: set[frozenset[int]] = set()
    admitted: list[frozenset[int]] = []
    running = 0
    budget_hit = False
    skipped = 0
    while heap:
        key, _, members, area = heapq.heappop(heap)
        if members in finalized:
            continue
        finalized.add(members)

        if len(members) > 1 and members not in backbone_set:
            cost = sum(k[a] for a in members)
            if running + cost > overlap_budget:
                budget_hit = True
                skipped = len(heap) + 1  # remaining pending (lower bound)
                break
            admitted.append(members)
            running += cost
            for a in members:
                k[a] += 1

        nbrs: set[int] = set()
        for a in members:
            nbrs |= adj[a]
        nbrs -= members
        for w in nbrs:
            na = area + areas[w]
            if na > max_area:
                continue
            new_members = members | {w}
            if new_members in finalized:
                continue
            walls = [edge_w(a, w) for a in members if w in adj[a]]
            ew = min(x for x in walls if x is not None)
            new_key = key if key > ew else ew  # max(key, ew); seeds at -inf
            heapq.heappush(heap, (new_key, next(counter), new_members, na))

    return admitted, BranchReport(
        admitted=len(admitted), skipped=skipped, budget_hit=budget_hit
    )


def enum_connected_unions(
    adj: dict[int, set[int]],
    areas: dict[int, int],
    max_atoms: int,
    max_area: int,
) -> list[frozenset[int]]:
    """All connected atom-subsets of size 1..max_atoms with total area ≤ max_area.

    BFS growth with a global ``seen`` set — each subset is reached exactly once,
    deduped by frozenset. This is the full lattice (the ``overlap_budget`` → large
    limit of :func:`build_atom_merge_tree` + :func:`branch_unions`); it explodes
    on dense frames, so it is retained as the equivalence reference for tests, not
    the default candidate-generation path.
    """
    seen: set[frozenset[int]] = set()
    out: list[frozenset[int]] = []
    frontier: list[tuple[frozenset[int], int]] = []
    for v, ar in areas.items():
        fs = frozenset((v,))
        seen.add(fs)
        out.append(fs)
        if ar <= max_area:
            frontier.append((fs, ar))
    while frontier:
        fs, area = frontier.pop()
        if len(fs) >= max_atoms:
            continue
        nbrs: set[int] = set()
        for v in fs:
            nbrs |= adj[v]
        nbrs -= fs
        for w in nbrs:
            na = area + areas[w]
            if na > max_area:
                continue
            nfs = fs | {w}
            if nfs in seen:
                continue
            seen.add(nfs)
            out.append(nfs)
            frontier.append((nfs, na))
    return out


def read_atoms_params(path) -> tuple[dict | None, str | None]:
    """Return ``(params_dict, fingerprint)`` embedded by ``write_atoms_tif``, or
    ``(None, None)`` if the file has no atom metadata."""
    if not Path(path).exists():
        return None, None
    with tifffile.TiffFile(str(path)) as tf:
        description = tf.pages[0].description or ""
    try:
        meta = json.loads(description)
    except (json.JSONDecodeError, TypeError):
        return None, None
    return meta.get(_PARAMS_KEY), meta.get(_FINGERPRINT_KEY)
