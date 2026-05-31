# src/cellflow/tracking_ultrack/atoms.py
"""Atom extraction: residual-conditioned foreground split by contour ridges.

Stage ① of the atom-based candidate pipeline. Pure, deterministic functions
shared by the interactive preview and the full-stack ``atoms.tif`` writer.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
from dataclasses import asdict, dataclass
from pathlib import Path
from scipy import ndimage as ndi
from skimage.filters import threshold_local
from skimage.segmentation import watershed

import tifffile

_PARAMS_KEY = "cellflow_atom_params"
_FINGERPRINT_KEY = "cellflow_atom_fingerprint"


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


def residual(frame: np.ndarray, window: int, strength: float = 1.0) -> np.ndarray:
    """Local-mean-subtracted residual: ``clip(frame - strength*localmean(frame), 0)``.

    Flattens each map's per-nucleus offset so a single global threshold works
    everywhere while staying ~0 in flat background. ``window`` is forced odd.

    ``strength`` blends between the raw map and the fully-flattened residual:
    ``1.0`` subtracts the whole local background (default), ``0.0`` subtracts
    nothing so the result is the raw (non-negative) map, and values in between
    partially flatten. Lowering it trades uniform-threshold behaviour for
    keeping more of the original signal where the background is already flat.
    """
    window = int(window) | 1
    frame = np.asarray(frame, dtype=np.float32)
    local_mean = threshold_local(frame, block_size=window, method="gaussian")
    return np.clip(frame - strength * local_mean, 0.0, None).astype(np.float32)


def extract_atoms_frame(
    residual_contour: np.ndarray,
    territory: np.ndarray,
    contour_floor: float,
    atom_min_area: int,
) -> np.ndarray:
    """Split ``territory`` into atoms along the cleaned contour ridge.

    ``ridge`` is where the residual contour exceeds ``contour_floor`` (a noise
    cutoff). Cores (ridge-free territory) seed a watershed that floods the
    residual-contour elevation, so a broken faint ridge still meets at the crest.
    Atoms smaller than ``atom_min_area`` are merged into a neighbour by dropping
    their markers and re-flooding, leaving no holes in the territory.
    """
    residual_contour = np.asarray(residual_contour, dtype=np.float32)
    territory = np.asarray(territory, dtype=bool)
    ridge = residual_contour > contour_floor
    cores = territory & ~ridge
    markers, _ = ndi.label(cores)
    atoms = watershed(residual_contour, markers=markers, mask=territory)
    if atom_min_area > 0:
        ids, counts = np.unique(atoms, return_counts=True)
        small = set(ids[(counts < atom_min_area) & (ids != 0)].tolist())
        if small:
            keep_markers = np.where(np.isin(atoms, list(small)), 0, atoms)
            # Only re-flood if some marker survives pruning; otherwise keeping the
            # original labels avoids silently blanking the whole territory.
            if keep_markers[territory].any():
                atoms = watershed(residual_contour, markers=keep_markers, mask=territory)
    return atoms.astype(np.int32)


def extract_atoms_stack_with_maps(
    fg: np.ndarray, contour: np.ndarray, params: AtomParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(atoms, territory, residual_foreground, residual_contour) stacks, each (T, Y, X)."""
    fg = np.asarray(fg, dtype=np.float32)
    contour = np.asarray(contour, dtype=np.float32)
    atoms_out = np.zeros(fg.shape, dtype=np.int32)
    territory_out = np.zeros(fg.shape, dtype=np.uint8)
    rf_out = np.zeros(fg.shape, dtype=np.float32)
    rc_out = np.zeros(fg.shape, dtype=np.float32)
    for t in range(fg.shape[0]):
        rf = residual(fg[t], params.fg_window, params.fg_strength)
        territory = rf > params.fg_cutoff
        rc = residual(contour[t], params.contour_window, params.contour_strength)
        atoms_out[t] = extract_atoms_frame(
            rc, territory, params.contour_floor, params.atom_min_area
        )
        territory_out[t] = territory.astype(np.uint8)
        rf_out[t] = rf
        rc_out[t] = rc
    return atoms_out, territory_out, rf_out, rc_out


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


def enum_connected_unions(
    adj: dict[int, set[int]],
    areas: dict[int, int],
    max_atoms: int,
    max_area: int,
) -> list[frozenset[int]]:
    """All connected atom-subsets of size 1..max_atoms with total area ≤ max_area.

    BFS growth with a global ``seen`` set — each subset is reached exactly once,
    deduped by frozenset. Exhaustive and correct; bounded by max_atoms/max_area
    so it does not explode on large frames.
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
