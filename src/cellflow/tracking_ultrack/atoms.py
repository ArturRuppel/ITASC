# src/cellflow/tracking_ultrack/atoms.py
"""Atom extraction: residual-conditioned foreground split by contour ridges.

Stage ① of the atom-based candidate pipeline. Pure, deterministic functions
shared by the interactive preview and the full-stack ``atoms.tif`` writer.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy import ndimage as ndi
from skimage.filters import threshold_local
from skimage.segmentation import watershed


@dataclass(frozen=True)
class AtomParams:
    """The five knobs that fully determine an atom segmentation."""
    fg_window: int = 51
    fg_cutoff: float = 0.002
    contour_window: int = 51
    contour_floor: float = 0.01
    atom_min_area: int = 100


def residual(frame: np.ndarray, window: int) -> np.ndarray:
    """Local-mean-subtracted residual: ``clip(frame - localmean(frame), 0)``.

    Flattens each map's per-nucleus offset so a single global threshold works
    everywhere while staying ~0 in flat background. ``window`` is forced odd.
    """
    window = int(window) | 1
    frame = np.asarray(frame, dtype=np.float32)
    local_mean = threshold_local(frame, block_size=window, method="gaussian")
    return np.clip(frame - local_mean, 0.0, None).astype(np.float32)


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
            atoms = watershed(residual_contour, markers=keep_markers, mask=territory)
    return atoms.astype(np.int32)


def extract_atoms_stack(
    fg: np.ndarray, contour: np.ndarray, params: AtomParams
) -> np.ndarray:
    """Atom label stack for a (T, Y, X) foreground + contour pair.

    Per frame: residual the fg map and threshold it into territory, residual the
    contour map into the watershed elevation, then ``extract_atoms_frame``.
    Returns the atom labels only — the residual contour is internal.
    """
    fg = np.asarray(fg, dtype=np.float32)
    contour = np.asarray(contour, dtype=np.float32)
    out = np.zeros(fg.shape, dtype=np.int32)
    for t in range(fg.shape[0]):
        territory = residual(fg[t], params.fg_window) > params.fg_cutoff
        residual_contour = residual(contour[t], params.contour_window)
        out[t] = extract_atoms_frame(
            residual_contour, territory, params.contour_floor, params.atom_min_area
        )
    return out
