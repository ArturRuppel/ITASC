# tests/tracking_ultrack/test_atoms.py
from __future__ import annotations

import numpy as np

from cellflow.tracking_ultrack.atoms import residual


def test_residual_is_zero_on_flat_input():
    frame = np.full((40, 40), 0.5, dtype=np.float32)
    out = residual(frame, window=11)
    assert out.shape == frame.shape
    assert out.dtype == np.float32
    assert np.allclose(out, 0.0, atol=1e-5)


def test_residual_is_nonnegative_and_peaks_on_local_bump():
    frame = np.zeros((40, 40), dtype=np.float32)
    frame[18:22, 18:22] = 1.0  # a bright patch on flat background
    out = residual(frame, window=11)
    assert out.min() >= 0.0
    assert out[20, 20] > 0.0


def test_residual_forces_odd_window():
    frame = np.random.default_rng(0).random((30, 30)).astype(np.float32)
    # even window must not raise and must equal the next odd window result
    assert np.allclose(residual(frame, window=10), residual(frame, window=11))


from cellflow.tracking_ultrack.atoms import extract_atoms_frame


def _two_blob_frame():
    # territory = two square nuclei; residual_contour = a ridge between them.
    territory = np.zeros((40, 80), dtype=bool)
    territory[10:30, 8:36] = True   # left nucleus
    territory[10:30, 44:72] = True  # right nucleus
    residual_contour = np.zeros((40, 80), dtype=np.float32)
    residual_contour[10:30, 35:37] = 0.0   # (separated already by background)
    return residual_contour, territory


def test_extract_atoms_frame_labels_each_territory_island():
    rc, territory = _two_blob_frame()
    atoms = extract_atoms_frame(rc, territory, contour_floor=0.05, atom_min_area=0)
    # two disconnected islands -> exactly two atoms, background stays 0
    assert atoms[territory].min() >= 1
    assert atoms[~territory].max() == 0
    assert len({int(v) for v in np.unique(atoms) if v != 0}) == 2


def test_extract_atoms_frame_splits_one_island_on_ridge():
    territory = np.zeros((40, 60), dtype=bool)
    territory[10:30, 8:52] = True  # one connected island
    rc = np.zeros((40, 60), dtype=np.float32)
    rc[10:30, 29:31] = 1.0         # a strong ridge down the middle
    atoms = extract_atoms_frame(rc, territory, contour_floor=0.1, atom_min_area=0)
    assert len({int(v) for v in np.unique(atoms) if v != 0}) == 2


def test_extract_atoms_frame_merges_small_atoms_and_leaves_no_holes():
    territory = np.zeros((40, 60), dtype=bool)
    territory[10:30, 8:52] = True
    rc = np.zeros((40, 60), dtype=np.float32)
    rc[10:30, 12:14] = 1.0  # ridge that carves off a tiny sliver (cols 8-12)
    atoms = extract_atoms_frame(rc, territory, contour_floor=0.1, atom_min_area=200)
    # tiny sliver merged away -> one atom, and every territory pixel is labelled
    assert len({int(v) for v in np.unique(atoms) if v != 0}) == 1
    assert np.all(atoms[territory] > 0)


from cellflow.tracking_ultrack.atoms import AtomParams, extract_atoms_stack


def test_atom_params_defaults_match_spec():
    p = AtomParams()
    assert p.fg_window == 51
    assert p.fg_cutoff == 0.002
    assert p.contour_window == 51
    assert p.contour_floor == 0.01
    assert p.atom_min_area == 100


def test_extract_atoms_stack_shape_and_determinism():
    rng = np.random.default_rng(0)
    fg = rng.random((3, 40, 40)).astype(np.float32)
    contour = rng.random((3, 40, 40)).astype(np.float32)
    params = AtomParams(fg_window=11, fg_cutoff=0.01, contour_window=11,
                        contour_floor=0.05, atom_min_area=0)
    a1 = extract_atoms_stack(fg, contour, params)
    a2 = extract_atoms_stack(fg, contour, params)
    assert a1.shape == (3, 40, 40)
    assert a1.dtype == np.int32
    assert np.array_equal(a1, a2)  # deterministic


from cellflow.tracking_ultrack.atoms import (
    params_fingerprint,
    write_atoms_tif,
    read_atoms_params,
)


def test_params_fingerprint_is_stable_and_param_sensitive():
    a = AtomParams()
    assert params_fingerprint(a) == params_fingerprint(AtomParams())
    assert params_fingerprint(a) != params_fingerprint(AtomParams(fg_cutoff=0.01))


def test_write_then_read_atoms_tif_round_trips_labels_and_params(tmp_path):
    import tifffile

    atoms = np.zeros((2, 16, 16), dtype=np.int32)
    atoms[0, 2:6, 2:6] = 1
    atoms[1, 8:12, 8:12] = 7
    params = AtomParams(fg_window=31, fg_cutoff=0.005)

    path = tmp_path / "atoms.tif"
    write_atoms_tif(path, atoms, params)

    assert np.array_equal(tifffile.imread(path), atoms)
    stored_params, stored_fp = read_atoms_params(path)
    assert stored_params["fg_window"] == 31
    assert stored_params["fg_cutoff"] == 0.005
    assert stored_fp == params_fingerprint(params)


def test_read_atoms_params_returns_none_when_absent(tmp_path):
    import tifffile

    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((4, 4), dtype=np.int32))
    params, fp = read_atoms_params(path)
    assert params is None and fp is None
