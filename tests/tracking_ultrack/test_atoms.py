# tests/tracking_ultrack/test_atoms.py
from __future__ import annotations

import numpy as np

from cellflow.tracking_ultrack.atoms import (
    AtomParams,
    atom_adjacency,
    enum_connected_unions,
    extract_atoms_frame,
    extract_atoms_stack,
    params_fingerprint,
    read_atoms_params,
    residual,
    write_atoms_tif,
)


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


def test_residual_strength_blends_to_raw_map():
    rng = np.random.default_rng(1)
    frame = rng.random((30, 30)).astype(np.float32)  # non-negative
    # strength=0 subtracts no background → raw map (clip is a no-op when >= 0).
    assert np.allclose(residual(frame, window=11, strength=0.0), frame)
    # strength defaults to 1.0 (full residual).
    assert np.allclose(residual(frame, window=11, strength=1.0),
                       residual(frame, window=11))
    # a partial strength sits between raw and full residual everywhere.
    raw = residual(frame, window=11, strength=0.0)
    full = residual(frame, window=11, strength=1.0)
    half = residual(frame, window=11, strength=0.5)
    assert np.all(half <= raw + 1e-6) and np.all(half >= full - 1e-6)


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
    atoms, _ridge = extract_atoms_frame(rc, territory, contour_floor=0.05, atom_min_area=0)
    # two disconnected islands -> exactly two atoms, background stays 0
    assert atoms[territory].min() >= 1
    assert atoms[~territory].max() == 0
    assert len({int(v) for v in np.unique(atoms) if v != 0}) == 2


def test_extract_atoms_frame_splits_one_island_on_ridge():
    territory = np.zeros((40, 60), dtype=bool)
    territory[10:30, 8:52] = True  # one connected island
    rc = np.zeros((40, 60), dtype=np.float32)
    rc[10:30, 29:31] = 1.0         # a strong ridge down the middle
    atoms, _ridge = extract_atoms_frame(rc, territory, contour_floor=0.1, atom_min_area=0)
    assert len({int(v) for v in np.unique(atoms) if v != 0}) == 2


def test_extract_atoms_frame_merges_small_atoms_and_leaves_no_holes():
    territory = np.zeros((40, 60), dtype=bool)
    territory[10:30, 8:52] = True
    rc = np.zeros((40, 60), dtype=np.float32)
    rc[10:30, 12:14] = 1.0  # ridge that carves off a tiny sliver (cols 8-12)
    atoms, _ridge = extract_atoms_frame(rc, territory, contour_floor=0.1, atom_min_area=200)
    # tiny sliver merged away -> one atom, and every territory pixel is labelled
    assert len({int(v) for v in np.unique(atoms) if v != 0}) == 1
    assert np.all(atoms[territory] > 0)


def test_extract_atoms_frame_returns_ridge_as_the_threshold_wall():
    # ridge is the exact array the watershed carves out of territory: it is
    # residual_contour > contour_floor, as uint8, sharing the frame shape.
    territory = np.zeros((40, 60), dtype=bool)
    territory[10:30, 8:52] = True
    rc = np.zeros((40, 60), dtype=np.float32)
    rc[10:30, 29:31] = 1.0
    floor = 0.1
    atoms, ridge = extract_atoms_frame(rc, territory, contour_floor=floor, atom_min_area=0)
    assert ridge.dtype == np.uint8
    assert ridge.shape == rc.shape
    assert np.array_equal(ridge, (rc > floor).astype(np.uint8))
    # min-area re-flooding must not move the ridge — it is purely the threshold.
    _atoms2, ridge2 = extract_atoms_frame(rc, territory, contour_floor=floor, atom_min_area=5000)
    assert np.array_equal(ridge, ridge2)


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


def test_extract_atoms_frame_keeps_labels_when_all_atoms_below_min_area():
    territory = np.zeros((20, 20), dtype=bool)
    territory[8:12, 8:12] = True  # a single 16-px atom
    rc = np.zeros((20, 20), dtype=np.float32)
    atoms, _ridge = extract_atoms_frame(rc, territory, contour_floor=0.1, atom_min_area=10000)
    # min_area far exceeds the atom; territory must stay labelled, not blanked
    assert np.all(atoms[territory] > 0)
    assert atoms[~territory].max() == 0


def test_extract_atoms_stack_with_maps_dtypes_and_consistency():
    from cellflow.tracking_ultrack.atoms import extract_atoms_stack_with_maps

    rng = np.random.default_rng(42)
    fg = rng.random((3, 40, 40)).astype(np.float32)
    contour = rng.random((3, 40, 40)).astype(np.float32)
    params = AtomParams(fg_window=11, fg_cutoff=0.01, contour_window=11,
                        contour_floor=0.05, atom_min_area=0)
    atoms, territory, residual_foreground, residual_contour, ridge = extract_atoms_stack_with_maps(
        fg, contour, params
    )
    # shapes
    assert atoms.shape == (3, 40, 40)
    assert territory.shape == (3, 40, 40)
    assert residual_foreground.shape == (3, 40, 40)
    assert residual_contour.shape == (3, 40, 40)
    assert ridge.shape == (3, 40, 40)
    # dtypes
    assert atoms.dtype == np.int32
    assert territory.dtype == np.uint8
    assert residual_foreground.dtype == np.float32
    assert residual_contour.dtype == np.float32
    assert ridge.dtype == np.uint8
    # territory is exactly the fg residual thresholded by fg_cutoff
    assert np.array_equal(territory, (residual_foreground > params.fg_cutoff).astype(np.uint8))
    # ridge is exactly the contour residual thresholded by contour_floor
    assert np.array_equal(ridge, (residual_contour > params.contour_floor).astype(np.uint8))
    # atoms matches extract_atoms_stack (which unpacks just the first element)
    assert np.array_equal(atoms, extract_atoms_stack(fg, contour, params))
    # territory and ridge are binary (0 or 1)
    assert set(np.unique(territory)).issubset({0, 1})
    assert set(np.unique(ridge)).issubset({0, 1})


def test_atom_adjacency_two_adjacent_labels():
    atoms = np.zeros((4, 6), dtype=np.int32)
    atoms[:, :3] = 1  # left half
    atoms[:, 3:] = 2  # right half (shares a 4-connected border with label 1)
    adj = atom_adjacency(atoms)
    # only the two foreground labels appear; background (0) is never a key
    assert set(adj) == {1, 2}
    # edges are symmetric
    assert adj[1] == {2}
    assert adj[2] == {1}


def test_atom_adjacency_diagonal_not_adjacent():
    # two labels touching only at a corner -> not adjacent under 4-connectivity
    atoms = np.zeros((4, 4), dtype=np.int32)
    atoms[0:2, 0:2] = 1  # top-left block
    atoms[2:4, 2:4] = 2  # bottom-right block (diagonal touch only)
    adj = atom_adjacency(atoms)
    assert adj[1] == set()
    assert adj[2] == set()


def test_enum_connected_unions_single_atom():
    adj = {1: set()}
    areas = {1: 10}
    unions = enum_connected_unions(adj, areas, max_atoms=3, max_area=1000)
    assert unions == [frozenset({1})]


def test_enum_connected_unions_two_adjacent():
    adj = {1: {2}, 2: {1}}
    areas = {1: 10, 2: 20}
    unions = enum_connected_unions(adj, areas, max_atoms=3, max_area=1000)
    assert set(unions) == {frozenset({1}), frozenset({2}), frozenset({1, 2})}


def test_enum_connected_unions_respects_max_atoms_and_max_area():
    # a chain 1-2-3, all adjacent in sequence
    adj = {1: {2}, 2: {1, 3}, 3: {2}}
    areas = {1: 10, 2: 10, 3: 10}

    # max_atoms=2 forbids the 3-atom union {1,2,3}
    capped_atoms = enum_connected_unions(adj, areas, max_atoms=2, max_area=1000)
    assert all(len(u) <= 2 for u in capped_atoms)
    assert frozenset({1, 2, 3}) not in capped_atoms
    assert frozenset({1, 2}) in capped_atoms
    assert frozenset({2, 3}) in capped_atoms

    # max_area=15 forbids any multi-atom union (each pair sums to 20 > 15)
    capped_area = enum_connected_unions(adj, areas, max_atoms=3, max_area=15)
    assert set(capped_area) == {frozenset({1}), frozenset({2}), frozenset({3})}
