# tests/tracking_ultrack/test_atoms.py
from __future__ import annotations

import numpy as np
import pytest

from itasc.tracking_ultrack.atoms import (
    AtomParams,
    atom_adjacency,
    atom_adjacency_weighted,
    branch_unions,
    build_atom_merge_tree,
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
    assert p.contour_window == 20
    assert p.contour_floor == 0.05
    assert p.atom_min_area == 10


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


def test_merge_small_atoms_keeps_isolated_small_atom_beside_a_large_one():
    from itasc.tracking_ultrack.atoms import _merge_small_atoms

    atoms = np.zeros((20, 40), dtype=np.int32)
    atoms[2:18, 2:18] = 1   # large atom (256 px)
    atoms[8:12, 30:34] = 2  # small isolated atom (16 px), not touching label 1
    merged = _merge_small_atoms(atoms, atom_min_area=100)
    # no neighbour to merge into -> the small atom survives, nothing blanked
    assert set(np.unique(merged).tolist()) == {0, 1, 2}
    assert np.array_equal(merged, atoms)


def test_merge_small_atoms_folds_into_longest_border_neighbour():
    from itasc.tracking_ultrack.atoms import _merge_small_atoms

    atoms = np.zeros((20, 20), dtype=np.int32)
    atoms[:, :8] = 1       # large left neighbour
    atoms[:4, :] = 1       # ...wrapping over the top, too
    atoms[:, 12:] = 2      # large right neighbour
    atoms[4:8, 8:12] = 3   # small sliver (16 px)
    # sliver 3 borders label 1 on its top + left edges (8 px) and label 2 on its
    # right edge (4 px) -> it must fold into label 1, the longest-border neighbour.
    merged = _merge_small_atoms(atoms, atom_min_area=50)
    assert np.all(merged[4:8, 8:12] == 1)
    assert 3 not in set(np.unique(merged).tolist())


def test_extract_atoms_frame_keeps_labels_when_all_atoms_below_min_area():
    territory = np.zeros((20, 20), dtype=bool)
    territory[8:12, 8:12] = True  # a single 16-px atom
    rc = np.zeros((20, 20), dtype=np.float32)
    atoms, _ridge = extract_atoms_frame(rc, territory, contour_floor=0.1, atom_min_area=10000)
    # min_area far exceeds the atom; territory must stay labelled, not blanked
    assert np.all(atoms[territory] > 0)
    assert atoms[~territory].max() == 0


def test_extract_atoms_stack_with_maps_dtypes_and_consistency():
    from itasc.tracking_ultrack.atoms import extract_atoms_stack_with_maps

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


# ── ridge-weighted adjacency ──────────────────────────────────────────────


def test_atom_adjacency_weighted_mean_ridge_on_border():
    # left/right atoms sharing the wall between cols 2 and 3 (4 border pixel-pairs).
    atoms = np.zeros((4, 6), dtype=np.int32)
    atoms[:, :3] = 1
    atoms[:, 3:] = 2
    rc = np.zeros((4, 6), dtype=np.float32)
    rc[:, 2] = 0.2  # left side of the wall
    rc[:, 3] = 0.4  # right side of the wall
    adj, weights = atom_adjacency_weighted(atoms, rc)
    assert adj == {1: {2}, 2: {1}}
    # wall value per pixel-pair = mean(0.2, 0.4) = 0.3, averaged over the border.
    assert weights[(1, 2)] == pytest.approx(0.3)


# ── backbone merge tree ───────────────────────────────────────────────────


def _chain_3():
    # 1-2-3 chain, weak wall {1,2} (0.1), strong wall {2,3} (0.5).
    adj = {1: {2}, 2: {1, 3}, 3: {2}}
    weights = {(1, 2): 0.1, (2, 3): 0.5}
    areas = {1: 10, 2: 10, 3: 10}
    return adj, weights, areas


def test_build_atom_merge_tree_keeps_singletons_and_root():
    adj, weights, areas = _chain_3()
    tree = build_atom_merge_tree(
        adj, weights, areas, min_area=0, max_area=10_000, min_frontier=0.0
    )
    # every atom is a leaf; the maximal connected merge is the root.
    for a in areas:
        assert frozenset({a}) in tree
    assert frozenset({1, 2, 3}) in tree


def test_build_atom_merge_tree_merges_weakest_wall_first():
    adj, weights, areas = _chain_3()
    tree = build_atom_merge_tree(
        adj, weights, areas, min_area=0, max_area=10_000, min_frontier=0.0
    )
    # the tree merges across the weak wall first -> {1,2} is a node, {2,3} is not.
    assert frozenset({1, 2}) in tree
    assert frozenset({2, 3}) not in tree


def test_build_atom_merge_tree_is_nested_ancestor_descendant_only():
    adj, weights, areas = _chain_3()
    tree = build_atom_merge_tree(
        adj, weights, areas, min_area=0, max_area=10_000, min_frontier=0.0
    )
    # any two tree nodes are nested or disjoint (so overlaps are ancestor↔descendant)
    for a in tree:
        for b in tree:
            if a is b:
                continue
            inter = a & b
            assert inter == frozenset() or inter == a or inter == b


def test_build_atom_merge_tree_max_area_drops_root():
    adj, weights, areas = _chain_3()
    # cap below the full merge area (30) but above a pair (20): root dropped,
    # singletons still kept.
    tree = build_atom_merge_tree(
        adj, weights, areas, min_area=0, max_area=25, min_frontier=0.0
    )
    assert frozenset({1, 2, 3}) not in tree
    assert frozenset({1, 2}) in tree
    for a in areas:
        assert frozenset({a}) in tree


# ── branch admission ──────────────────────────────────────────────────────


def _triangle():
    # complete triangle with distinct walls: (1,2)=0.1 weakest, (2,3)=0.2, (1,3)=0.9.
    adj = {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}
    weights = {(1, 2): 0.1, (2, 3): 0.2, (1, 3): 0.9}
    areas = {1: 10, 2: 10, 3: 10}
    return adj, weights, areas


def _backbone(adj, weights, areas, **kw):
    kw.setdefault("min_area", 0)
    kw.setdefault("max_area", 10_000)
    kw.setdefault("min_frontier", 0.0)
    return build_atom_merge_tree(adj, weights, areas, **kw)


def _count_overlap_pairs(candidates):
    atom_to_ids = {}
    for i, fs in enumerate(candidates):
        for a in fs:
            atom_to_ids.setdefault(a, []).append(i)
    pairs = set()
    for ids in atom_to_ids.values():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pairs.add((ids[i], ids[j]))
    return len(pairs)


def test_branch_unions_budget_zero_admits_nothing():
    adj, weights, areas = _triangle()
    backbone = _backbone(adj, weights, areas)
    branches, report = branch_unions(
        adj, weights, areas, backbone, max_area=10_000, overlap_budget=0
    )
    assert branches == []
    assert report.admitted == 0


def test_branch_unions_large_budget_equals_full_lattice():
    adj, weights, areas = _triangle()
    backbone = _backbone(adj, weights, areas)
    branches, _ = branch_unions(
        adj, weights, areas, backbone, max_area=10_000, overlap_budget=10**9
    )
    full = set(enum_connected_unions(adj, areas, max_atoms=99, max_area=10_000))
    assert set(backbone) | set(branches) == full


def test_branch_unions_admits_most_ambiguous_first():
    adj, weights, areas = _triangle()
    backbone = _backbone(adj, weights, areas)
    # {2,3} (wall 0.2) is more ambiguous than {1,3} (wall 0.9): with a budget that
    # fits only one branch, the lower-wall one is admitted.
    full = set(enum_connected_unions(adj, areas, max_atoms=99, max_area=10_000))
    assert {frozenset({2, 3}), frozenset({1, 3})} == full - set(backbone)
    branches, report = branch_unions(
        adj, weights, areas, backbone, max_area=10_000, overlap_budget=5
    )
    assert branches == [frozenset({2, 3})]
    assert report.budget_hit


def test_branch_unions_candidate_and_overlap_counts_are_monotone():
    adj, weights, areas = _triangle()
    backbone = _backbone(adj, weights, areas)
    counts = []
    overlaps = []
    for budget in (0, 5, 10**9):
        branches, _ = branch_unions(
            adj, weights, areas, backbone, max_area=10_000, overlap_budget=budget
        )
        counts.append(len(backbone) + len(branches))
        overlaps.append(_count_overlap_pairs(backbone + branches))
    assert counts == sorted(counts)
    assert overlaps == sorted(overlaps)


def test_branch_unions_overlap_charge_never_exceeds_budget():
    # dense clump (complete K5): the branch-introduced overlaps stay within budget,
    # and the backbone is fully present even when the budget is hit.
    atoms = list(range(1, 6))
    adj = {a: set(atoms) - {a} for a in atoms}
    weights = {}
    w = 0.0
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            w += 0.01
            weights[(atoms[i], atoms[j])] = w
    areas = {a: 10 for a in atoms}
    backbone = _backbone(adj, weights, areas)
    budget = 12
    branches, report = branch_unions(
        adj, weights, areas, backbone, max_area=10_000, overlap_budget=budget
    )
    new_overlaps = _count_overlap_pairs(backbone + branches) - _count_overlap_pairs(backbone)
    assert new_overlaps <= budget
    assert report.budget_hit
    assert set(backbone).issubset(set(backbone + branches))
