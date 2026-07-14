"""Tests for ``itasc.segmentation.cell_label_icm`` (unary-only API)."""
from __future__ import annotations

import numpy as np
import pytest

from itasc.segmentation.cell_label_icm import (
    _INF,
    CellICMState,
    CellLabelICMParams,
    _argmin_init_from_dict,
    _read_unary_cache,
    _unary_cache_key,
    assemble_cost_field,
    commit_labels,
    initialize_icm,
)


def test_unary_cache_key_depends_on_input_content():
    """Two runs with identical shape and α/γ but different input arrays must not
    collide on the same cache key — otherwise a stale segmentation is returned."""
    shape = (2, 4, 4)
    rng = np.random.default_rng(0)
    contours_a = rng.random((2, 4, 4), dtype=np.float32)
    contours_b = rng.random((2, 4, 4), dtype=np.float32)

    key_a = _unary_cache_key(shape, 1.0, 2.0, contours_a)
    key_b = _unary_cache_key(shape, 1.0, 2.0, contours_b)
    # Same content → stable key; different content → different key.
    assert key_a == _unary_cache_key(shape, 1.0, 2.0, contours_a.copy())
    assert key_a != key_b
    # α/γ still participate.
    assert key_a != _unary_cache_key(shape, 1.5, 2.0, contours_a)
    # None slots (e.g. absent foreground_scores) are handled.
    assert _unary_cache_key(shape, 1.0, 2.0, contours_a, None).startswith("unary_")


def test_read_unary_cache_missing_is_silent_cold_miss(tmp_path, caplog):
    """A genuinely absent cache returns None without logging a warning."""
    with caplog.at_level("WARNING"):
        assert _read_unary_cache(tmp_path, "nope") is None
    assert not caplog.records


def test_read_unary_cache_corrupt_warns_and_recomputes(tmp_path, caplog):
    """A present-but-unreadable cache must warn (distinct from a cold miss) and
    still degrade gracefully to None."""
    from itasc.segmentation.cell_label_icm import _unary_cache_path

    path = _unary_cache_path(tmp_path, "bad")
    path.write_bytes(b"not an hdf5 file")
    with caplog.at_level("WARNING"):
        assert _read_unary_cache(tmp_path, "bad") is None
    assert any("unreadable unary cache" in r.message for r in caplog.records)


# ── Synthetic fixture ──────────────────────────────────────────────────────

@pytest.fixture
def synthetic_3d() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (nuc_tracks, fg_mask, contours) — T=3, Y=32, X=32.

    Two nucleus tracks (1, 2) placed left and right in every frame.
    Foreground is the whole frame; contours are zero, so the geodesic unary
    reduces to Euclidean distance and the cells split at the Voronoi boundary.
    """
    T, Y, X = 3, 32, 32
    nuc = np.zeros((T, Y, X), dtype=np.uint32)
    nuc[:, 14:18, 8:12] = 1
    nuc[:, 14:18, 20:24] = 2
    fg = np.ones((T, Y, X), dtype=bool)
    ct = np.zeros((T, Y, X), dtype=np.float32)
    return nuc, fg, ct


# ── initialize_icm (per-pixel argmin) ───────────────────────────────────────

def test_initialize_returns_state_and_labels(synthetic_3d):
    nuc, fg, ct = synthetic_3d
    state, labels = initialize_icm(nuc, fg, ct, CellLabelICMParams())

    assert isinstance(state, CellICMState)
    assert labels.shape == (3, 32, 32)
    assert labels.dtype == np.uint32
    assert state.shape == (3, 32, 32)
    assert state.n_labels == 2


def test_all_foreground_pixels_labelled(synthetic_3d):
    nuc, fg, ct = synthetic_3d
    _state, labels = initialize_icm(nuc, fg, ct, CellLabelICMParams())
    assert np.all(labels[fg] > 0)


def test_only_track_ids_appear_and_anchors_preserved(synthetic_3d):
    nuc, fg, ct = synthetic_3d
    _state, labels = initialize_icm(nuc, fg, ct, CellLabelICMParams())

    assert set(np.unique(labels[fg]).tolist()) == {1, 2}
    nuc_mask = nuc > 0
    assert np.all(labels[nuc_mask] == nuc[nuc_mask])


def test_state_has_no_pairwise_fields():
    # The pairwise/refine machinery is gone — the state is unary-only.
    for attr in ("h", "v", "dr", "dl", "tw"):
        assert not hasattr(CellICMState, attr)


# ── assemble_cost_field ──────────────────────────────────────────────────────

def test_assemble_cost_field_matches_formula():
    Y, X = 6, 6
    fg = np.zeros((Y, X), dtype=bool)
    fg[1:5, 1:5] = True
    contours = np.linspace(0, 1, Y * X).reshape(Y, X).astype(np.float32)
    fg_scores = np.full((Y, X), 0.25, dtype=np.float32)

    cost = assemble_cost_field(contours, fg, alpha_unary=10.0,
                               fg_scores_t=fg_scores, gamma_unary=2.0)

    assert np.all(np.isinf(cost[~fg]))
    expected = 1.0 + 10.0 * contours[fg] + 2.0 * (1.0 - 0.25)
    np.testing.assert_allclose(cost[fg], expected, rtol=1e-5)


# ── commit_labels ─────────────────────────────────────────────────────────────

def test_argmin_init_leaves_unreached_foreground_as_background():
    """A foreground pixel that no seed's geodesic front reaches must stay
    background (0), not collapse to label_ids[0] via the best_ki=0 default."""
    label_ids = np.array([10, 20], dtype=np.uint32)
    fg_mask = np.ones((1, 1, 3), dtype=bool)
    unary = {
        (0, 10): np.array([[[0.5, _INF, _INF]]], dtype=np.float32)[0],
        (0, 20): np.array([[[_INF, 0.3, _INF]]], dtype=np.float32)[0],
    }
    # pixel 2 is reached by neither label → must be 0, not 10 (label_ids[0]).
    out = _argmin_init_from_dict(unary, fg_mask, label_ids)
    np.testing.assert_array_equal(out[0, 0], np.array([10, 20, 0], dtype=np.uint32))


def test_commit_labels_writes_tiff(tmp_path):
    import tifffile
    labels = np.zeros((2, 8, 8), dtype=np.uint32)
    labels[:, 2:4, 2:4] = 7
    out = tmp_path / "3_cell" / "tracked_labels.tif"
    commit_labels(labels, out)
    assert out.exists()
    loaded = tifffile.imread(str(out))
    np.testing.assert_array_equal(loaded, labels.astype(np.uint16))
    assert loaded.dtype == np.uint16


def test_commit_labels_preserves_ids_above_uint16(tmp_path):
    """Track ids beyond 65535 must not wrap to uint16 (silent cell merge)."""
    import tifffile
    labels = np.zeros((1, 4, 4), dtype=np.uint32)
    labels[0, 1, 1] = 70000  # > np.iinfo(np.uint16).max
    out = tmp_path / "tracked_labels.tif"
    commit_labels(labels, out)
    loaded = tifffile.imread(str(out))
    assert loaded.dtype == np.uint32
    assert int(loaded.max()) == 70000
