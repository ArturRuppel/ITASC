"""Tests for ``cellflow.segmentation.cell_label_icm`` (unary-only API)."""
from __future__ import annotations

import numpy as np
import pytest

from cellflow.segmentation.cell_label_icm import (
    CellICMState,
    CellLabelICMParams,
    assemble_cost_field,
    commit_labels,
    initialize_icm,
)


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

def test_commit_labels_writes_tiff(tmp_path):
    import tifffile
    labels = np.zeros((2, 8, 8), dtype=np.uint32)
    labels[:, 2:4, 2:4] = 7
    out = tmp_path / "3_cell" / "tracked_labels.tif"
    commit_labels(labels, out)
    assert out.exists()
    loaded = tifffile.imread(str(out))
    np.testing.assert_array_equal(loaded, labels.astype(np.uint16))
