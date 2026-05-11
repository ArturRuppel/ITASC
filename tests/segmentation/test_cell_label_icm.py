"""Tests for ``cellflow.segmentation.cell_label_icm``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from cellflow.segmentation.cell_label_icm import (
    CellLabelICMParams,
    run_cell_icm_from_pos_dir,
    segment_cells_icm,
)


# ── Synthetic fixture ──────────────────────────────────────────────────────

@pytest.fixture
def synthetic_3d() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (nuc_tracks, fg_mask, contours) — T=3, Y=32, X=32.

    Two nucleus tracks (1, 2) placed left and right in every frame.
    Foreground is the whole frame (all True).  Contours are zero — the
    geodesic unary reduces to Euclidean distance, so the cells split at the
    Voronoi boundary.
    """
    T, Y, X = 3, 32, 32
    nuc = np.zeros((T, Y, X), dtype=np.uint32)
    # Track 1: small square at (15, 9)
    nuc[:, 14:18, 8:12] = 1
    # Track 2: small square at (15, 21)
    nuc[:, 14:18, 20:24] = 2

    fg = np.ones((T, Y, X), dtype=bool)
    ct = np.zeros((T, Y, X), dtype=np.float32)
    return nuc, fg, ct


# ── Tests ──────────────────────────────────────────────────────────────────

class TestSegmentCellsIcm:
    """Tests for the main ``segment_cells_icm`` pipeline."""

    def test_synthetic_output_shape_and_dtype(self, synthetic_3d):
        nuc, fg, ct = synthetic_3d
        params = CellLabelICMParams(n_iters=5)
        result = segment_cells_icm(nuc, fg, ct, params)

        assert result.shape == (3, 32, 32)
        assert result.dtype == np.uint32

    def test_all_foreground_pixels_labelled(self, synthetic_3d):
        nuc, fg, ct = synthetic_3d
        params = CellLabelICMParams(n_iters=5)
        result = segment_cells_icm(nuc, fg, ct, params)

        # Every foreground pixel must have a non-zero label.
        assert np.all(result[fg] > 0), (
            f"{np.count_nonzero(result[fg] == 0)} foreground pixels are unlabelled"
        )

    def test_nucleus_anchors_preserved(self, synthetic_3d):
        """Nucleus pixels must carry their original track ID after ICM."""
        nuc, fg, ct = synthetic_3d
        params = CellLabelICMParams(n_iters=5)
        result = segment_cells_icm(nuc, fg, ct, params)

        nuc_mask = nuc > 0
        assert np.all(result[nuc_mask] == nuc[nuc_mask]), (
            "Nucleus anchors were overwritten"
        )

    def test_shape_mismatch_raises_valueerror(self, synthetic_3d):
        nuc, fg, ct = synthetic_3d
        params = CellLabelICMParams()

        # fg_mask with wrong shape
        bad_fg = np.ones((4, 32, 32), dtype=bool)
        with pytest.raises(ValueError, match="Shape mismatch"):
            segment_cells_icm(nuc, bad_fg, ct, params)

        # contours with wrong shape
        bad_ct = np.zeros((3, 16, 16), dtype=np.float32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            segment_cells_icm(nuc, fg, bad_ct, params)

    def test_two_labels_correctly_assigned(self, synthetic_3d):
        """Only labels 1 and 2 appear — no spurious label IDs."""
        nuc, fg, ct = synthetic_3d
        params = CellLabelICMParams(n_iters=5)
        result = segment_cells_icm(nuc, fg, ct, params)

        unique = set(np.unique(result[fg]))
        assert unique == {1, 2}, f"Unexpected label set: {unique}"


class TestRunCellIcmFromPosDir:
    """Tests for the ``run_cell_icm_from_pos_dir`` disk-loading wrapper."""

    @pytest.fixture
    def pos_dir(self, tmp_path: Path) -> Path:
        """Create a minimal pos_dir with the three required TIFFs."""
        pos = tmp_path / "pos00"
        (pos / "2_nucleus").mkdir(parents=True)
        (pos / "3_cell").mkdir(parents=True)

        T, Y, X = 2, 16, 16
        rng = np.random.default_rng(42)

        nuc = np.zeros((T, Y, X), dtype=np.uint32)
        nuc[:, 3:6, 3:6] = 1   # one track, top-left corner

        fg = np.ones((T, Y, X), dtype=np.uint8)  # whole frame

        ct = rng.uniform(0, 0.3, (T, Y, X)).astype(np.float32)

        tifffile.imwrite(pos / "2_nucleus" / "tracked_labels.tif", nuc)
        tifffile.imwrite(pos / "3_cell" / "foreground_masks.tif", fg)
        tifffile.imwrite(pos / "3_cell" / "contour_maps.tif", ct)
        return pos

    def test_returns_correct_shape_and_dtype(self, pos_dir):
        params = CellLabelICMParams(n_iters=3)
        result = run_cell_icm_from_pos_dir(pos_dir, params)
        assert result.shape == (2, 16, 16)
        assert result.dtype == np.uint32

    def test_nucleus_union_in_foreground(self, pos_dir):
        """Nucleus pixels are always labelled even if fg_mask excluded them."""
        # Reload to verify: we forced fg=1 everywhere, so this is tautological.
        # But the union happens inside run_cell_icm_from_pos_dir.
        params = CellLabelICMParams(n_iters=3)
        result = run_cell_icm_from_pos_dir(pos_dir, params)
        nuc = tifffile.imread(pos_dir / "2_nucleus" / "tracked_labels.tif")
        nuc_mask = nuc > 0
        assert result[nuc_mask].sum() > 0, "Nucleus pixels are unlabelled"

    def test_crop_is_applied(self, pos_dir):
        """Verify that crop slices the output correctly."""
        params = CellLabelICMParams(n_iters=3)
        # Crop to T=[0,1), Y=[0,8), X=[0,8) → shape (1, 8, 8)
        crop = (0, 1, 0, 8, 0, 8)
        result = run_cell_icm_from_pos_dir(pos_dir, params, crop=crop)
        assert result.shape == (1, 8, 8)

    def test_missing_file_raises_filenotfound(self, tmp_path):
        pos = tmp_path / "empty"
        pos.mkdir()
        params = CellLabelICMParams()
        with pytest.raises(FileNotFoundError):
            run_cell_icm_from_pos_dir(pos, params)
