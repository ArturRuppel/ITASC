from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from cellflow.segmentation.cellpose_probability_zavg import (
    CellposeProbabilityZavgResult,
    sigmoid_z_average,
    write_cellpose_probability_zavgs_for_root,
)


def test_sigmoid_z_average_transforms_each_z_slice_before_averaging() -> None:
    logits = np.array(
        [
            [
                [[-2.0, 0.0], [1.0, 2.0]],
                [[0.5, -0.5], [3.0, -3.0]],
            ],
        ],
        dtype=np.float32,
    )

    result = sigmoid_z_average(logits)

    expected = (1.0 / (1.0 + np.exp(-logits))).mean(axis=1)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    assert result.dtype == np.float32


def test_write_cellpose_probability_zavgs_for_root_writes_all_complete_positions(
    tmp_path: Path,
) -> None:
    complete = tmp_path / "pos00"
    incomplete = tmp_path / "pos01"
    (complete / "1_cellpose").mkdir(parents=True)
    (incomplete / "1_cellpose").mkdir(parents=True)
    cell_logits = np.array(
        [
            [
                np.full((2, 3), -2.0, dtype=np.float32),
                np.full((2, 3), 2.0, dtype=np.float32),
            ],
            [
                np.full((2, 3), 0.0, dtype=np.float32),
                np.full((2, 3), 1.0, dtype=np.float32),
            ],
        ],
        dtype=np.float32,
    )
    nucleus_logits = np.array(
        [
            [
                np.full((2, 3), -1.0, dtype=np.float32),
                np.full((2, 3), 3.0, dtype=np.float32),
            ],
            [
                np.full((2, 3), -3.0, dtype=np.float32),
                np.full((2, 3), 0.5, dtype=np.float32),
            ],
        ],
        dtype=np.float32,
    )
    tifffile.imwrite(complete / "1_cellpose" / "cell_prob_3dt.tif", cell_logits)
    tifffile.imwrite(
        complete / "1_cellpose" / "nucleus_prob_3dt.tif",
        nucleus_logits,
    )
    tifffile.imwrite(
        incomplete / "1_cellpose" / "nucleus_prob_3dt.tif",
        nucleus_logits,
    )

    results = write_cellpose_probability_zavgs_for_root(tmp_path)

    assert results == [
        CellposeProbabilityZavgResult(
            position_dir=complete,
            wrote_cell=True,
            wrote_nucleus=True,
            skipped=False,
            message="wrote cell_prob_zavg.tif, nucleus_prob_zavg.tif",
        ),
        CellposeProbabilityZavgResult(
            position_dir=incomplete,
            wrote_cell=False,
            wrote_nucleus=True,
            skipped=False,
            message="wrote nucleus_prob_zavg.tif; missing cell_prob_3dt.tif",
        ),
    ]
    np.testing.assert_allclose(
        tifffile.imread(complete / "1_cellpose" / "cell_prob_zavg.tif"),
        (1.0 / (1.0 + np.exp(-cell_logits))).mean(axis=1),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        tifffile.imread(complete / "1_cellpose" / "nucleus_prob_zavg.tif"),
        (1.0 / (1.0 + np.exp(-nucleus_logits))).mean(axis=1),
        rtol=1e-6,
        atol=1e-6,
    )
    assert not (incomplete / "1_cellpose" / "cell_prob_zavg.tif").exists()
    assert (incomplete / "1_cellpose" / "nucleus_prob_zavg.tif").exists()
