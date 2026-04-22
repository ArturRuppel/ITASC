from __future__ import annotations

import numpy as np
import tifffile

from cellflow.cellpose.stages.cellpose_cluster import CellposeClusterStage
from cellflow.core.paths import stage_dir


def test_cellpose_cluster_stage_contract(tmp_path):
    stage = CellposeClusterStage

    assert stage.name == "cellpose_cluster"
    assert stage.display_name == "Cluster Cellpose"

    root = tmp_path
    pos = 0
    raw_dir = stage_dir(root, pos, "raw_import")
    out_dir = stage_dir(root, pos, "cellpose_cluster")
    raw_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    for rel in [
        "nucleus_4d.tif",
        "cell_4d.tif",
        "nucleus_zavg.tif",
        "cell_zavg.tif",
    ]:
        tifffile.imwrite(str(raw_dir / rel), np.zeros((1, 1), dtype=np.uint16), compression="zlib")

    assert not stage.is_complete(root, pos)

    for rel in [
        "nucleus_dp.tif",
        "nucleus_prob.tif",
        "nucleus_dp_zavg.tif",
        "nucleus_prob_zavg.tif",
        "cell_dp.tif",
        "cell_prob.tif",
        "cell_dp_zavg.tif",
        "cell_prob_zavg.tif",
    ]:
        tifffile.imwrite(str(out_dir / rel), np.zeros((1, 1), dtype=np.uint16), compression="zlib")

    assert stage.is_complete(root, pos)


def test_cellpose_cluster_validate_inputs_requires_contract(tmp_path):
    stage = CellposeClusterStage
    root = tmp_path
    pos = 0

    result = stage.validate_inputs(schema=None, root_dir=root, pos=pos)
    assert not result.ok
    assert any("Required file not found" in error for error in result.errors)
