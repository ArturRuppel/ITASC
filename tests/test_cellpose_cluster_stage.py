from __future__ import annotations

import sys
import types

import numpy as np
import tifffile

from cellflow.cellpose.stages.cellpose_cluster import CellposeClusterStage
from cellflow.core.paths import stage_dir


def test_cellpose_cluster_stage_contract(tmp_path):
    stage = CellposeClusterStage

    assert stage.name == "cellpose_cluster"
    assert stage.display_name == "Cellpose Cluster"

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
        "nucleus_dp_4d.tif",
        "nucleus_prob_4d.tif",
        "cell_dp_4d.tif",
        "cell_prob_4d.tif",
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


def test_stitched_mode_uses_slice_masks_and_stitching(monkeypatch):
    from cellflow.cellpose.config import CellposeContoursConfig
    from cellflow.cellpose.stages.contours import _stitch_volume_masks

    calls: dict[str, object] = {}

    fake_torch = types.SimpleNamespace(device=lambda name: name)
    fake_dynamics = types.ModuleType("cellpose.dynamics")
    fake_utils = types.ModuleType("cellpose.utils")

    def fake_compute_masks(dp, prob, *, cellprob_threshold, do_3D, device, **_kwargs):
        calls.setdefault("compute", []).append(
            {
                "dp_shape": tuple(dp.shape),
                "prob_shape": tuple(prob.shape),
                "threshold": cellprob_threshold,
                "do_3D": do_3D,
                "device": device,
            }
        )
        return (prob > cellprob_threshold).astype(np.uint16)

    def fake_stitch3D(masks, stitch_threshold=0.25):
        calls["stitch_threshold"] = stitch_threshold
        return np.asarray(masks, dtype=np.uint16) + 1

    fake_dynamics.compute_masks = fake_compute_masks
    fake_utils.stitch3D = fake_stitch3D

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "cellpose.dynamics", fake_dynamics)
    monkeypatch.setitem(sys.modules, "cellpose.utils", fake_utils)

    dp = np.zeros((2, 2, 4, 4), dtype=np.float32)
    prob = np.zeros((2, 4, 4), dtype=np.float32)
    prob[:, 1:3, 1:3] = 1.0

    cfg = CellposeContoursConfig(
        cellprob_threshold=0.5,
        do_3D=False,
        stitch_threshold=0.3,
        device="cpu",
    )

    masks = _stitch_volume_masks(dp, prob, cfg)

    assert calls["stitch_threshold"] == 0.3
    assert len(calls["compute"]) == 2
    assert calls["compute"][0]["do_3D"] is False
    assert masks.shape == prob.shape
    assert masks.dtype == np.uint32
