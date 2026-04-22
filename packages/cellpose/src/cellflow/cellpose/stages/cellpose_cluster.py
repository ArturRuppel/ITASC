"""s01 — Informational cluster-side Cellpose contract."""

from __future__ import annotations

from pathlib import Path

from cellflow.cellpose.config import CellposeConfig
from cellflow.core.paths import stage_dir
from cellflow.core.protocol import ValidationResult


_INPUT_FILES = [
    "0_input/nucleus_4d.tif",
    "0_input/cell_4d.tif",
    "0_input/nucleus_zavg.tif",
    "0_input/cell_zavg.tif",
]

_OUTPUT_FILES = [
    "1_cellpose/nucleus_dp.tif",
    "1_cellpose/nucleus_prob.tif",
    "1_cellpose/nucleus_dp_zavg.tif",
    "1_cellpose/nucleus_prob_zavg.tif",
    "1_cellpose/cell_dp.tif",
    "1_cellpose/cell_prob.tif",
    "1_cellpose/cell_dp_zavg.tif",
    "1_cellpose/cell_prob_zavg.tif",
]


class _CellposeClusterStageClass:
    name = "cellpose_cluster"
    display_name = "Cluster Cellpose"

    def __init__(self):
        self.config = CellposeConfig()

    def run(self, **kwargs):
        raise NotImplementedError(
            "cellpose_cluster is informational only; run Cellpose externally and "
            "write the expected files into 1_cellpose/."
        )
        yield  # pragma: no cover - keep structural generator compatibility

    def validate_inputs(self, schema, root_dir, pos) -> ValidationResult:
        from cellflow.core.validation import validate_inputs

        raw_dir = stage_dir(root_dir, pos, "raw_import")
        pos_dir = stage_dir(root_dir, pos, "cellpose_cluster")
        required = [raw_dir / rel for rel in _INPUT_FILES]
        required.extend(pos_dir / rel for rel in [
            "nucleus_dp.tif",
            "nucleus_prob.tif",
            "nucleus_dp_zavg.tif",
            "nucleus_prob_zavg.tif",
            "cell_dp.tif",
            "cell_prob.tif",
            "cell_dp_zavg.tif",
            "cell_prob_zavg.tif",
        ])
        return validate_inputs(required)

    def is_complete(self, root_dir, pos) -> bool:
        d = stage_dir(root_dir, pos, "cellpose_cluster")
        return all((d / rel).exists() for rel in [
            path.removeprefix("1_cellpose/")
            for path in _OUTPUT_FILES
        ])


CellposeClusterStage = _CellposeClusterStageClass()
