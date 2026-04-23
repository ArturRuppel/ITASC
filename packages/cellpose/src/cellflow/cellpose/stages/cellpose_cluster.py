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
    "1_cellpose/nucleus_dp_4d.tif",
    "1_cellpose/nucleus_prob_4d.tif",
    "1_cellpose/cell_dp_4d.tif",
    "1_cellpose/cell_prob_4d.tif",
    "1_cellpose/cell_dp_zavg.tif",
    "1_cellpose/cell_prob_zavg.tif",
]


def _exists_any(dir_path: Path, *names: str) -> bool:
    return any((dir_path / name).exists() for name in names)


class _CellposeClusterStageClass:
    name = "cellpose_cluster"
    display_name = "Cellpose Cluster"

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
        base_result = validate_inputs([raw_dir / rel for rel in _INPUT_FILES])
        errors = list(base_result.errors)
        if not _exists_any(pos_dir, "nucleus_dp_4d.tif", "nucleus_dp.tif"):
            errors.append(f"Required file not found: {pos_dir / 'nucleus_dp_4d.tif'}")
        if not _exists_any(pos_dir, "nucleus_prob_4d.tif", "nucleus_prob.tif"):
            errors.append(f"Required file not found: {pos_dir / 'nucleus_prob_4d.tif'}")
        if not _exists_any(pos_dir, "cell_dp_4d.tif", "cell_dp.tif"):
            errors.append(f"Required file not found: {pos_dir / 'cell_dp_4d.tif'}")
        if not _exists_any(pos_dir, "cell_prob_4d.tif", "cell_prob.tif"):
            errors.append(f"Required file not found: {pos_dir / 'cell_prob_4d.tif'}")
        if not (pos_dir / "cell_dp_zavg.tif").exists():
            errors.append(f"Required file not found: {pos_dir / 'cell_dp_zavg.tif'}")
        if not (pos_dir / "cell_prob_zavg.tif").exists():
            errors.append(f"Required file not found: {pos_dir / 'cell_prob_zavg.tif'}")
        return ValidationResult(ok=not errors, errors=errors)

    def is_complete(self, root_dir, pos) -> bool:
        d = stage_dir(root_dir, pos, "cellpose_cluster")
        return (
            _exists_any(d, "nucleus_dp_4d.tif", "nucleus_dp.tif")
            and _exists_any(d, "nucleus_prob_4d.tif", "nucleus_prob.tif")
            and _exists_any(d, "cell_dp_4d.tif", "cell_dp.tif")
            and _exists_any(d, "cell_prob_4d.tif", "cell_prob.tif")
            and (d / "cell_dp_zavg.tif").exists()
            and (d / "cell_prob_zavg.tif").exists()
        )


CellposeClusterStage = _CellposeClusterStageClass()
