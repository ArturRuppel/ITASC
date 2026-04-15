"""s05 — Cell boundary expansion (stub — not yet implemented)."""
from __future__ import annotations

from cellflow.core.protocol import StageProgress, ValidationResult


class _CellLabelsStageClass:
    name = "cell_labels"
    display_name = "Cell Labels"

    def __init__(self):
        self.config = None

    def run(self, **kwargs):
        raise NotImplementedError("cell_labels stage is not yet implemented")
        yield  # make it a generator

    def validate_inputs(self, schema, root_dir, pos) -> ValidationResult:
        return ValidationResult(ok=False, errors=["cell_labels stage is not yet implemented"])

    def is_complete(self, root_dir, pos) -> bool:
        return False


CellLabelsStage = _CellLabelsStageClass()
