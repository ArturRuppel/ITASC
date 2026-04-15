"""s04 — 3D to 2D label projection (stub — not yet implemented)."""
from __future__ import annotations

from cellflow.core.protocol import StageProgress, ValidationResult


class _Project2DStageClass:
    name = "project2d"
    display_name = "3D → 2D Projection"

    def __init__(self):
        self.config = None

    def run(self, **kwargs):
        raise NotImplementedError("project2d stage is not yet implemented")
        yield  # make it a generator

    def validate_inputs(self, schema, root_dir, pos) -> ValidationResult:
        return ValidationResult(ok=False, errors=["project2d stage is not yet implemented"])

    def is_complete(self, root_dir, pos) -> bool:
        return False


Project2DStage = _Project2DStageClass()
