"""StageProtocol and StageProgress — the universal stage contract."""
from __future__ import annotations

from typing import TYPE_CHECKING, Generator, List, NamedTuple, Protocol

from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path

    from cellflow.core.schema import PipelineSchema


class StageProgress(NamedTuple):
    """Yielded by every stage's ``run()`` generator."""

    done: int
    total: int
    message: str


class ValidationResult(NamedTuple):
    ok: bool
    errors: List[str]


class StageProtocol(Protocol):
    """Structural interface that every stage must satisfy.

    Stages are *not* required to inherit from this class — structural
    (duck-typing) compatibility is enough.  The Protocol definition is
    used for static type-checking only.
    """

    #: Must match the entry-point key registered in pyproject.toml.
    name: str
    #: Displayed in the napari tab header.
    display_name: str
    #: Pydantic model holding all configurable parameters for this stage.
    config: BaseModel

    def run(self, **kwargs) -> Generator[StageProgress, None, None]:
        """Execute the stage; yield progress until done."""
        ...

    def validate_inputs(
        self,
        schema: "PipelineSchema",
        root_dir: "Path",
        pos: int,
    ) -> ValidationResult:
        """Return a ValidationResult for the given position."""
        ...

    def is_complete(self, root_dir: "Path", pos: int) -> bool:
        """Return True when the stage outputs already exist for *pos*."""
        ...
