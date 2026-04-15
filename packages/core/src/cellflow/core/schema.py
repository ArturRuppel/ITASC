"""PipelineSchema — experiment-level I/O contract written once by the wizard."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class InterfaceSpec(BaseModel):
    """Describes a single data artifact at a stage boundary."""

    path_template: str
    """Path relative to experiment root; supports ``{pos:02d}`` and ``{stem}``."""

    shape: str
    """Axis string, e.g. ``"CZYX"`` or ``"THW"``."""

    dtype: str
    """NumPy dtype string, e.g. ``"float32"``."""

    entry_note: str = ""
    """Human-readable description for external entry points."""


class PipelineMetadata(BaseModel):
    pixel_size_um: Optional[float] = None
    time_interval_s: Optional[float] = None
    conditions: Dict[str, str] = Field(default_factory=dict)


class PipelineSchema(BaseModel):
    schema_version: str = "1.0"
    created: str = Field(default_factory=lambda: date.today().isoformat())
    stages: List[str] = Field(default_factory=list)
    interfaces: Dict[str, InterfaceSpec] = Field(default_factory=dict)
    metadata: PipelineMetadata = Field(default_factory=PipelineMetadata)
    input_dir: Optional[str] = None
    """Absolute path to the raw acquisition data directory (e.g. NDTiff dataset root)."""

    @classmethod
    def load(cls, path: Path | str) -> "PipelineSchema":
        return cls.model_validate_json(Path(path).read_text())

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2))
