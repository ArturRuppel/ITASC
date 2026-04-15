"""PipelineManifest — per-position crash/resume state written after every stage."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field

StageStatus = Literal["pending", "running", "complete", "stale", "failed"]


class StageRecord(BaseModel):
    status: StageStatus = "pending"
    config_hash: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None


class PipelineManifest(BaseModel):
    stages: Dict[str, StageRecord] = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str) -> "PipelineManifest":
        """Load from disk; return an empty manifest when the file is absent."""
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.model_validate_json(p.read_text())

    def save(self, path: Path | str) -> None:
        """Atomic write: temp file → fsync → rename (never corrupts on crash)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump_json(indent=2).encode()
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        try:
            os.write(fd, data)
            os.fsync(fd)
            os.close(fd)
            os.replace(tmp, p)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def mark_running(self, stage: str) -> None:
        self.stages[stage] = StageRecord(status="running")

    def mark_complete(self, stage: str, config_hash: str | None = None) -> None:
        self.stages[stage] = StageRecord(
            status="complete",
            config_hash=config_hash,
            finished_at=datetime.utcnow().isoformat(),
        )

    def mark_failed(self, stage: str, error: str = "") -> None:
        prev = self.stages.get(stage, StageRecord())
        self.stages[stage] = StageRecord(
            status="failed",
            config_hash=prev.config_hash,
            error=error,
        )

    def mark_stale(self, stage: str) -> None:
        prev = self.stages.get(stage, StageRecord())
        self.stages[stage] = StageRecord(
            status="stale",
            config_hash=prev.config_hash,
        )

    def is_complete(self, stage: str) -> bool:
        return self.stages.get(stage, StageRecord()).status == "complete"

    def is_runnable(self, stage: str) -> bool:
        """Return True when the stage should be (re-)run."""
        return self.stages.get(stage, StageRecord()).status not in ("complete",)
