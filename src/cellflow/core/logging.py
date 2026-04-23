"""StageLogger — JSON-lines logger writing to per-position pipeline.log."""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional


class StageLogger:
    """Context manager that appends JSON-lines entries to pipeline.log."""

    def __init__(
        self,
        pipeline_log: Path,
        stage_name: str,
    ) -> None:
        self._pipeline_log = pipeline_log
        self._stage_name = stage_name

    def _write(self, level: str, message: str) -> None:
        entry = json.dumps(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "stage": self._stage_name,
                "level": level,
                "message": message,
            }
        )
        self._pipeline_log.parent.mkdir(parents=True, exist_ok=True)
        with self._pipeline_log.open("a", encoding="utf-8") as fh:
            fh.write(entry + "\n")

    def info(self, message: str) -> None:
        self._write("INFO", message)

    def warning(self, message: str) -> None:
        self._write("WARNING", message)

    def error(self, message: str) -> None:
        self._write("ERROR", message)

    def __enter__(self) -> "StageLogger":
        self.info("Stage started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is None:
            self.info("Stage completed successfully")
        else:
            self.error(f"Stage failed: {exc_val}")
        return False
