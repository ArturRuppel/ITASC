"""StageLogger — JSON-lines logger writing to per-position pipeline.log."""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional


class StageLogger:
    """Context manager that appends JSON-lines entries to ``pipeline.log``.

    Each log entry is a JSON object with keys: ``ts``, ``stage``,
    ``level``, ``message``.

    Optionally also writes to a per-stage ``run.log`` when *stage_log*
    is provided.

    Usage::

        log_file = paths.log_path(root, pos)
        with StageLogger(log_file, "cellpose_nucleus") as log:
            log.info("Starting with tile size 224")
            ...
            log.warning("GPU not available, falling back to CPU")
    """

    def __init__(
        self,
        pipeline_log: Path,
        stage_name: str,
        stage_log: Optional[Path] = None,
    ) -> None:
        self._pipeline_log = pipeline_log
        self._stage_name = stage_name
        self._stage_log = stage_log

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, level: str, message: str) -> None:
        entry = json.dumps(
            {
                "ts": datetime.utcnow().isoformat(),
                "stage": self._stage_name,
                "level": level,
                "message": message,
            }
        )
        line = entry + "\n"
        self._pipeline_log.parent.mkdir(parents=True, exist_ok=True)
        with self._pipeline_log.open("a", encoding="utf-8") as fh:
            fh.write(line)
        if self._stage_log is not None:
            self._stage_log.parent.mkdir(parents=True, exist_ok=True)
            with self._stage_log.open("a", encoding="utf-8") as fh:
                fh.write(line)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        return False  # never suppress exceptions


@contextmanager
def stage_logger(
    pipeline_log: Path,
    stage_name: str,
    stage_log: Optional[Path] = None,
) -> Generator[StageLogger, None, None]:
    """Convenience context manager wrapping :class:`StageLogger`."""
    logger = StageLogger(pipeline_log, stage_name, stage_log)
    with logger:
        yield logger
