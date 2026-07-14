"""The Qt-free editing controller behind the napari curation tool.

It is the single seam the :class:`CurationWidget` talks to: it owns the curation
CSV path, loads it on construction (empty when absent), mutates it through the
pure authoring ops in :mod:`itasc.contact_analysis.curation`, and
**auto-saves** after every change so the table is always the source of truth. It
exposes per-position queries so the widget renders badges/lists without touching
pandas. Keeping it Qt-free keeps the table logic unit-testable headless.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from itasc.contact_analysis.curation import (
    append_exclusion,
    empty_curation,
    read_curation,
    remove_exclusion,
    write_curation,
)


class CurationController:
    """Load → mutate-and-autosave → query over one curation CSV."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        loaded = read_curation(self.path)
        self.curation = empty_curation() if loaded is None else loaded

    # --------------------------------------------------------------- mutations
    def exclude_frame(
        self, *, experiment_id: str, position_id: str, frame: int, reason: str
    ) -> None:
        self.curation = append_exclusion(
            self.curation,
            experiment_id=experiment_id,
            position_id=position_id,
            frame=int(frame),
            reason=reason,
        )
        self._save()

    def exclude_position(
        self, *, experiment_id: str, position_id: str, reason: str
    ) -> None:
        self.curation = append_exclusion(
            self.curation,
            experiment_id=experiment_id,
            position_id=position_id,
            frame=None,
            reason=reason,
        )
        self._save()

    def remove(
        self, *, experiment_id: str, position_id: str, frame: int | None
    ) -> None:
        self.curation = remove_exclusion(
            self.curation,
            experiment_id=experiment_id,
            position_id=position_id,
            frame=None if frame is None else int(frame),
        )
        self._save()

    # ----------------------------------------------------------------- queries
    def exclusions_for(self, *, experiment_id: str, position_id: str) -> pd.DataFrame:
        """The exclusion rows for one position (reset index), for the widget list."""
        cur = self.curation
        key = (cur["experiment_id"].astype(str) == str(experiment_id)) & (
            cur["position_id"].astype(str) == str(position_id)
        )
        return cur.loc[key].reset_index(drop=True)

    def is_position_excluded(self, *, experiment_id: str, position_id: str) -> bool:
        rows = self.exclusions_for(experiment_id=experiment_id, position_id=position_id)
        return bool(rows["frame"].isna().any())

    def is_frame_excluded(
        self, *, experiment_id: str, position_id: str, frame: int
    ) -> bool:
        rows = self.exclusions_for(experiment_id=experiment_id, position_id=position_id)
        if rows.empty:
            return False
        if rows["frame"].isna().any():
            return True
        return bool((pd.to_numeric(rows["frame"], errors="coerce") == float(frame)).any())

    # -------------------------------------------------------------------- save
    def _save(self) -> None:
        write_curation(self.path, self.curation)
