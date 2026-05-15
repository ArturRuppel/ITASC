"""On-disk artifact paths for a nucleus-workflow position directory.

A single source of truth for the file layout under ``<pos_dir>/``:

    0_input/        — cell_zavg.tif, nucleus_zavg.tif, NLS_zavg.tif
    1_cellpose/     — nucleus_prob_3dt.tif, nucleus_dp_3dt.tif, nucleus_prob_zavg.tif
    2_nucleus/      — contours.tif (or legacy contour_maps.tif), contour_sources.tif,
                      foreground_sources.tif, foreground_scores.tif,
                      tracked_labels.tif, ultrack_workdir/data.db
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NucleusArtifactPaths:
    """Resolve nucleus-workflow artifact locations under a position directory."""

    pos_dir: Path

    # 0_input
    @property
    def cell_zavg(self) -> Path:
        return self.pos_dir / "0_input" / "cell_zavg.tif"

    @property
    def nucleus_zavg(self) -> Path:
        return self.pos_dir / "0_input" / "nucleus_zavg.tif"

    @property
    def nls_zavg(self) -> Path:
        return self.pos_dir / "0_input" / "NLS_zavg.tif"

    # 1_cellpose
    @property
    def prob(self) -> Path:
        return self.pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif"

    @property
    def dp(self) -> Path:
        return self.pos_dir / "1_cellpose" / "nucleus_dp_3dt.tif"

    @property
    def nucleus_prob_zavg(self) -> Path:
        return self.pos_dir / "1_cellpose" / "nucleus_prob_zavg.tif"

    # 2_nucleus
    @property
    def nucleus_dir(self) -> Path:
        return self.pos_dir / "2_nucleus"

    @property
    def contours(self) -> Path:
        """Prefer ``contours.tif``; fall back to legacy ``contour_maps.tif`` only if
        it exists and the preferred name does not."""
        preferred = self.nucleus_dir / "contours.tif"
        fallback = self.nucleus_dir / "contour_maps.tif"
        if preferred.exists() or not fallback.exists():
            return preferred
        return fallback

    @property
    def contour_sources(self) -> Path:
        return self.nucleus_dir / "contour_sources.tif"

    @property
    def foreground_sources(self) -> Path:
        return self.nucleus_dir / "foreground_sources.tif"

    @property
    def foreground_scores(self) -> Path:
        return self.nucleus_dir / "foreground_scores.tif"

    @property
    def tracked(self) -> Path:
        return self.nucleus_dir / "tracked_labels.tif"

    @property
    def ultrack_workdir(self) -> Path:
        return self.nucleus_dir / "ultrack_workdir"

    @property
    def ultrack_db(self) -> Path:
        return self.ultrack_workdir / "data.db"
