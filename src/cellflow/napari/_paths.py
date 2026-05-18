"""On-disk artifact paths for CellFlow position directories.

Canonical file layout under ``<pos_dir>/``:

    0_input/              — cell_zavg.tif, nucleus_zavg.tif, NLS_zavg.tif
    1_cellpose/           — nucleus_prob_3dt.tif, nucleus_dp_3dt.tif,
                            cell_prob_3dt.tif, cell_dp_3dt.tif,
                            nucleus_contours.tif, nucleus_foreground.tif,
                            cell_contours.tif, cell_foreground.tif
    2_nucleus/            — contour_sources.tif, foreground_sources.tif,
                            tracked_labels.tif, ultrack_workdir/data.db
    3_cell/               — filtered_dp.tif, foreground_masks.tif,
                            contours.tif, foreground_scores.tif,
                            tracked_labels.tif
    4_contact_analysis/   — contact_analysis.h5
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

    # 1_cellpose — Cellpose outputs
    @property
    def prob(self) -> Path:
        return self.pos_dir / "1_cellpose" / "nucleus_prob_3dt.tif"

    @property
    def cell_prob(self) -> Path:
        return self.pos_dir / "1_cellpose" / "cell_prob_3dt.tif"

    @property
    def dp(self) -> Path:
        return self.pos_dir / "1_cellpose" / "nucleus_dp_3dt.tif"

    @property
    def cell_dp(self) -> Path:
        return self.pos_dir / "1_cellpose" / "cell_dp_3dt.tif"

    # 1_cellpose — Divergence-map outputs (per channel)
    @property
    def nucleus_contours(self) -> Path:
        return self.pos_dir / "1_cellpose" / "nucleus_contours.tif"

    @property
    def nucleus_foreground(self) -> Path:
        return self.pos_dir / "1_cellpose" / "nucleus_foreground.tif"

    @property
    def cell_contours(self) -> Path:
        return self.pos_dir / "1_cellpose" / "cell_contours.tif"

    @property
    def cell_foreground(self) -> Path:
        return self.pos_dir / "1_cellpose" / "cell_foreground.tif"

    # Aliases for the nucleus channel — historical names retained for callers
    # that don't need a channel suffix (e.g. tracking-input writers).
    @property
    def contours(self) -> Path:
        return self.nucleus_contours

    @property
    def foreground(self) -> Path:
        return self.nucleus_foreground

    # 2_nucleus
    @property
    def nucleus_dir(self) -> Path:
        return self.pos_dir / "2_nucleus"

    @property
    def contour_sources(self) -> Path:
        return self.nucleus_dir / "contour_sources.tif"

    @property
    def foreground_sources(self) -> Path:
        return self.nucleus_dir / "foreground_sources.tif"

    @property
    def tracked(self) -> Path:
        return self.nucleus_dir / "tracked_labels.tif"

    @property
    def ultrack_workdir(self) -> Path:
        return self.nucleus_dir / "ultrack_workdir"

    @property
    def ultrack_db(self) -> Path:
        return self.ultrack_workdir / "data.db"


@dataclass(frozen=True)
class CellArtifactPaths:
    """Resolve cell-workflow artifact locations under a position directory."""

    pos_dir: Path

    @property
    def cell_dir(self) -> Path:
        return self.pos_dir / "3_cell"

    @property
    def filtered_dp(self) -> Path:
        return self.cell_dir / "filtered_dp.tif"

    @property
    def foreground_masks(self) -> Path:
        return self.cell_dir / "foreground_masks.tif"

    @property
    def contours(self) -> Path:
        return self.cell_dir / "contours.tif"

    @property
    def foreground_scores(self) -> Path:
        return self.cell_dir / "foreground_scores.tif"

    @property
    def tracked(self) -> Path:
        return self.cell_dir / "tracked_labels.tif"


@dataclass(frozen=True)
class ContactAnalysisPaths:
    """Resolve contact-analysis artifact locations under a position directory."""

    pos_dir: Path

    @property
    def contact_analysis_dir(self) -> Path:
        return self.pos_dir / "4_contact_analysis"

    @property
    def h5(self) -> Path:
        return self.contact_analysis_dir / "contact_analysis.h5"
