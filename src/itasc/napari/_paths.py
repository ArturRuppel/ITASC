"""On-disk artifact paths for ITASC position directories.

Canonical file layout under ``<pos_dir>/``:

    0_input/              — cell_zavg.tif, nucleus_zavg.tif, NLS_zavg.tif
    1_cellpose/           — nucleus_prob.tif, nucleus_dp.tif,
                            cell_prob.tif, cell_dp.tif,
                            nucleus_contours.tif, nucleus_foreground.tif,
                            cell_contours.tif, cell_foreground.tif
    2_nucleus/            — contour_sources.tif, foreground_sources.tif,
                            tracked_labels.tif, ultrack_workdir/data.db
    3_cell/               — filtered_dp.tif, foreground_masks.tif,
                            contours.tif, foreground_scores.tif,
                            tracked_labels.tif
    nucleus_labels.tif    — FINAL output: committed nucleus tracked labels
    cell_labels.tif       — FINAL output: committed cell tracked labels
    contact_analysis.h5   — FINAL output: contact graph built from the labels

The numbered stage dirs hold re-runnable *working* artifacts; the base-folder
``nucleus_labels.tif`` / ``cell_labels.tif`` are the committed, downstream-stable
*final* outputs (see :func:`itasc.core.commit.promote_labels`), and
``contact_analysis.h5`` — the graph derived from them — sits beside them. Contact
Analysis discovery defaults to these committed names.
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
        return self.pos_dir / "1_cellpose" / "nucleus_prob.tif"

    @property
    def cell_prob(self) -> Path:
        return self.pos_dir / "1_cellpose" / "cell_prob.tif"

    @property
    def dp(self) -> Path:
        return self.pos_dir / "1_cellpose" / "nucleus_dp.tif"

    @property
    def cell_dp(self) -> Path:
        return self.pos_dir / "1_cellpose" / "cell_dp.tif"

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

    @property
    def nucleus_atoms(self) -> Path:
        return self.nucleus_dir / "atoms.tif"

    # 3_cell working output (the cell-workflow tracked labels, pre-commit)
    @property
    def cell_tracked(self) -> Path:
        return self.pos_dir / "3_cell" / "tracked_labels.tif"

    # Final outputs (committed) — base folder, downstream-stable
    @property
    def nucleus_labels(self) -> Path:
        return self.pos_dir / "nucleus_labels.tif"

    @property
    def cell_labels(self) -> Path:
        return self.pos_dir / "cell_labels.tif"


@dataclass(frozen=True)
class NucleusWorkspace:
    """Single base directory for the nucleus tracking/correction piece.

    All artifacts (database, tracked labels, atoms, source maps) and the
    annotation store (validation/correction JSONs) live directly under
    ``nucleus_dir``; the two foreground/contour *inputs* are explicit paths.

    Three layouts are supported via the constructors:

    * :meth:`staged` — the full orchestrator: ``nucleus_dir`` is
      ``<pos>/2_nucleus`` and the inputs come from ``<pos>/1_cellpose``.
    * :meth:`flat` — a standalone working directory that holds
      ``foreground.tif`` + ``contours.tif`` and receives every output.
    * :meth:`files` — the standalone tracking piece: explicit foreground and
      contour file paths (any name, any location) plus an output directory that
      receives every output.
    """

    nucleus_dir: Path
    foreground: Path
    contours: Path
    # Optional cross-stage cell foreground, used purely as a correction-mode
    # background overlay. Only the staged orchestrator supplies it.
    cell_foreground: Path | None = None

    @classmethod
    def staged(cls, pos_dir: Path) -> NucleusWorkspace:
        paths = NucleusArtifactPaths(Path(pos_dir))
        return cls(
            nucleus_dir=paths.nucleus_dir,
            foreground=paths.nucleus_foreground,
            contours=paths.nucleus_contours,
            cell_foreground=paths.cell_foreground,
        )

    @classmethod
    def flat(cls, work_dir: Path) -> NucleusWorkspace:
        work_dir = Path(work_dir)
        return cls(
            nucleus_dir=work_dir,
            foreground=work_dir / "foreground.tif",
            contours=work_dir / "contours.tif",
        )

    @classmethod
    def files(
        cls,
        *,
        foreground: Path | str,
        contours: Path | str,
        output_dir: Path | str | None = None,
    ) -> NucleusWorkspace:
        """Explicit foreground/contour file paths + an output directory.

        The two inputs may have any name and live anywhere; every output is
        written under ``output_dir`` (defaulting to the foreground file's parent
        when omitted).
        """
        foreground = Path(foreground)
        contours = Path(contours)
        nucleus_dir = Path(output_dir) if output_dir else foreground.parent
        return cls(
            nucleus_dir=nucleus_dir,
            foreground=foreground,
            contours=contours,
        )

    # Artifacts (outputs), all directly under nucleus_dir
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

    @property
    def atoms(self) -> Path:
        return self.nucleus_dir / "atoms.tif"


