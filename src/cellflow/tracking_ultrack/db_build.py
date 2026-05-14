"""Shared Ultrack database construction pipeline."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.corrections import (
    Correction,
    apply_corrections_to_database,
    corrections_from_validated_tracks,
)
from cellflow.tracking_ultrack.ingest import _build_ultrack_config
from cellflow.tracking_ultrack.linking import run_linking
from cellflow.tracking_ultrack.seed_prior import write_seed_prior_node_probs


@dataclass(frozen=True)
class UltrackDatabaseBuildReport:
    real_nodes: int = 0
    skipped_validated: int = 0
    fake_nodes: int = 0
    overlaps_added: int = 0
    anchor_nodes: int = 0
    anchor_links: int = 0
    scored_nodes: int = 0
    seed_nodes: int = 0
    boosted_edges: int = 0


def _notify(progress_cb: Callable[[str], None] | None, message: str) -> None:
    if progress_cb is not None:
        progress_cb(message)


def _load_ultrack_inputs(
    contour_maps_path: str | Path,
    foreground_masks_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    contours = np.asarray(tifffile.imread(str(contour_maps_path)), dtype=np.float32)
    foreground = np.asarray(tifffile.imread(str(foreground_masks_path)), dtype=np.float32)
    if contours.ndim == 4 and contours.shape[1] == 1:
        contours = contours[:, 0]
    if foreground.ndim == 4 and foreground.shape[1] == 1:
        foreground = foreground[:, 0]
    return contours, foreground


def _run_ultrack_segment(
    foreground: np.ndarray,
    contours: np.ndarray,
    ultrack_cfg,
    cfg: TrackingConfig,
) -> None:
    try:
        from ultrack.core.segmentation.processing import segment as ultrack_segment
    except ImportError as exc:
        raise ImportError(
            "ultrack must be installed (conda env cellflow) to build data.db"
        ) from exc

    ultrack_segment(
        foreground,
        contours,
        ultrack_cfg,
        max_segments_per_time=cfg.max_segments_per_time,
        overwrite=True,
    )


def build_ultrack_database(
    contour_maps_path: str | Path,
    foreground_masks_path: str | Path,
    nucleus_prob_zavg_path: str | Path,
    working_dir: str | Path,
    cfg: TrackingConfig,
    validated_tracks: dict[int, set[int]] | None = None,
    tracked_labels: np.ndarray | None = None,
    corrections: list[Correction] | None = None,
    use_validated: bool = False,
    progress_cb: Callable[[str], None] | None = None,
) -> UltrackDatabaseBuildReport:
    """Build ``data.db`` from canonical Ultrack segmentation and linking.

    The foreground input is used both for candidate segmentation and for the
    image-quality term in node-probability scoring. ``nucleus_prob_zavg_path``
    is retained for API compatibility.

    Corrections annotate canonical candidates in place. Validated frames mark
    nearby candidates ``FAKE``; anchor frames mark nearest candidates ``REAL``.
    """
    if use_validated and corrections is None and (not validated_tracks or tracked_labels is None):
        raise ValueError(
            "Validated-aware DB generation requires validated tracks and tracked labels."
        )

    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    _notify(progress_cb, "Loading contour maps and foreground masks...")
    contours, foreground = _load_ultrack_inputs(contour_maps_path, foreground_masks_path)
    ultrack_cfg = _build_ultrack_config(cfg, working_dir)

    _notify(progress_cb, "Segmenting candidates (ultrack hierarchy)...")
    _run_ultrack_segment(foreground, contours, ultrack_cfg, cfg)

    if corrections is None and use_validated:
        corrections = corrections_from_validated_tracks(
            validated_tracks or {},
            np.asarray(tracked_labels, dtype=np.uint32),
        )

    real_nodes = skipped_validated = overlaps_added = 0
    fake_nodes = anchor_nodes = anchor_links = 0
    if corrections:
        _notify(progress_cb, "Applying correction annotations...")
        correction_report = apply_corrections_to_database(
            working_dir,
            corrections,
            cfg,
            annotate_anchor_links=False,
        )
        fake_nodes = int(correction_report.fake_nodes)
        anchor_nodes = int(correction_report.anchor_nodes)
        _notify(
            progress_cb,
            f"Marked {fake_nodes} FAKE node(s) and {anchor_nodes} anchor node(s).",
        )

    _notify(progress_cb, "Scoring node probabilities...")
    score_report = write_seed_prior_node_probs(working_dir, foreground_masks_path, cfg)
    scored_nodes = int(getattr(score_report, "scored", 0))
    seed_nodes = int(getattr(score_report, "seeds", 0))
    _notify(progress_cb, f"Scored {scored_nodes} node(s) using {seed_nodes} seed node(s).")

    _notify(progress_cb, "Linking candidates...")
    for step, total, label in run_linking(working_dir, cfg):
        _notify(progress_cb, f"[link {step}/{total}] {label}")

    boosted_edges = 0
    if corrections:
        _notify(progress_cb, "Applying correction link annotations...")
        correction_report = apply_corrections_to_database(
            working_dir,
            corrections,
            cfg,
            annotate_anchor_links=True,
        )
        anchor_links = int(correction_report.anchor_links)
        _notify(progress_cb, f"Marked {anchor_links} anchor link(s).")

    return UltrackDatabaseBuildReport(
        real_nodes=real_nodes,
        skipped_validated=skipped_validated,
        fake_nodes=fake_nodes,
        overlaps_added=overlaps_added,
        anchor_nodes=anchor_nodes,
        anchor_links=anchor_links,
        scored_nodes=scored_nodes,
        seed_nodes=seed_nodes,
        boosted_edges=boosted_edges,
    )
