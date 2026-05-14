"""Export selected NodeDB nodes to a (T, Z, Y, X) tracked labelmap."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.corrections import (
    Correction,
    apply_post_solve_corrections,
    corrections_from_validated_tracks,
)
from cellflow.tracking_ultrack.solve import database_has_annotations


def _build_export_config(cfg: TrackingConfig, working_dir: Path):
    from cellflow.tracking_ultrack.ingest import _build_ultrack_config

    return _build_ultrack_config(cfg, working_dir)


def _materialize_labels(labels) -> np.ndarray:
    if hasattr(labels, "compute"):
        labels = labels.compute()
    labels = np.asarray(labels, dtype=np.uint32)
    if labels.ndim == 4 and labels.shape[1] == 1:
        labels = labels[:, 0]
    return labels


def export_tracked_labels(
    working_dir: str | Path,
    cfg: TrackingConfig,
    output_path: str | Path,
    *,
    corrections: list[Correction] | None = None,
    validated_tracks: dict[int, set[int]] | None = None,
    tracked_labels: np.ndarray | None = None,
    preserve_validated_ids: bool | None = None,
) -> np.ndarray:
    """Write ``tracked_labels.tif`` and return the (T, [Z,] Y, X) array."""
    wd = Path(working_dir)
    output_path = Path(output_path)
    annotated_db = database_has_annotations(wd)
    if preserve_validated_ids is None:
        preserve_validated_ids = annotated_db
    if corrections is None and validated_tracks and tracked_labels is not None:
        corrections = corrections_from_validated_tracks(
            validated_tracks,
            np.asarray(tracked_labels, dtype=np.uint32),
        )
    if preserve_validated_ids and not corrections and (
        not validated_tracks or tracked_labels is None
    ):
        raise ValueError(
            "Validated-aware export requires validated tracks and tracked labels."
        )

    labels = _export_tracked_labels_raw(wd, cfg, output_path)
    if corrections:
        if tracked_labels is None:
            tracked_labels = np.zeros_like(labels, dtype=np.uint32)
        labels, _report = apply_post_solve_corrections(
            labels,
            corrections,
            np.asarray(tracked_labels, dtype=np.uint32),
            cfg,
        )
        tifffile.imwrite(str(output_path), labels, compression="zlib")
    return labels


def _export_tracked_labels_raw(
    working_dir: str | Path,
    cfg: TrackingConfig,
    output_path: str | Path,
) -> np.ndarray:
    wd = Path(working_dir)
    output_path = Path(output_path)
    ultrack_cfg = _build_export_config(cfg, wd)

    # Prefer public track export: tracks_to_zarr rasterizes each segment with
    # its track_id, while label-export helpers may expose per-frame segment IDs.
    try:
        from ultrack.core.export import to_tracks_layer, tracks_to_zarr  # type: ignore[import]

        tracks_df, _graph = to_tracks_layer(ultrack_cfg)
        labels = _materialize_labels(
            tracks_to_zarr(ultrack_cfg, tracks_df, overwrite=True)
        )
        tifffile.imwrite(str(output_path), labels, compression="zlib")
        return labels
    except Exception:
        pass

    # Try the modern to_labels API next (returns dask or numpy)
    try:
        from ultrack.core.export.labels import to_labels  # type: ignore[import]

        labels = _materialize_labels(to_labels(ultrack_cfg))
        tifffile.imwrite(str(output_path), labels, compression="zlib")
        return labels
    except Exception:
        pass

    # Fallback: CTC export → stack TIFFs
    tmpdir = Path(tempfile.mkdtemp(prefix="ultrack_ctc_"))
    try:
        from ultrack.core.export.ctc import to_ctc  # type: ignore[import]

        to_ctc(tmpdir, ultrack_cfg, overwrite=True)
        mask_files = sorted(tmpdir.rglob("mask*.tif"))
        if not mask_files:
            mask_files = sorted(tmpdir.rglob("man_track*.tif"))
        if not mask_files:
            mask_files = sorted(tmpdir.rglob("*.tif"))
        if not mask_files:
            raise RuntimeError("CTC export produced no mask files.")
        frames = [tifffile.imread(str(f)) for f in mask_files]
        labels = _materialize_labels(np.stack(frames, axis=0))
        tifffile.imwrite(str(output_path), labels, compression="zlib")
        return labels
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
