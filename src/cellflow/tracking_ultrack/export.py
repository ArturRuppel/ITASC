"""Export selected NodeDB nodes to a (T, Z, Y, X) tracked labelmap."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import tifffile

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.ingest import _build_ultrack_config


def export_tracked_labels(
    working_dir: str | Path,
    cfg: TrackingConfig,
    output_path: str | Path,
) -> np.ndarray:
    """Write ``tracked_labels.tif`` and return the (T, [Z,] Y, X) array."""
    wd = Path(working_dir)
    output_path = Path(output_path)
    ultrack_cfg = _build_ultrack_config(cfg, wd)

    # Try the modern to_labels API first (returns dask or numpy)
    try:
        from ultrack.core.export.labels import to_labels  # type: ignore[import]

        labels = to_labels(ultrack_cfg)
        if hasattr(labels, "compute"):
            labels = labels.compute()
        labels = np.asarray(labels, dtype=np.uint32)
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
        labels = np.stack(frames, axis=0).astype(np.uint32)
        tifffile.imwrite(str(output_path), labels, compression="zlib")
        return labels
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
