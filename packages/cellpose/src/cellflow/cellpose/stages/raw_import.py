"""
s00 — Raw data export from NDTiff to per-timepoint TIFFs.

Outputs (per position)
----------------------
  0_input/nucleus/
    nucleus_3d_t<TTT>.tif       (Z, H, W)       uint16  — one per timepoint
    nucleus_zavg.tif            (T, H, W)       uint16  — Z-mean of nucleus channel
  0_input/cell/
    cell_zavg.tif               (T, H, W)       uint16  — Z-mean of membrane channel (488)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import tifffile
from skimage.transform import downscale_local_mean

from cellflow.cellpose.config import DatasetConfig
from cellflow.core.paths import stage_dir
from cellflow.core.protocol import StageProgress, ValidationResult


def raw_dir(root_dir, pos):
    return stage_dir(root_dir, pos, "raw_import")


def nucleus_subdir(root_dir, pos):
    return raw_dir(root_dir, pos) / "nucleus"


def cell_subdir(root_dir, pos):
    return raw_dir(root_dir, pos) / "cell"


def nucleus_3d_path(root_dir, pos, t):
    return nucleus_subdir(root_dir, pos) / f"nucleus_3d_t{t:03d}.tif"


def nucleus_zavg_path(root_dir, pos):
    return nucleus_subdir(root_dir, pos) / "nucleus_zavg.tif"


def cell_zavg_path(root_dir, pos):
    return cell_subdir(root_dir, pos) / "cell_zavg.tif"

# Channel indices (0-based) in the NDTiff dataset
# Dataset ChNames: ['CSUTRANS', 'CSU405 ', 'CSU488', 'CSU561']
_CH_405 = 1  # CSU405  — nuclear marker (NLS-mCherry)
_CH_488 = 2  # CSU488  — membrane marker


# ── Helpers ──────────────────────────────────────────────────────────────────


def _read_z_stack(
    ds: Any, position: int, time: int, channel: int, z_indices: list[int]
) -> np.ndarray:
    """Return a (Z, H, W) uint16 array for one (pos, time, channel)."""
    slices = []
    for z in z_indices:
        img = ds.read_image(position=position, time=time, channel=channel, z=z)
        if img is None:
            img = np.zeros((ds.image_height, ds.image_width), dtype=np.uint16)
        slices.append(img)
    return np.stack(slices, axis=0)


def _xy_avg(arr: np.ndarray, factor: int) -> np.ndarray:
    """Block-average XY by *factor*. Accepts (Z, H, W) or (H, W); returns uint16."""
    if factor <= 1:
        return arr.astype(np.uint16)
    if arr.ndim == 3:
        downsampled = downscale_local_mean(arr, (1, factor, factor))
    else:
        downsampled = downscale_local_mean(arr, (factor, factor))
    return downsampled.astype(np.uint16)


# ── Export: nucleus ──────────────────────────────────────────────────────────


def _export_nucleus(
    ds: Any,
    config: DatasetConfig,
    pos: int,
    time_list: list[int],
    z_indices: list[int],
    overwrite: bool,
) -> Generator[tuple[int, int, str], None, None]:
    """Export 405-channel Z-stacks: one TIFF per timepoint + Z-mean stack."""
    nuc_dir = nucleus_subdir(config.root_dir, pos)
    nuc_dir.mkdir(parents=True, exist_ok=True)
    xy_factor = config.xy_downsample
    total = len(time_list)

    zavg_out = nucleus_zavg_path(config.root_dir, pos)
    need_zavg = overwrite or not zavg_out.exists()
    z_means: list[np.ndarray] = [] if need_zavg else []

    for i, t in enumerate(time_list):
        out_path = nucleus_3d_path(config.root_dir, pos, t)
        skip_3d = out_path.exists() and not overwrite

        if skip_3d and not need_zavg:
            yield (i + 1, total, "nucleus")
            continue

        volume = _read_z_stack(ds, pos, t, _CH_405, z_indices)
        volume = _xy_avg(volume, xy_factor)

        if not skip_3d:
            tifffile.imwrite(
                str(out_path),
                volume,
                compression="zlib",
                metadata={"axes": "ZYX"},
            )

        if need_zavg:
            z_means.append(volume.mean(axis=0).astype(np.uint16))

        yield (i + 1, total, "nucleus")

    if need_zavg and z_means:
        zavg = np.stack(z_means, axis=0)  # (T, H, W) uint16
        tifffile.imwrite(
            str(zavg_out),
            zavg,
            compression="zlib",
            metadata={"axes": "TYX"},
        )


# ── Export: cell ─────────────────────────────────────────────────────────────


def _export_cell(
    ds: Any,
    config: DatasetConfig,
    pos: int,
    time_list: list[int],
    z_indices: list[int],
    overwrite: bool,
) -> Generator[tuple[int, int, str], None, None]:
    """Export 488-channel Z-mean → (T, H, W) stack at cell/cell_zavg.tif."""
    out_path = cell_zavg_path(config.root_dir, pos)
    if out_path.exists() and not overwrite:
        return

    cell_dir = cell_subdir(config.root_dir, pos)
    cell_dir.mkdir(parents=True, exist_ok=True)

    xy_factor = config.xy_downsample
    h_out = math.ceil(ds.image_height / xy_factor)
    w_out = math.ceil(ds.image_width / xy_factor)
    n_t = len(time_list)

    stack = np.zeros((n_t, h_out, w_out), dtype=np.uint16)

    for ti, t in enumerate(time_list):
        volume = _read_z_stack(ds, pos, t, _CH_488, z_indices)
        projected = volume.mean(axis=0).astype(np.uint16)
        stack[ti] = _xy_avg(projected, xy_factor)
        yield (ti + 1, n_t, "cell")

    tifffile.imwrite(
        str(out_path),
        stack,
        compression="zlib",
        metadata={"axes": "TYX"},
    )


# ── Public API ───────────────────────────────────────────────────────────────


def run(
    config: DatasetConfig,
    pos: int,
    overwrite: bool = False,
) -> Generator[tuple[int, int, str], None, None]:
    """
    Export raw NDTiff data for one position.

    Yields (done, total, label) tuples for progress reporting.
    """
    from ndtiff import Dataset  # optional dep — only needed at runtime

    ds = Dataset(config.ndtiff_path)

    axes = ds.axes
    all_times = sorted(axes.get("time", [0]))
    z_indices = sorted(axes.get("z", [0]))

    time_list = config.timepoints if config.timepoints is not None else all_times

    # Extract pixel size and time interval from dataset summary metadata
    pixel_size_um: Optional[float] = None
    time_interval_s: Optional[float] = None
    try:
        summary = getattr(ds, "summary_metadata", None) or {}
        px = summary.get("PixelSizeUm")
        if px is not None and float(px) > 0:
            pixel_size_um = float(px)
        interval_ms = summary.get("Interval_ms")
        if interval_ms is None:
            interval_ms = summary.get("CustomIntervals_ms")
            if isinstance(interval_ms, list) and interval_ms:
                interval_ms = interval_ms[0]
        if interval_ms is not None:
            time_interval_s = float(interval_ms) / 1000.0
    except Exception:
        pass

    yield from _export_nucleus(ds, config, pos, time_list, z_indices, overwrite)
    yield from _export_cell(ds, config, pos, time_list, z_indices, overwrite)

    # Write run_params.json
    out_dir = raw_dir(config.root_dir, pos)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_params = {
        "stage": "raw",
        "pos": pos,
        "xy_downsample": config.xy_downsample,
        "timepoints": time_list,
        "z_indices": z_indices,
        "ndtiff_path": config.ndtiff_path,
    }
    if pixel_size_um is not None:
        run_params["pixel_size_um"] = pixel_size_um
    if time_interval_s is not None:
        run_params["time_interval_s"] = time_interval_s
    (out_dir / "run_params.json").write_text(
        json.dumps(run_params, indent=2), encoding="utf-8"
    )


# ── StageProtocol wrapper ────────────────────────────────────────────────────


class _RawImportStageClass:
    name = "raw_import"
    display_name = "Raw Import"

    def __init__(self):
        self.config = DatasetConfig(ndtiff_path="", root_dir="", positions=[])

    def run(self, config: DatasetConfig, pos: int, overwrite: bool = False):
        from cellflow.core.logging import StageLogger
        from cellflow.core.paths import log_path

        log = StageLogger(log_path(config.root_dir, pos), self.name)
        with log:
            for progress in run(config=config, pos=pos, overwrite=overwrite):
                yield StageProgress(*progress)

    def validate_inputs(self, schema, root_dir, pos) -> ValidationResult:
        # Raw import reads from NDTiff — no local files to validate here.
        return ValidationResult(ok=True, errors=[])

    def is_complete(self, root_dir, pos) -> bool:
        return (
            nucleus_zavg_path(root_dir, pos).exists()
            and cell_zavg_path(root_dir, pos).exists()
        )


RawImportStage = _RawImportStageClass()
