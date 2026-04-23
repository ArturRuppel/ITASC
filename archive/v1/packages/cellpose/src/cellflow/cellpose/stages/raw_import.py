"""
s00 — Raw data export from NDTiff with z-shift correction.

Outputs (per position)
----------------------
  0_input/nucleus_4d.tif        (T, Z, H, W)    uint16  — z-corrected nucleus stack
  0_input/cell_4d.tif           (T, Z, H, W)    uint16  — z-corrected 488 stack
  0_input/z_shift.csv           CSV             per-timepoint z-shift estimate
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import tifffile
from scipy.interpolate import interp1d
from scipy.ndimage import shift as nd_shift
from scipy.optimize import least_squares
from skimage.transform import downscale_local_mean

from cellflow.cellpose.config import DatasetConfig
from cellflow.core.paths import stage_dir
from cellflow.core.protocol import StageProgress, ValidationResult


def discover_metadata(ndtiff_path: str) -> dict:
    """Open an NDTiff dataset and return its metadata without exporting anything.

    Returns a dict with keys:
      - ``positions``: sorted list of available position indices
      - ``pixel_size_um``: raw pixel size in µm (before any downsampling), or None
      - ``time_interval_s``: time interval in seconds, or None
    Safe to call from a background thread.
    """
    from ndtiff import Dataset

    ds = Dataset(ndtiff_path)
    axes = ds.axes
    positions = sorted(axes.get("position", axes.get("p", [0])))

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

    # Fall back to per-image metadata if summary didn't have pixel size
    if pixel_size_um is None:
        try:
            coords_list = ds.get_image_coordinates_list()
            if coords_list:
                img_meta = ds.read_metadata(**coords_list[0])
                px = img_meta.get("PixelSizeUm")
                if px is not None and float(px) > 0:
                    pixel_size_um = float(px)
        except Exception:
            pass

    return {
        "positions": positions,
        "pixel_size_um": pixel_size_um,
        "time_interval_s": time_interval_s,
    }


def raw_dir(root_dir, pos):
    return stage_dir(root_dir, pos, "raw_import")


def nucleus_3d_path(root_dir, pos, t):
    return raw_dir(root_dir, pos) / f"nucleus_3d_t{t:03d}.tif"


def nucleus_zavg_path(root_dir, pos):
    return raw_dir(root_dir, pos) / "nucleus_zavg.tif"


def cell_3d_path(root_dir, pos, t):
    return raw_dir(root_dir, pos) / f"cell_3d_t{t:03d}.tif"


def cell_zavg_path(root_dir, pos):
    return raw_dir(root_dir, pos) / "cell_zavg.tif"


def nucleus_4d_path(root_dir, pos):
    return raw_dir(root_dir, pos) / "nucleus_4d.tif"


def cell_4d_path(root_dir, pos):
    return raw_dir(root_dir, pos) / "cell_4d.tif"


def z_shift_csv_path(root_dir, pos):
    return raw_dir(root_dir, pos) / "z_shift.csv"

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


def _mean_profile(volume: np.ndarray) -> np.ndarray:
    """Return the mean intensity profile over x/y for each z-slice."""
    return volume.astype(np.float32).mean(axis=(1, 2))


def _smooth_profile(profile: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Lightly smooth a 1D profile to reduce slice-to-slice noise."""
    if profile.size < 3 or sigma <= 0:
        return profile.astype(np.float64)
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(profile.astype(np.float64), sigma=sigma, mode="nearest")


def _fill_profile_nans(profile: np.ndarray) -> np.ndarray:
    """Fill NaNs in a 1D profile by linear interpolation."""
    y = profile.astype(np.float64)
    mask = np.isfinite(y)
    if mask.all():
        return y
    if not mask.any():
        return np.zeros_like(y, dtype=np.float64)
    x = np.arange(y.size, dtype=np.float64)
    y[~mask] = np.interp(x[~mask], x[mask], y[mask])
    return y


def _double_sigmoid_profile(
    z: np.ndarray,
    offset: float,
    amplitude: float,
    center: float,
    span: float,
    left_width: float,
    right_width: float,
    slope: float,
) -> np.ndarray:
    """Return a smooth double-sigmoid bump centered at ``center``."""
    left_edge = center - 0.5 * span
    right_edge = center + 0.5 * span
    left = 1.0 / (1.0 + np.exp(-(z - left_edge) / left_width))
    right = 1.0 / (1.0 + np.exp(-(z - right_edge) / right_width))
    return offset + slope * (z - center) + amplitude * (left - right)


def _fit_double_sigmoid_profile(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a double-sigmoid curve to a 1D intensity profile.

    Returns the fitted profile and the fitted parameter vector:
    ``(offset, amplitude, center, span, left_width, right_width, slope)``.
    """
    y = _fill_profile_nans(_smooth_profile(profile))
    z = np.arange(y.size, dtype=np.float64)

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_range = max(y_max - y_min, 1.0)

    half_level = y_min + 0.5 * y_range
    support = np.flatnonzero(y >= half_level)
    if support.size >= 2:
        center0 = float(0.5 * (support[0] + support[-1]))
        span0 = float(max(2.0, support[-1] - support[0] + 1))
    else:
        weights = np.clip(y - y_min, 0.0, None)
        if weights.sum() > 0:
            center0 = float(np.average(z, weights=weights))
        else:
            center0 = float(0.5 * (y.size - 1))
        span0 = float(max(2.0, min(y.size - 1.0, y.size / 3.0)))

    width0 = float(max(0.5, min(span0 / 3.0, y.size / 4.0)))
    p0 = np.array(
        [y_min, y_range, center0, span0, width0, width0, 0.0],
        dtype=np.float64,
    )

    lower = np.array(
        [
            y_min - 2.0 * y_range,
            0.0,
            0.0,
            0.5,
            0.1,
            0.1,
            -y_range,
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            y_max + 2.0 * y_range,
            20.0 * y_range,
            float(y.size - 1),
            float(max(1.0, y.size - 1)),
            float(max(1.0, y.size)),
            float(max(1.0, y.size)),
            y_range,
        ],
        dtype=np.float64,
    )

    def residuals(params: np.ndarray) -> np.ndarray:
        return _double_sigmoid_profile(z, *params) - y

    result = least_squares(
        residuals,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=max(1.0, 0.1 * y_range),
        max_nfev=2000,
    )

    params = result.x if result.success else p0
    fitted = _double_sigmoid_profile(z, *params)
    return fitted, params


def _affine_fit_mse(reference: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    """Fit ``target ≈ a * reference + b`` and return ``(mse, a, b)``."""
    mask = np.isfinite(reference) & np.isfinite(target)
    if mask.sum() < 2:
        return float("inf"), 1.0, 0.0

    ref = reference[mask].astype(np.float64)
    tgt = target[mask].astype(np.float64)
    design = np.column_stack([ref, np.ones_like(ref)])
    coeffs, _, _, _ = np.linalg.lstsq(design, tgt, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    resid = tgt - (a * ref + b)
    mse = float(np.mean(resid ** 2))
    return mse, a, b


def _shift_profile(profile: np.ndarray, shift_slices: float) -> np.ndarray:
    """Linearly shift a 1D profile along z by ``shift_slices``."""
    z = np.arange(profile.size, dtype=np.float64)
    interpolator = interp1d(
        z,
        profile.astype(np.float64),
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=True,
    )
    return interpolator(z - shift_slices)


def _estimate_z_shift(
    reference_profile: np.ndarray,
    target_profile: np.ndarray,
    max_shift_slices: float,
) -> tuple[float, float, float, float]:
    """Estimate the z-shift from fitted double-sigmoid profile centers."""
    ref_fit, ref_params = _fit_double_sigmoid_profile(reference_profile)
    tgt_fit, tgt_params = _fit_double_sigmoid_profile(target_profile)

    ref_center = float(ref_params[2])
    tgt_center = float(tgt_params[2])
    shift_slices = float(np.clip(tgt_center - ref_center, -max_shift_slices, max_shift_slices))

    shifted_ref = _shift_profile(ref_fit, shift_slices)
    mse, scale, offset = _affine_fit_mse(shifted_ref, tgt_fit)
    return shift_slices, scale, offset, mse


def _shift_volume(volume: np.ndarray, shift_slices: float) -> np.ndarray:
    """Apply a linear z-shift to a (Z, H, W) volume."""
    if abs(shift_slices) < 1e-9:
        return volume
    shifted = nd_shift(
        volume.astype(np.float32),
        shift=(shift_slices, 0.0, 0.0),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    return np.clip(np.rint(shifted), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def _read_corrected_volume(
    ds: Any,
    position: int,
    time: int,
    channel: int,
    z_indices: list[int],
    xy_factor: int,
    z_shift_slices: float,
) -> np.ndarray:
    volume = _read_z_stack(ds, position, time, channel, z_indices)
    volume = _xy_avg(volume, xy_factor)
    return _shift_volume(volume, -z_shift_slices)


# ── Export: nucleus ──────────────────────────────────────────────────────────


def _export_nucleus(
    ds: Any,
    config: DatasetConfig,
    pos: int,
    time_list: list[int],
    z_indices: list[int],
    z_shifts: dict[int, float],
    overwrite: bool,
) -> Generator[tuple[int, int, str], None, None]:
    """Export the z-corrected 405-channel stack as a single 4D TIFF."""
    xy_factor = config.xy_downsample
    total = len(time_list)
    first = _read_corrected_volume(
        ds, pos, time_list[0], _CH_405, z_indices, xy_factor, z_shifts[time_list[0]]
    )
    stack = np.empty((total,) + first.shape, dtype=np.uint16)
    stack[0] = first
    yield (1, total, "nucleus")

    for i, t in enumerate(time_list[1:], start=1):
        volume = _read_corrected_volume(
            ds, pos, t, _CH_405, z_indices, xy_factor, z_shifts[t]
        )
        stack[i] = volume
        yield (i + 1, total, "nucleus")

    out_path = nucleus_4d_path(config.root_dir, pos)
    if overwrite or not out_path.exists():
        tifffile.imwrite(
            str(out_path),
            stack,
            compression="zlib",
            metadata={"axes": "TZYX"},
        )


# ── Export: cell ─────────────────────────────────────────────────────────────


def _export_cell(
    ds: Any,
    config: DatasetConfig,
    pos: int,
    time_list: list[int],
    z_indices: list[int],
    z_shifts: dict[int, float],
    overwrite: bool,
) -> Generator[tuple[int, int, str], None, None]:
    """Export the z-corrected 488-channel stack as a single 4D TIFF."""
    xy_factor = config.xy_downsample
    n_t = len(time_list)
    first = _read_corrected_volume(
        ds, pos, time_list[0], _CH_488, z_indices, xy_factor, z_shifts[time_list[0]]
    )
    stack = np.empty((n_t,) + first.shape, dtype=np.uint16)
    stack[0] = first
    yield (1, n_t, "cell")

    for i, t in enumerate(time_list[1:], start=1):
        volume = _read_corrected_volume(
            ds, pos, t, _CH_488, z_indices, xy_factor, z_shifts[t]
        )
        stack[i] = volume
        yield (i + 1, n_t, "cell")

    out_path = cell_4d_path(config.root_dir, pos)
    if overwrite or not out_path.exists():
        tifffile.imwrite(
            str(out_path),
            stack,
            compression="zlib",
            metadata={"axes": "TZYX"},
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
    available_positions = sorted(axes.get("position", axes.get("p", [0])))
    if pos not in available_positions:
        raise ValueError(
            f"Position {pos} not found in dataset (available: {available_positions})"
        )

    all_times = sorted(axes.get("time", [0]))
    z_indices = sorted(axes.get("z", [0]))
    time_list = all_times
    if not time_list:
        raise ValueError("No timepoints found in dataset.")
    max_shift_slices = max(1.0, min(8.0, (len(z_indices) - 1) / 2.0))

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

    # Fall back to per-image metadata if summary didn't have pixel size
    if pixel_size_um is None:
        try:
            coords_list = ds.get_image_coordinates_list()
            if coords_list:
                img_meta = ds.read_metadata(**coords_list[0])
                px = img_meta.get("PixelSizeUm")
                if px is not None and float(px) > 0:
                    pixel_size_um = float(px)
        except Exception:
            pass

    out_dir = raw_dir(config.root_dir, pos)
    out_dir.mkdir(parents=True, exist_ok=True)
    nuc_out = nucleus_4d_path(config.root_dir, pos)
    cell_out = cell_4d_path(config.root_dir, pos)
    shift_out = z_shift_csv_path(config.root_dir, pos)
    run_params_out = out_dir / "run_params.json"
    if (
        not overwrite
        and nuc_out.exists()
        and cell_out.exists()
        and shift_out.exists()
        and run_params_out.exists()
    ):
        yield (len(time_list), len(time_list), "done")
        return

    # First pass: estimate the z-shift from the 488 channel.
    reference_profile: Optional[np.ndarray] = None
    shift_rows: list[dict[str, float]] = []
    z_shifts: dict[int, float] = {}
    for i, t in enumerate(time_list):
        volume_488 = _xy_avg(
            _read_z_stack(ds, pos, t, _CH_488, z_indices), config.xy_downsample
        )
        profile = _mean_profile(volume_488)
        if reference_profile is None:
            z_shift_slices = 0.0
            scale = 1.0
            offset = 0.0
            mse = 0.0
            reference_profile = profile
        else:
            z_shift_slices, scale, offset, mse = _estimate_z_shift(
                reference_profile,
                profile,
                max_shift_slices=max_shift_slices,
            )
        z_shifts[t] = z_shift_slices
        shift_rows.append(
            {
                "time": float(t),
                "z_shift_slices": z_shift_slices,
                "intensity_scale": scale,
                "intensity_offset": offset,
                "fit_mse": mse,
            }
        )
        yield (i + 1, len(time_list), "z-shift")

    with shift_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time",
                "z_shift_slices",
                "intensity_scale",
                "intensity_offset",
                "fit_mse",
            ],
        )
        writer.writeheader()
        writer.writerows(shift_rows)

    yield from _export_nucleus(ds, config, pos, time_list, z_indices, z_shifts, overwrite)
    yield from _export_cell(ds, config, pos, time_list, z_indices, z_shifts, overwrite)

    # Write run_params.json
    run_params = {
        "stage": "raw",
        "pos": pos,
        "xy_downsample": config.xy_downsample,
        "timepoints": time_list,
        "z_indices": z_indices,
        "ndtiff_path": config.ndtiff_path,
        "z_shift_csv": str(shift_out),
        "nucleus_output": str(nuc_out),
        "cell_output": str(cell_out),
    }
    if pixel_size_um is not None:
        run_params["pixel_size_um"] = pixel_size_um * config.xy_downsample
    if time_interval_s is not None:
        run_params["time_interval_s"] = time_interval_s
    run_params_out.write_text(
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
            nucleus_4d_path(root_dir, pos).exists()
            and cell_4d_path(root_dir, pos).exists()
            and z_shift_csv_path(root_dir, pos).exists()
        )


RawImportStage = _RawImportStageClass()


# ── CLI entrypoint ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Export raw NDTiff data with z-shift correction.")
    parser.add_argument("--ndtiff-path", required=True, help="Path to NDTiff dataset directory")
    parser.add_argument("--root-dir", required=True, help="Project root directory")
    parser.add_argument("--pos", type=int, action="append", required=True, dest="positions",
                        metavar="N", help="Position index to export (repeatable)")
    parser.add_argument("--xy-downsample", type=int, default=3, metavar="N",
                        help="XY block-average downsample factor (default: 3)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = parser.parse_args()

    config = DatasetConfig(
        ndtiff_path=args.ndtiff_path,
        root_dir=args.root_dir,
        positions=args.positions,
        xy_downsample=args.xy_downsample,
    )

    n_pos = len(args.positions)
    for p_idx, pos in enumerate(args.positions):
        print(f"[{p_idx + 1}/{n_pos}] pos{pos:02d}")
        try:
            for done, total, label in run(config, pos, overwrite=args.overwrite):
                print(f"  [{done}/{total}] {label:<10}", end="\r", flush=True)
            print()
        except Exception as exc:
            print(f"\nError exporting pos{pos:02d}: {exc}", file=sys.stderr)
            sys.exit(1)

    print("Done.")
