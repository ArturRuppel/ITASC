"""Backend logic for raw data import and preparation."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Optional

import numpy as np
import tifffile
from scipy.interpolate import interp1d
from scipy.ndimage import shift as nd_shift
from scipy.optimize import least_squares
from skimage.transform import downscale_local_mean

from cellflow.core.paths import stage_dir


@dataclass
class DatasetConfig:
    """Raw data source configuration."""
    ndtiff_path: str
    root_dir: str
    positions: list[int]
    xy_downsample: int = 3
    z_downsample: int = 1
    frame_start: int = 0
    frame_end: int = -1
    cell_channel: int = 1
    nls_channel: int = 2
    nucleus_channel: int = 3


# Channel indices (0-based) in the NDTiff dataset
_CH_642 = 3  # CSU642  — nuclear marker
_CH_488 = 1  # CSU488  — membrane marker
_CH_561 = 2  # CSU561  — NLS-mCherry marker


def discover_display_channels(ndtiff_path: str | Path) -> list[str]:
    """Return channel names from Micro-Manager DisplaySettings.json when present."""
    path = Path(ndtiff_path)
    settings_path = path / "DisplaySettings.json" if path.is_dir() else path.parent / "DisplaySettings.json"
    if not settings_path.exists():
        return []
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        channel_settings = data["map"]["ChannelSettings"]["array"]
    except Exception:
        return []

    names: list[str] = []
    for idx, item in enumerate(channel_settings):
        try:
            raw_name = item["Channel"]["scalar"]
        except Exception:
            raw_name = ""
        name = str(raw_name).strip()
        names.append(name or f"Channel {idx}")
    return names


def discover_metadata(ndtiff_path: str) -> dict:
    """Open an NDTiff dataset and return its metadata."""
    from ndtiff import Dataset

    channel_names = discover_display_channels(ndtiff_path)
    ds = Dataset(ndtiff_path)
    axes = ds.axes
    positions = sorted(axes.get("position", axes.get("p", [0])))
    if not channel_names:
        channels = sorted(axes.get("channel", axes.get("c", [])))
        channel_names = [f"Channel {int(ch)}" for ch in channels]

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
        "channel_names": channel_names,
    }


def _raw_dir(root_dir, pos):
    return stage_dir(root_dir, pos, "raw_import")


def _nucleus_4d_path(root_dir, pos):
    return _raw_dir(root_dir, pos) / "nucleus_zavg.tif"


def _nucleus_3dt_path(root_dir, pos):
    return _raw_dir(root_dir, pos) / "nucleus_3dt.tif"


def _cell_4d_path(root_dir, pos):
    return _raw_dir(root_dir, pos) / "cell_zavg.tif"


def _cell_3dt_path(root_dir, pos):
    return _raw_dir(root_dir, pos) / "cell_3dt.tif"


def _nls_4d_path(root_dir, pos):
    return _raw_dir(root_dir, pos) / "NLS_zavg.tif"


def _nls_3dt_path(root_dir, pos):
    return _raw_dir(root_dir, pos) / "NLS_3dt.tif"


def _z_shift_csv_path(root_dir, pos):
    return _raw_dir(root_dir, pos) / "z_shift.csv"


def _read_z_stack(
    ds: Any, position: int, time: int, channel: int, z_indices: list[int]
) -> np.ndarray:
    slices = []
    for z in z_indices:
        img = ds.read_image(position=position, time=time, channel=channel, z=z)
        if img is None:
            img = np.zeros((ds.image_height, ds.image_width), dtype=np.uint16)
        slices.append(img)
    return np.stack(slices, axis=0)


def _xy_avg(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr.astype(np.uint16)
    if arr.ndim == 3:
        downsampled = downscale_local_mean(arr, (1, factor, factor))
    else:
        downsampled = downscale_local_mean(arr, (factor, factor))
    return downsampled.astype(np.uint16)


def _z_avg(volume: np.ndarray, factor: int) -> np.ndarray:
    factor = max(1, int(factor))
    if factor <= 1:
        return volume.astype(np.uint16)
    chunks = [
        volume[start:start + factor].mean(axis=0)
        for start in range(0, volume.shape[0], factor)
    ]
    return np.stack(chunks, axis=0).astype(np.uint16)


def _mean_profile(volume: np.ndarray) -> np.ndarray:
    return volume.astype(np.float32).mean(axis=(1, 2))


def _smooth_profile(profile: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if profile.size < 3 or sigma <= 0:
        return profile.astype(np.float64)
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(profile.astype(np.float64), sigma=sigma, mode="nearest")


def _fill_profile_nans(profile: np.ndarray) -> np.ndarray:
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
    left_edge = center - 0.5 * span
    right_edge = center + 0.5 * span
    left = 1.0 / (1.0 + np.exp(-(z - left_edge) / left_width))
    right = 1.0 / (1.0 + np.exp(-(z - right_edge) / right_width))
    return offset + slope * (z - center) + amplitude * (left - right)


def _fit_double_sigmoid_profile(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = _fill_profile_nans(_smooth_profile(profile))
    z = np.arange(y.size, dtype=np.float64)

    y_min, y_max = float(np.min(y)), float(np.max(y))
    y_range = max(y_max - y_min, 1.0)

    half_level = y_min + 0.5 * y_range
    support = np.flatnonzero(y >= half_level)
    if support.size >= 2:
        center0 = float(0.5 * (support[0] + support[-1]))
        span0 = float(max(2.0, support[-1] - support[0] + 1))
    else:
        weights = np.clip(y - y_min, 0.0, None)
        center0 = float(np.average(z, weights=weights)) if weights.sum() > 0 else float(0.5 * (y.size - 1))
        span0 = float(max(2.0, min(y.size - 1.0, y.size / 3.0)))

    width0 = float(max(0.5, min(span0 / 3.0, y.size / 4.0)))
    p0 = np.array([y_min, y_range, center0, span0, width0, width0, 0.0], dtype=np.float64)

    lower = np.array([y_min - 2.0 * y_range, 0.0, 0.0, 0.5, 0.1, 0.1, -y_range], dtype=np.float64)
    upper = np.array([y_max + 2.0 * y_range, 20.0 * y_range, float(y.size - 1), float(max(1.0, y.size - 1)), float(max(1.0, y.size)), float(max(1.0, y.size)), y_range], dtype=np.float64)

    def residuals(params: np.ndarray) -> np.ndarray:
        return _double_sigmoid_profile(z, *params) - y

    result = least_squares(residuals, p0, bounds=(lower, upper), loss="soft_l1", f_scale=max(1.0, 0.1 * y_range), max_nfev=2000)
    params = result.x if result.success else p0
    return _double_sigmoid_profile(z, *params), params


def _affine_fit_mse(reference: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    mask = np.isfinite(reference) & np.isfinite(target)
    if mask.sum() < 2:
        return float("inf"), 1.0, 0.0
    ref, tgt = reference[mask].astype(np.float64), target[mask].astype(np.float64)
    design = np.column_stack([ref, np.ones_like(ref)])
    coeffs, _, _, _ = np.linalg.lstsq(design, tgt, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    return float(np.mean((tgt - (a * ref + b)) ** 2)), a, b


def _shift_profile(profile: np.ndarray, shift_slices: float) -> np.ndarray:
    z = np.arange(profile.size, dtype=np.float64)
    interpolator = interp1d(z, profile.astype(np.float64), kind="linear", bounds_error=False, fill_value=np.nan, assume_sorted=True)
    return interpolator(z - shift_slices)


def _estimate_z_shift(reference_profile: np.ndarray, target_profile: np.ndarray, max_shift_slices: float) -> tuple[float, float, float, float]:
    ref_fit, ref_params = _fit_double_sigmoid_profile(reference_profile)
    tgt_fit, tgt_params = _fit_double_sigmoid_profile(target_profile)
    shift_slices = float(np.clip(float(tgt_params[2]) - float(ref_params[2]), -max_shift_slices, max_shift_slices))
    shifted_ref = _shift_profile(ref_fit, shift_slices)
    mse, scale, offset = _affine_fit_mse(shifted_ref, tgt_fit)
    return shift_slices, scale, offset, mse


def _shift_volume(volume: np.ndarray, shift_slices: float) -> np.ndarray:
    if abs(shift_slices) < 1e-9:
        return volume
    shifted = nd_shift(volume.astype(np.float32), shift=(shift_slices, 0.0, 0.0), order=1, mode="constant", cval=0.0, prefilter=False)
    return np.clip(np.rint(shifted), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def _read_corrected_volume(
    ds: Any, position: int, time: int, channel: int, z_indices: list[int], xy_factor: int, z_factor: int, z_shift_slices: float
) -> np.ndarray:
    volume = _read_z_stack(ds, position, time, channel, z_indices)
    volume = _xy_avg(volume, xy_factor)
    volume = _shift_volume(volume, -z_shift_slices)
    volume = _z_avg(volume, z_factor)
    return volume


def _read_corrected_z_avg(
    ds: Any, position: int, time: int, channel: int, z_indices: list[int], xy_factor: int, z_factor: int, z_shift_slices: float
) -> np.ndarray:
    volume = _read_corrected_volume(ds, position, time, channel, z_indices, xy_factor, z_factor, z_shift_slices)
    return volume.mean(axis=0).astype(np.uint16)


def _validate_channel(name: str, value: int) -> int:
    channel = int(value)
    if channel < 0:
        raise ValueError(f"{name} channel must be >= 0, got {channel}.")
    return channel


def run(config: DatasetConfig, pos: int, overwrite: bool = False) -> Generator[tuple[int, int, str], None, None]:
    """Export raw NDTiff data for one position as Z-averages."""
    from ndtiff import Dataset

    cell_channel = _validate_channel("cell", config.cell_channel)
    nls_channel = _validate_channel("NLS", config.nls_channel)
    nucleus_channel = _validate_channel("nucleus", config.nucleus_channel)
    ds = Dataset(config.ndtiff_path)
    axes = ds.axes
    available_positions = sorted(axes.get("position", axes.get("p", [0])))
    if pos not in available_positions:
        raise ValueError(f"Position {pos} not found.")

    available_times = sorted(axes.get("time", [0]))
    z_indices = sorted(axes.get("z", [0]))
    if not available_times:
        raise ValueError("No timepoints found.")
    start = max(0, int(config.frame_start))
    end = int(config.frame_end)
    if end < 0:
        end = len(available_times) - 1
    end = min(end, len(available_times) - 1)
    if start > end:
        raise ValueError(f"Invalid frame range: start {start} is after end {end}.")
    all_times = available_times[start:end + 1]
    max_shift_slices = max(1.0, min(8.0, (len(z_indices) - 1) / 2.0))

    out_dir = _raw_dir(config.root_dir, pos)
    out_dir.mkdir(parents=True, exist_ok=True)
    nuc_out, cell_out = _nucleus_4d_path(config.root_dir, pos), _cell_4d_path(config.root_dir, pos)
    nuc_3dt_out, cell_3dt_out = _nucleus_3dt_path(config.root_dir, pos), _cell_3dt_path(config.root_dir, pos)
    nls_out, nls_3dt_out = _nls_4d_path(config.root_dir, pos), _nls_3dt_path(config.root_dir, pos)
    shift_out, run_params_out = _z_shift_csv_path(config.root_dir, pos), out_dir / "run_params.json"

    if not overwrite and all(p.exists() for p in [nuc_out, cell_out, nls_out, nuc_3dt_out, cell_3dt_out, nls_3dt_out, shift_out, run_params_out]):
        yield (len(all_times), len(all_times), "done")
        return

    reference_profile = None
    shift_rows, z_shifts = [], {}
    for i, t in enumerate(all_times):
        profile = _mean_profile(_xy_avg(_read_z_stack(ds, pos, t, cell_channel, z_indices), config.xy_downsample))
        if reference_profile is None:
            z_shift_slices, scale, offset, mse = 0.0, 1.0, 0.0, 0.0
            reference_profile = profile
        else:
            z_shift_slices, scale, offset, mse = _estimate_z_shift(reference_profile, profile, max_shift_slices)
        z_shifts[t] = z_shift_slices
        shift_rows.append({"time": float(t), "z_shift_slices": z_shift_slices, "intensity_scale": scale, "intensity_offset": offset, "fit_mse": mse})
        yield (i + 1, len(all_times), "z-shift")

    with shift_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "z_shift_slices", "intensity_scale", "intensity_offset", "fit_mse"])
        writer.writeheader()
        writer.writerows(shift_rows)

    # Export nucleus volumes
    h = ds.image_height // config.xy_downsample
    w = ds.image_width // config.xy_downsample
    z_factor = max(1, int(config.z_downsample))
    nz = (len(z_indices) + z_factor - 1) // z_factor
    nuc_4d = np.empty((len(all_times), nz, h, w), dtype=np.uint16)
    nuc_stack = np.empty((len(all_times), h, w), dtype=np.uint16)
    for i, t in enumerate(all_times):
        vol = _read_corrected_volume(ds, pos, t, nucleus_channel, z_indices, config.xy_downsample, z_factor, z_shifts[t])
        nuc_4d[i] = vol
        nuc_stack[i] = vol.mean(axis=0).astype(np.uint16)
        yield (i + 1, len(all_times), "nucleus")
    tifffile.imwrite(str(nuc_3dt_out), nuc_4d, compression="zlib", metadata={"axes": "TZYX"})
    tifffile.imwrite(str(nuc_out), nuc_stack, compression="zlib", metadata={"axes": "TYX"})

    # Export cell volumes
    cell_4d = np.empty((len(all_times), nz, h, w), dtype=np.uint16)
    cell_stack = np.empty((len(all_times), h, w), dtype=np.uint16)
    for i, t in enumerate(all_times):
        vol = _read_corrected_volume(ds, pos, t, cell_channel, z_indices, config.xy_downsample, z_factor, z_shifts[t])
        cell_4d[i] = vol
        cell_stack[i] = vol.mean(axis=0).astype(np.uint16)
        yield (i + 1, len(all_times), "cell")
    tifffile.imwrite(str(cell_3dt_out), cell_4d, compression="zlib", metadata={"axes": "TZYX"})
    tifffile.imwrite(str(cell_out), cell_stack, compression="zlib", metadata={"axes": "TYX"})

    # Export NLS-mCherry volumes from CSU561
    nls_4d = np.empty((len(all_times), nz, h, w), dtype=np.uint16)
    nls_stack = np.empty((len(all_times), h, w), dtype=np.uint16)
    for i, t in enumerate(all_times):
        vol = _read_corrected_volume(ds, pos, t, nls_channel, z_indices, config.xy_downsample, z_factor, z_shifts[t])
        nls_4d[i] = vol
        nls_stack[i] = vol.mean(axis=0).astype(np.uint16)
        yield (i + 1, len(all_times), "NLS")
    tifffile.imwrite(str(nls_3dt_out), nls_4d, compression="zlib", metadata={"axes": "TZYX"})
    tifffile.imwrite(str(nls_out), nls_stack, compression="zlib", metadata={"axes": "TYX"})

    # Metadata
    meta = discover_metadata(config.ndtiff_path)
    run_params = {
        "stage": "raw", "pos": pos, "xy_downsample": config.xy_downsample,
        "z_downsample": z_factor,
        "frame_start": start, "frame_end": end,
        "cell_channel": cell_channel,
        "nls_channel": nls_channel,
        "nucleus_channel": nucleus_channel,
        "pixel_size_um": (meta["pixel_size_um"] or 0.1) * config.xy_downsample,
        "time_interval_s": meta["time_interval_s"] or 60.0,
    }
    run_params_out.write_text(json.dumps(run_params, indent=2), encoding="utf-8")
