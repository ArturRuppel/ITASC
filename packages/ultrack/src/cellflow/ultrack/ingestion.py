"""Shared helpers for direct hypothesis ingestion into Ultrack."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter


def _contours_from_2d_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 2D label frame into foreground and contour maps."""
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D label frame, got shape {labels.shape}")

    fg = (labels > 0).astype(np.float32)

    ct = np.zeros_like(labels, dtype=np.float32)
    for axis in range(labels.ndim):
        sl_a = [slice(None)] * labels.ndim
        sl_b = [slice(None)] * labels.ndim
        sl_a[axis] = slice(None, -1)
        sl_b[axis] = slice(1, None)
        diff = (labels[tuple(sl_a)] != labels[tuple(sl_b)]).astype(np.float32)
        ct[tuple(sl_a)] = np.maximum(ct[tuple(sl_a)], diff)
        ct[tuple(sl_b)] = np.maximum(ct[tuple(sl_b)], diff)

    return fg, ct


def _smooth_contours(contours: np.ndarray, smooth_sigma: float) -> np.ndarray:
    """Smooth a contour map without mixing across the leading axes."""
    if smooth_sigma <= 0:
        return contours.astype(np.float32)

    contours = np.asarray(contours, dtype=np.float32)
    if contours.ndim == 2:
        smoothed = gaussian_filter(contours, sigma=smooth_sigma)
        max_value = float(smoothed.max())
        if max_value > 0:
            smoothed /= max_value
        return smoothed.astype(np.float32)

    sigma = (0.0,) * (contours.ndim - 2) + (smooth_sigma, smooth_sigma)
    smoothed = gaussian_filter(contours, sigma=sigma)
    max_per_frame = smoothed.max(axis=tuple(range(smoothed.ndim - 2, smoothed.ndim)), keepdims=True)
    smoothed = np.divide(smoothed, max_per_frame, out=np.zeros_like(smoothed), where=max_per_frame > 0)
    return smoothed.astype(np.float32)


def _contours_from_labels(labels: np.ndarray, smooth_sigma: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Convert a label stack into foreground and contour maps.

    The maps mirror the current Ultrack contour conventions so callers can
    ingest direct label hypotheses without first materializing stage-local
    foreground/contours files.
    """
    labels = np.asarray(labels)

    if labels.ndim == 2:
        fg, ct = _contours_from_2d_labels(labels)
        return fg.astype(np.float32), _smooth_contours(ct, smooth_sigma)

    if labels.ndim == 3:
        fg_frames: list[np.ndarray] = []
        ct_frames: list[np.ndarray] = []
        for frame in labels:
            fg, ct = _contours_from_2d_labels(frame)
            fg_frames.append(fg)
            ct_frames.append(ct)
        fg_stack = np.stack(fg_frames, axis=0).astype(np.float32)
        ct_stack = _smooth_contours(np.stack(ct_frames, axis=0), smooth_sigma)
        return fg_stack, ct_stack

    if labels.ndim == 4:
        fg_frames = []
        ct_frames = []
        for frame in labels:
            fg_slices: list[np.ndarray] = []
            ct_slices: list[np.ndarray] = []
            for z_slice in frame:
                fg, ct = _contours_from_2d_labels(z_slice)
                fg_slices.append(fg)
                ct_slices.append(ct)
            fg_frames.append(np.mean(fg_slices, axis=0).astype(np.float32))
            ct_frames.append(np.mean(ct_slices, axis=0).astype(np.float32))
        fg_stack = np.stack(fg_frames, axis=0).astype(np.float32)
        ct_stack = _smooth_contours(np.stack(ct_frames, axis=0), smooth_sigma)
        return fg_stack, ct_stack

    raise ValueError(f"Expected a 2D, 3D, or 4D label stack, got shape {labels.shape}")


def labels_batch_to_foreground_contours(
    labelmaps: Sequence[np.ndarray],
    smooth_sigma: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Average foreground and contour maps derived from a batch of labelmaps."""
    fg_maps: list[np.ndarray] = []
    ct_maps: list[np.ndarray] = []
    for labels in labelmaps:
        fg, ct = _contours_from_labels(labels, smooth_sigma=smooth_sigma)
        fg_maps.append(fg)
        ct_maps.append(ct)

    if not fg_maps:
        raise ValueError("labelmaps must contain at least one array")

    return (
        np.mean(fg_maps, axis=0).astype(np.float32),
        np.mean(ct_maps, axis=0).astype(np.float32),
    )


def write_foreground_contours(
    output_dir: str | Path,
    foreground: np.ndarray,
    contours: np.ndarray,
) -> tuple[Path, Path]:
    """Persist Ultrack foreground and contour maps to TIFF files."""
    out = Path(output_dir)
    fg_path = out / "foreground.tif"
    ct_path = out / "contours.tif"
    tifffile.imwrite(str(fg_path), np.asarray(foreground, dtype=np.float32), compression="zlib")
    tifffile.imwrite(str(ct_path), np.asarray(contours, dtype=np.float32), compression="zlib")
    return fg_path, ct_path


def prepare_hypothesis_labelmaps_for_ingestion(
    labelmaps: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Normalize persisted labelmaps to the shapes expected by Ultrack ingestion.

    ``(T, Z, Y, X)`` inputs are preserved so the contour helper can average
    over z while keeping the resulting foreground/contours 2D. ``(Y, X)``
    inputs are promoted to ``(1, Y, X)`` so Ultrack sees a single-timepoint 2D
    stack. ``(T, Y, X)`` inputs are preserved as-is.
    """
    prepared: list[np.ndarray] = []
    for labels in labelmaps:
        labels = np.asarray(labels, dtype=np.uint32)
        if labels.ndim in (3, 4):
            prepared.append(labels)
        elif labels.ndim == 2:
            prepared.append(labels[np.newaxis, ...])
        else:
            raise ValueError(
                f"Expected a 2D, 3D, or 4D labelmap for Ultrack ingestion, got shape {labels.shape}"
            )

    return prepared


def write_hypothesis_labelmaps(
    output_dir: str | Path,
    labelmaps: Sequence[np.ndarray],
    *,
    stage_name: str,
    source: str | None = None,
) -> Path:
    """Persist label hypotheses and write a manifest describing them.

    Parameters
    ----------
    output_dir:
        Stage output directory that will receive ``labelmaps/`` and
        ``hypotheses_manifest.json``.
    labelmaps:
        Sequence of per-hypothesis label arrays. Each entry is written to
        ``labelmaps/labelmap_XXX.tif``.
    stage_name:
        Human-readable stage name recorded in the manifest.
    source:
        Optional description of the upstream hypothesis generator.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    out = Path(output_dir)
    labelmap_dir = out / "labelmaps"
    labelmap_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "version": 1,
        "stage": stage_name,
        "source": source,
        "labelmap_count": len(labelmaps),
        "labelmaps": [],
    }

    entries: list[dict[str, object]] = []
    for index, labels in enumerate(labelmaps):
        labels = np.asarray(labels, dtype=np.uint32)
        rel_path = Path("labelmaps") / f"labelmap_{index:03d}.tif"
        tifffile.imwrite(str(out / rel_path), labels, compression="zlib")
        entries.append(
            {
                "index": index,
                "path": rel_path.as_posix(),
                "shape": list(labels.shape),
                "dtype": str(labels.dtype),
            }
        )

    manifest["labelmaps"] = entries
    manifest_path = out / "hypotheses_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def load_hypothesis_labelmaps(
    input_dir: str | Path,
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Load label hypotheses and their manifest from a stage directory."""
    inp = Path(input_dir)
    manifest_path = inp / "hypotheses_manifest.json"
    labelmaps: list[np.ndarray] = []

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("labelmaps", [])
        for entry in entries:
            rel_path = Path(entry["path"])
            labelmaps.append(tifffile.imread(str(inp / rel_path)).astype(np.uint32))
        return labelmaps, manifest

    labelmap_dir = inp / "labelmaps"
    for path in sorted(labelmap_dir.glob("labelmap_*.tif")):
        labelmaps.append(tifffile.imread(str(path)).astype(np.uint32))
    return labelmaps, {
        "version": 1,
        "stage": None,
        "source": None,
        "labelmap_count": len(labelmaps),
        "labelmaps": [
            {
                "index": i,
                "path": (Path("labelmaps") / f"labelmap_{i:03d}.tif").as_posix(),
                "shape": list(labels.shape),
                "dtype": str(labels.dtype),
            }
            for i, labels in enumerate(labelmaps)
        ],
    }
