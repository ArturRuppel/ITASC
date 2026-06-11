"""Backfill physical calibration into a position's TIFF stacks.

Older CellFlow runs wrote stacks with no physical calibration (``XResolution``
1/1, ``ResolutionUnit`` none). This tool rewrites them carrying the real pixel
size, Z spacing and frame interval so Fiji / napari / the cell-shape pipeline
read true microns and seconds.

Format:
- Most stacks become **OME-TIFF** (``PhysicalSizeX/Y[/Z]`` in µm, ``TimeIncrement``
  in s). OME holds calibration for any dtype, including the ``uint32`` /
  ``int32`` label stacks that the ImageJ-TIFF format cannot represent, and
  encodes the axis order natively.
- ``atoms.tif`` is special: its ImageDescription holds load-bearing
  ``cellflow_atom_params`` JSON that ``tracking_ultrack.atoms.read_atoms_params``
  reads back. OME-XML would clobber it, so that file keeps its description
  verbatim and only gains baseline ``XResolution`` pixel-size tags.

Safety: each file is written to a sibling temp file and atomically swapped in,
so an interrupted run cannot truncate a real result. Default is ``--dry-run``;
pass ``--apply`` to write.

Pixel size / frame interval default to the position's ``0_input/run_params.json``
(``pixel_size_um`` is taken as the *effective* size of the stored pixels — the
import downsample is already folded in). Z spacing is not in run_params and
defaults to ``--z-size`` (1.0 µm).
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import tifffile

#: Files whose ImageDescription carries JSON the pipeline reads back, so the
#: description must survive untouched (no OME-XML overwrite).
_PRESERVE_DESCRIPTION = {"atoms.tif"}


def canonical_axes(series_axes: str) -> str:
    """tifffile labels an unlabeled leading dim ``Q``; in this pipeline every
    such stack is a time series, so ``Q`` is time. Explicit axes pass through."""
    return series_axes.replace("Q", "T")


def _ome_metadata(axes: str, px: float, z_um: float | None, dt_s: float | None) -> dict:
    md: dict = {
        "axes": axes,
        "PhysicalSizeX": px, "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": px, "PhysicalSizeYUnit": "µm",
    }
    if "Z" in axes and z_um:
        md["PhysicalSizeZ"] = z_um
        md["PhysicalSizeZUnit"] = "µm"
    if "T" in axes and dt_s:
        md["TimeIncrement"] = dt_s
        md["TimeIncrementUnit"] = "s"
    return md


def _baseline_resolution(px: float) -> tuple[float, float]:
    """Pixels-per-centimetre for baseline ``XResolution`` (ResolutionUnit=cm)."""
    ppcm = 1e4 / px
    return (ppcm, ppcm)


def plan_file(path: Path) -> dict:
    """Inspect one TIFF and return what the conversion would do (no writes)."""
    with tifffile.TiffFile(str(path)) as tf:
        s = tf.series[0]
        axes_in = s.axes
        dtype = str(s.dtype)
        shape = tuple(s.shape)
    axes_out = canonical_axes(axes_in)
    preserve = path.name in _PRESERVE_DESCRIPTION
    return {
        "path": path, "shape": shape, "dtype": dtype,
        "axes_in": axes_in, "axes_out": axes_out,
        "format": "baseline (description preserved)" if preserve else "OME-TIFF",
        "preserve": preserve,
    }


def convert_file(path: Path, px: float, z_um: float | None, dt_s: float | None) -> dict:
    """Rewrite *path* in place (via atomic temp swap) with calibration."""
    with tifffile.TiffFile(str(path)) as tf:
        s = tf.series[0]
        data = s.asarray()
        axes_out = canonical_axes(s.axes)
        description = tf.pages[0].description or ""

    tmp = path.with_name(path.stem + ".cf_tmp.tif")
    with warnings.catch_warnings():
        # OME content in a plain ``.tif`` name is intentional — the pipeline
        # keys on the ``.tif`` filename; the extension is cosmetic.
        warnings.filterwarnings("ignore", message=".*OME-TIFF.*extension.*")
        if path.name in _PRESERVE_DESCRIPTION:
            tifffile.imwrite(
                str(tmp), data, photometric="minisblack",
                description=description,
                resolution=_baseline_resolution(px), resolutionunit="CENTIMETER",
            )
        else:
            tifffile.imwrite(
                str(tmp), data, ome=True, photometric="minisblack",
                compression="zlib",
                metadata=_ome_metadata(axes_out, px, z_um, dt_s),
            )
    os.replace(str(tmp), str(path))
    return verify_file(path)


def verify_file(path: Path) -> dict:
    """Re-open and read back the calibration that was written."""
    with tifffile.TiffFile(str(path)) as tf:
        out: dict = {"px": None, "z": None, "dt": None}
        xml = tf.ome_metadata
        if xml:
            import xml.etree.ElementTree as ET
            for el in ET.fromstring(xml).iter():
                if el.tag.endswith("Pixels"):
                    out["px"] = el.get("PhysicalSizeX")
                    out["z"] = el.get("PhysicalSizeZ")
                    out["dt"] = el.get("TimeIncrement")
                    break
        else:
            tag = tf.pages[0].tags.get("XResolution")
            if tag is not None:
                num, den = tag.value
                out["px"] = f"{den / num * 1e4:.4g} µm (baseline)" if num else None
    return out


def _defaults_from_run_params(position_dir: Path) -> tuple[float | None, float | None]:
    rp = position_dir / "0_input" / "run_params.json"
    if not rp.is_file():
        return None, None
    try:
        d = json.loads(rp.read_text())
    except (OSError, ValueError):
        return None, None
    return d.get("pixel_size_um"), d.get("time_interval_s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("position_dir", type=Path, help="e.g. .../pos00")
    ap.add_argument("--pixel-size", type=float, default=None, help="µm/px (default: run_params)")
    ap.add_argument("--z-size", type=float, default=1.0, help="µm per Z step (default 1.0)")
    ap.add_argument("--time-interval", type=float, default=None, help="s/frame (default: run_params)")
    ap.add_argument("--apply", action="store_true", help="write files (default: dry-run)")
    args = ap.parse_args()

    pos = args.position_dir
    rp_px, rp_dt = _defaults_from_run_params(pos)
    px = args.pixel_size if args.pixel_size is not None else rp_px
    dt = args.time_interval if args.time_interval is not None else rp_dt
    z = args.z_size

    if not px or px <= 0:
        ap.error("no positive pixel size (give --pixel-size or fix run_params.json)")

    tifs = sorted(pos.rglob("*.tif"))
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] {pos}")
    print(f"  pixel_size={px} µm/px   z_step={z} µm   time_interval={dt} s")
    print(f"  {len(tifs)} TIFF(s)\n")

    for path in tifs:
        info = plan_file(path)
        rel = path.relative_to(pos)
        # Preserved-description files keep their on-disk axes untouched.
        if info["preserve"] or info["axes_in"] == info["axes_out"]:
            ax = info["axes_in"]
        else:
            ax = f"{info['axes_in']}->{info['axes_out']}"
        line = f"  {str(rel):34s} {info['dtype']:8s} {ax:11s} {info['format']}"
        if args.apply:
            v = convert_file(path, px, z if "Z" in info["axes_out"] else None,
                             dt if "T" in info["axes_out"] else None)
            line += f"   ✓ px={v['px']} z={v['z']} dt={v['dt']}"
        print(line)

    if not args.apply:
        print("\n(dry-run — no files written; re-run with --apply)")


if __name__ == "__main__":
    main()
