"""Derive small 2D / 2D+t / 3D / 3D+t raw-input stacks for dimensionality testing.

The divergence-map path runs raw image -> `to_tzyx(arr, layout)` -> cellpose
`run_*_stack` -> `write_outputs` (prob/dp) -> `build_divergence_maps`. The TODO item
"check the nucleus/cell divergence path works on 2D, 2Dt, 3D, 3Dt inputs" needs one
small raw stack of each layout to feed that path end to end.

Source: `examples/data/full_example/pos00/0_input/{nucleus,cell}.tif`, real
(T=10, Z=4, 256, 256) uint16 intensity. We crop to a central 128x128 window and keep
T=3, Z=3 so the fixtures are tiny (~0.3 MB/channel) but still contain cells.

Each file is written with explicit `axes` metadata. This matters for the ambiguous
case: a 2D+t stack `(T, Y, X)` and a 3D stack `(Z, Y, X)` are shape-identical
`(3, 128, 128)` and `infer_layout_from_ndim(3)` returns None, so only the axes label
distinguishes them.

Run from the repo root:  python scripts/make_dimensionality_fixtures.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from cellflow.core.tiff import imwrite_grayscale

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "examples/data/full_example/pos00/0_input"
OUT = REPO / "examples/data/dimensionality_test"

# Central spatial crop + how many frames/slices to keep. Small on purpose.
Y0, Y1 = 64, 192
X0, X1 = 64, 192
N_T = 3
N_Z = 3
# A mid t / mid z to collapse when a layout drops that axis (in-focus, mid-movie).
T_PICK = 5
Z_PICK = 2


def _load(channel: str) -> np.ndarray:
    """Full source stack for a channel, cropped spatially. Shape (T, Z, Yc, Xc)."""
    arr = tifffile.imread(SRC / f"{channel}.tif")
    if arr.ndim != 4:
        raise ValueError(f"expected (T, Z, Y, X) source, got {arr.shape}")
    return arr[:, :, Y0:Y1, X0:X1]


def _write(path: Path, data: np.ndarray, axes: str) -> None:
    imwrite_grayscale(path, data, compression="zlib", metadata={"axes": axes})
    print(f"  {path.name:20s} {axes:5s} {tuple(data.shape)}  {data.dtype}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for channel in ("nucleus", "cell"):
        full = _load(channel)  # (T, Z, Yc, Xc)
        print(f"{channel}: source crop {tuple(full.shape)}")
        # 2D    -> (Y, X)          one t, one z
        _write(OUT / f"{channel}_2d.tif", full[T_PICK, Z_PICK], "YX")
        # 2D+t  -> (T, Y, X)       one z, several t
        _write(OUT / f"{channel}_2dt.tif", full[:N_T, Z_PICK], "TYX")
        # 3D    -> (Z, Y, X)       one t, several z
        _write(OUT / f"{channel}_3d.tif", full[T_PICK, :N_Z], "ZYX")
        # 3D+t  -> (T, Z, Y, X)    several t, several z
        _write(OUT / f"{channel}.tif", full[:N_T, :N_Z], "TZYX")
    print(f"\nWrote fixtures to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
