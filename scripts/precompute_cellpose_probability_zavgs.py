#!/usr/bin/env python
"""Precompute sigmoid z-averaged Cellpose probability maps for positions."""
from __future__ import annotations

import argparse
from pathlib import Path

from cellflow.segmentation.cellpose_probability_zavg import (
    write_cellpose_probability_zavgs_for_root,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Position directory or parent directory containing position folders.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Keep existing cell_prob_zavg.tif and nucleus_prob_zavg.tif files.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results = write_cellpose_probability_zavgs_for_root(
        args.root,
        overwrite=not args.no_overwrite,
    )
    if not results:
        print(f"No position directories with 1_cellpose found under {args.root}")
        return 1
    for result in results:
        status = "SKIP" if result.skipped else "OK"
        print(f"{status} {result.position_dir}: {result.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
