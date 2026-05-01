#!/usr/bin/env python
"""Run the 2D frame Viterbi selector on a CellFlow hypotheses.h5 file."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import tifffile

from cellflow.tracking.frame_selector import (
    SelectorWeights,
    RankedPath,
    load_hypothesis_frame_stats,
    select_top_k_paths,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hypotheses_h5", type=Path, help="Path to 3_cell/hypotheses.h5")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <hypotheses parent>/frame_selector_experiment.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of ranked paths to keep.")
    parser.add_argument("--beam-width", type=int, default=200, help="Number of active partial paths to keep per frame.")
    parser.add_argument("--export-top", type=int, default=3, help="Number of top paths to export as TIFF stacks.")
    parser.add_argument("--area-weight", type=float, default=1.0)
    parser.add_argument("--shape-weight", type=float, default=1.0)
    parser.add_argument("--missing-weight", type=float, default=5.0)
    parser.add_argument("--extra-weight", type=float, default=2.0)
    parser.add_argument("--switch-weight", type=float, default=0.05)
    return parser.parse_args()


def _write_paths_csv(path: Path, paths: list[RankedPath]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "score", "p_path", "z_path", "state_path"])
        writer.writeheader()
        for rank, ranked_path in enumerate(paths, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "score": f"{ranked_path.score:.8f}",
                    "p_path": " ".join(str(state.p) for state in ranked_path.states),
                    "z_path": " ".join(str(state.z) for state in ranked_path.states),
                    "state_path": " ".join(f"p{state.p}:z{state.z}" for state in ranked_path.states),
                }
            )


def _write_transitions_csv(path: Path, paths: list[RankedPath]) -> None:
    with path.open("w", newline="") as f:
        fieldnames = [
            "rank",
            "t",
            "p_prev",
            "z_prev",
            "p",
            "z",
            "total",
            "area_cost",
            "shape_cost",
            "missing_count",
            "extra_count",
            "switch_cost",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, ranked_path in enumerate(paths, start=1):
            for state, prev_state, transition in zip(
                ranked_path.states[1:],
                ranked_path.states[:-1],
                ranked_path.transitions,
            ):
                writer.writerow(
                    {
                        "rank": rank,
                        "t": state.t,
                        "p_prev": prev_state.p,
                        "z_prev": prev_state.z,
                        "p": state.p,
                        "z": state.z,
                        "total": f"{transition.total:.8f}",
                        "area_cost": f"{transition.area_cost:.8f}",
                        "shape_cost": f"{transition.shape_cost:.8f}",
                        "missing_count": transition.missing_count,
                        "extra_count": transition.extra_count,
                        "switch_cost": f"{transition.switch_cost:.8f}",
                    }
                )


def _read_path_stack(h5_path: Path, ranked_path: RankedPath) -> np.ndarray:
    frames = []
    with h5py.File(h5_path, "r") as h5:
        root = h5["hypotheses"]
        for state in ranked_path.states:
            frames.append(root[f"t{state.t:03d}"][f"p{state.p:03d}"]["labels"][state.z])
    return np.stack(frames, axis=0)


def _export_tiffs(output_dir: Path, h5_path: Path, paths: list[RankedPath], export_top: int) -> None:
    for rank, ranked_path in enumerate(paths[:export_top], start=1):
        stack = _read_path_stack(h5_path, ranked_path)
        tifffile.imwrite(
            output_dir / f"rank_{rank:02d}_labels.tif",
            stack,
            compression="zlib",
        )


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir or args.hypotheses_h5.parent / "frame_selector_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = SelectorWeights(
        area=args.area_weight,
        shape=args.shape_weight,
        missing=args.missing_weight,
        extra=args.extra_weight,
        parameter_switch=args.switch_weight,
    )

    print(f"Loading frame stats from {args.hypotheses_h5}")
    candidates = load_hypothesis_frame_stats(args.hypotheses_h5)
    print(f"Loaded {len(candidates)} timepoints, {sum(len(c) for c in candidates)} 2D candidates")

    paths = select_top_k_paths(candidates, k=args.top_k, beam_width=args.beam_width, weights=weights)
    _write_paths_csv(output_dir / "ranked_paths.csv", paths)
    _write_transitions_csv(output_dir / "transition_scores.csv", paths)

    summary = {
        "hypotheses_h5": str(args.hypotheses_h5),
        "n_timepoints": len(candidates),
        "n_candidates": sum(len(c) for c in candidates),
        "top_k": args.top_k,
        "beam_width": args.beam_width,
        "export_top": args.export_top,
        "weights": {
            "area": weights.area,
            "shape": weights.shape,
            "missing": weights.missing,
            "extra": weights.extra,
            "parameter_switch": weights.parameter_switch,
        },
        "paths": [
            {
                "rank": rank,
                "score": path.score,
                "p_path": [state.p for state in path.states],
                "z_path": [state.z for state in path.states],
                "state_path": [{"t": state.t, "p": state.p, "z": state.z} for state in path.states],
            }
            for rank, path in enumerate(paths, start=1)
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if args.export_top > 0:
        print(f"Exporting top {min(args.export_top, len(paths))} TIFF stack(s)")
        _export_tiffs(output_dir, args.hypotheses_h5, paths, args.export_top)

    print(f"Wrote results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
