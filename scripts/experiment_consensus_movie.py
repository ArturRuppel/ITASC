#!/usr/bin/env python
"""Build compactness-grouped consensus label movies from hypotheses.h5."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import tifffile

from cellflow.tracking.consensus_movie import (
    CompactnessGroup,
    build_consensus_movie_with_thresholds,
    load_compactness_groups,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hypotheses_h5", type=Path, help="Path to 3_cell/hypotheses.h5")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <hypotheses parent>/consensus_movie_experiment.",
    )
    parser.add_argument(
        "--vote-threshold",
        type=float,
        default=0.5,
        help="Fixed support threshold, or fallback threshold for percentile mode.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("fixed", "percentile"),
        default="fixed",
        help="Use one fixed threshold or choose one threshold per frame from support percentiles.",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=60.0,
        help="Support percentile used when --threshold-mode=percentile.",
    )
    parser.add_argument(
        "--min-vote-threshold",
        type=float,
        default=0.35,
        help="Lower clamp for dynamic percentile thresholds.",
    )
    parser.add_argument(
        "--max-vote-threshold",
        type=float,
        default=0.65,
        help="Upper clamp for dynamic percentile thresholds.",
    )
    parser.add_argument(
        "--compactness",
        type=float,
        action="append",
        default=None,
        help="Compactness value to export. May be repeated. Defaults to all compactness values.",
    )
    parser.add_argument(
        "--no-temporal-smoothing",
        action="store_true",
        help="Use per-frame consensus only, without 3-frame temporal smoothing.",
    )
    parser.add_argument(
        "--no-export-support",
        action="store_true",
        help="Do not export uint8 support maps.",
    )
    return parser.parse_args()


def _format_compactness(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def _select_groups(
    groups: list[CompactnessGroup],
    compactness_values: list[float] | None,
) -> list[CompactnessGroup]:
    if compactness_values is None:
        return groups
    selected = []
    for requested in compactness_values:
        matches = [group for group in groups if np.isclose(group.compactness, requested)]
        if not matches:
            available = ", ".join(f"{group.compactness:g}" for group in groups)
            raise ValueError(f"Compactness {requested:g} not found. Available: {available}")
        selected.append(matches[0])
    return selected


def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "compactness",
        "n_parameters",
        "n_votes_per_frame",
        "label_path",
        "support_path",
        "mean_support",
        "mean_foreground_fraction",
        "threshold_min",
        "threshold_mean",
        "threshold_max",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_thresholds_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["compactness", "t", "threshold"])
        writer.writeheader()
        writer.writerows(rows)


def _z_slices_for_group(hypotheses_h5: Path, group: CompactnessGroup) -> int:
    with h5py.File(hypotheses_h5, "r") as h5:
        root = h5["hypotheses"]
        first_t = sorted(k for k in root.keys() if k.startswith("t"))[0]
        return int(root[first_t][f"p{group.members[0].p:03d}"]["labels"].shape[0])


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir or args.hypotheses_h5.parent / "consensus_movie_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = _select_groups(load_compactness_groups(args.hypotheses_h5), args.compactness)
    rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    for group in groups:
        prefix = f"compactness_{_format_compactness(group.compactness)}"
        z_slices = _z_slices_for_group(args.hypotheses_h5, group)
        votes_per_frame = len(group.members) * z_slices
        print(
            f"Building {prefix}: {len(group.members)} parameter(s), "
            f"{votes_per_frame} votes per frame"
        )
        movie = build_consensus_movie_with_thresholds(
            args.hypotheses_h5,
            group,
            vote_threshold=args.vote_threshold,
            smooth_temporally=not args.no_temporal_smoothing,
            threshold_mode=args.threshold_mode,
            threshold_percentile=args.threshold_percentile,
            min_vote_threshold=args.min_vote_threshold,
            max_vote_threshold=args.max_vote_threshold,
        )
        labels = movie.labels
        support = movie.support

        label_path = output_dir / f"{prefix}_labels.tif"
        tifffile.imwrite(label_path, labels, compression="zlib")

        support_path = ""
        if not args.no_export_support:
            support_u8 = np.clip(np.rint(support * 255.0), 0, 255).astype(np.uint8)
            support_tif = output_dir / f"{prefix}_support_u8.tif"
            tifffile.imwrite(support_tif, support_u8, compression="zlib")
            support_path = str(support_tif)

        for t, threshold in enumerate(movie.thresholds):
            threshold_rows.append(
                {
                    "compactness": group.compactness,
                    "t": t,
                    "threshold": f"{float(threshold):.6f}",
                }
            )

        rows.append(
            {
                "compactness": group.compactness,
                "n_parameters": len(group.members),
                "n_votes_per_frame": votes_per_frame,
                "label_path": str(label_path),
                "support_path": support_path,
                "mean_support": f"{float(np.mean(support)):.6f}",
                "mean_foreground_fraction": f"{float(np.mean(labels > 0)):.6f}",
                "threshold_min": f"{float(np.min(movie.thresholds)):.6f}",
                "threshold_mean": f"{float(np.mean(movie.thresholds)):.6f}",
                "threshold_max": f"{float(np.max(movie.thresholds)):.6f}",
            }
        )

    _write_summary_csv(output_dir / "summary.csv", rows)
    _write_thresholds_csv(output_dir / "thresholds.csv", threshold_rows)
    summary = {
        "hypotheses_h5": str(args.hypotheses_h5),
        "threshold_mode": args.threshold_mode,
        "vote_threshold": args.vote_threshold,
        "threshold_percentile": args.threshold_percentile,
        "min_vote_threshold": args.min_vote_threshold,
        "max_vote_threshold": args.max_vote_threshold,
        "temporal_smoothing": not args.no_temporal_smoothing,
        "export_support": not args.no_export_support,
        "groups": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote consensus outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
