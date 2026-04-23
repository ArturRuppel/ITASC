from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "core" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "ultrack" / "src"))

from cellflow.ultrack.ingestion import write_hypothesis_labelmaps  # noqa: E402
from cellflow.ultrack.stages.seeded_tracker import (  # noqa: E402
    build_seeded_tracker_inputs,
    _match_frame,
    run_seeded_tracker,
)


def test_run_seeded_tracker_builds_a_single_2d_labels_stack(tmp_path):
    labelmap_a = np.array(
        [
            [0, 1, 1],
            [0, 1, 2],
            [3, 3, 2],
        ],
        dtype=np.uint32,
    )
    labelmap_b = np.array(
        [
            [0, 1, 1],
            [0, 4, 2],
            [3, 3, 2],
        ],
        dtype=np.uint32,
    )

    write_hypothesis_labelmaps(tmp_path, [labelmap_a, labelmap_b], stage_name="nucleus_ultrack")

    progress = list(run_seeded_tracker(tmp_path, overwrite=True))

    tracked = tifffile.imread(str(tmp_path / "tracked_labels.tif"))
    tracks = list(csv.DictReader((tmp_path / "tracks.csv").open()))

    assert progress[-1][2] == "Seeded tracker done."
    assert tracked.shape == (1, 3, 3)
    assert tracked.dtype == np.uint32
    assert tracked[0, 0, 1] == 1
    assert tracked[0, 1, 2] == 2
    assert tracks


def test_run_seeded_tracker_propagates_track_ids_across_frames(tmp_path):
    labelmap_a = np.array(
        [
            [
                [0, 1, 1],
                [0, 1, 1],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 2, 2],
                [0, 2, 2],
            ],
        ],
        dtype=np.uint32,
    )
    labelmap_b = np.array(
        [
            [
                [0, 1, 1],
                [0, 1, 1],
                [0, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 2, 2],
                [0, 2, 2],
            ],
        ],
        dtype=np.uint32,
    )

    write_hypothesis_labelmaps(tmp_path, [labelmap_a, labelmap_b], stage_name="nucleus_ultrack")

    list(run_seeded_tracker(tmp_path, overwrite=True))

    tracked = tifffile.imread(str(tmp_path / "tracked_labels.tif"))

    assert tracked.shape == (2, 3, 3)
    assert tracked[0, 0, 1] == tracked[1, 1, 1]
    assert tracked[0, 0, 1] == 1


def test_match_frame_does_not_create_new_ids(tmp_path):
    current = np.array(
        [
            [0, 1, 1, 0],
            [0, 2, 2, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint32,
    )
    candidate = np.array(
        [
            [0, 10, 10, 0],
            [0, 20, 20, 0],
            [0, 0, 0, 30],
        ],
        dtype=np.uint32,
    )

    next_frame, rows = _match_frame(current, candidate)

    assert next_frame.max() == 2
    assert np.array_equal(
        next_frame,
        np.array(
            [
                [0, 1, 1, 0],
                [0, 2, 2, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.uint32,
        ),
    )
    assert [row["track_id"] for row in rows] == [1, 2]
    assert all(row["candidate_label_id"] in {10, 20} for row in rows)


def test_build_seeded_tracker_inputs_uses_overlap_medoid_seed(tmp_path):
    labelmap_a = np.array(
        [
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ],
        dtype=np.uint32,
    )
    labelmap_b = np.array(
        [
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ],
        dtype=np.uint32,
    )
    labelmap_c = np.array(
        [
            [
                [0, 1, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 1, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        ],
        dtype=np.uint32,
    )

    write_hypothesis_labelmaps(tmp_path, [labelmap_a, labelmap_b, labelmap_c], stage_name="nucleus_ultrack")

    _, _, seed, seed_source = build_seeded_tracker_inputs(tmp_path)

    assert seed_source.startswith("medoid:")
    assert seed.shape == (4, 4)
    assert np.array_equal(seed, labelmap_a[0])
