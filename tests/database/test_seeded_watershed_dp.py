from __future__ import annotations

import numpy as np
import h5py

from cellflow.database import hypotheses
from cellflow.database.hypotheses import (
    ContourWatershedParams,
    ContourWatershedSweepSpec,
    HypothesisRecord,
    SeededWatershedParams,
    SeededWatershedSweepSpec,
    iter_contour_watershed_records,
    iter_seeded_watershed_records,
    iter_write_hypothesis_sweep_h5,
    read_hypothesis_labels,
    _ordered_bounded_map,
)
from cellflow.segmentation import compute_contour_watershed


def test_seeded_watershed_records_normalize_channel_first_dp(monkeypatch):
    prob = np.zeros((1, 3, 4, 5), dtype=np.float32)
    nucleus = np.ones((1, 3, 4, 5), dtype=np.uint32)
    dp = np.zeros((1, 2, 3, 4, 5), dtype=np.float32)
    dp[:, 0] = 3
    dp[:, 1] = 4
    seen_shapes = []

    def fake_seeded_watershed(prob_2d, dp_2d, seeds_2d, params):
        seen_shapes.append(dp_2d.shape)
        return np.ones(prob_2d.shape, dtype=np.uint32)

    monkeypatch.setattr(hypotheses, "compute_seeded_watershed", fake_seeded_watershed)

    spec = SeededWatershedSweepSpec(
        basin="flow_mag",
        foreground_threshold=0.5,
        foreground_threshold_min=0.5,
        foreground_threshold_max=0.5,
        compactness=0.0,
        compactness_min=0.0,
        compactness_max=0.0,
    )

    records = list(iter_seeded_watershed_records(prob, dp, nucleus, spec))

    assert len(records) == 1
    assert records[0].labels.shape == (3, 4, 5)
    assert seen_shapes == [(2, 4, 5), (2, 4, 5), (2, 4, 5)]


def test_seeded_watershed_records_prefer_canonical_dp_when_z_is_channel_sized(monkeypatch):
    prob = np.zeros((1, 2, 4, 5), dtype=np.float32)
    nucleus = np.ones((1, 2, 4, 5), dtype=np.uint32)
    dp = np.zeros((1, 2, 2, 4, 5), dtype=np.float32)
    dp[0, 0, 0] = 1
    dp[0, 0, 1] = 10
    dp[0, 1, 0] = 2
    dp[0, 1, 1] = 20
    seen_values = []

    def fake_seeded_watershed(prob_2d, dp_2d, seeds_2d, params):
        seen_values.append((float(dp_2d[0, 0, 0]), float(dp_2d[1, 0, 0])))
        return np.ones(prob_2d.shape, dtype=np.uint32)

    monkeypatch.setattr(hypotheses, "compute_seeded_watershed", fake_seeded_watershed)

    spec = SeededWatershedSweepSpec(
        basin="flow_mag",
        foreground_threshold=0.5,
        foreground_threshold_min=0.5,
        foreground_threshold_max=0.5,
        compactness=0.0,
        compactness_min=0.0,
        compactness_max=0.0,
    )

    list(iter_seeded_watershed_records(prob, dp, nucleus, spec))

    assert seen_values == [(1.0, 10.0), (2.0, 20.0)]


def test_seeded_watershed_records_accept_ui_time_first_2d_nucleus_labels(monkeypatch):
    prob = np.zeros((2, 3, 4, 5), dtype=np.float32)
    nucleus = np.zeros((1, 2, 4, 5), dtype=np.uint32)
    nucleus[0, 0, 1:3, 2:4] = 7
    nucleus[0, 1, 1:3, 2:4] = 11
    seen_seed_maxima = []

    def fake_seeded_watershed(prob_2d, dp_2d, seeds_2d, params):
        seen_seed_maxima.append(int(seeds_2d.max()))
        return np.full(prob_2d.shape, int(seeds_2d.max()), dtype=np.uint32)

    monkeypatch.setattr(hypotheses, "compute_seeded_watershed", fake_seeded_watershed)

    spec = SeededWatershedSweepSpec(
        foreground_threshold=0.5,
        foreground_threshold_min=0.5,
        foreground_threshold_max=0.5,
        compactness=0.0,
        compactness_min=0.0,
        compactness_max=0.0,
    )

    records = list(iter_seeded_watershed_records(prob, None, nucleus, spec))

    assert [record.t for record in records] == [0, 1]
    assert [int(record.labels.max()) for record in records] == [7, 11]
    assert seen_seed_maxima == [7, 7, 7, 11, 11, 11]


def test_iter_write_hypothesis_sweep_h5_streams_records(tmp_path):
    output_path = tmp_path / "hypotheses.h5"
    params = SeededWatershedParams()
    consumed = []

    def records():
        for t in range(3):
            consumed.append(t)
            labels = np.full((1, 4, 5), t + 1, dtype=np.uint32)
            yield HypothesisRecord(t=t, p=0, labels=labels, params=params)

    progress = iter_write_hypothesis_sweep_h5(output_path, records(), overwrite=True)

    assert consumed == []
    assert next(progress) == 1
    assert consumed == [0]
    assert np.array_equal(read_hypothesis_labels(output_path, 0, 0), np.full((1, 4, 5), 1, dtype=np.uint32))

    assert list(progress) == [2, 3]
    assert consumed == [0, 1, 2]


def test_iter_write_hypothesis_sweep_h5_accepts_fast_compression(tmp_path):
    output_path = tmp_path / "hypotheses.h5"
    params = SeededWatershedParams()
    records = [
        HypothesisRecord(
            t=0,
            p=0,
            labels=np.ones((1, 4, 5), dtype=np.uint32),
            params=params,
        )
    ]

    list(
        iter_write_hypothesis_sweep_h5(
            output_path,
            records,
            overwrite=True,
            compression="lzf",
        )
    )

    with h5py.File(output_path, "r") as h5:
        labels = h5["hypotheses/t000/p000/labels"]
        assert labels.compression == "lzf"
        assert labels.compression_opts is None


def test_ordered_bounded_map_does_not_consume_all_inputs_before_first_result():
    consumed = []

    def inputs():
        for value in range(10):
            consumed.append(value)
            yield value

    mapped = _ordered_bounded_map(lambda value: value * 2, inputs(), max_workers=2)

    assert next(mapped) == 0
    assert consumed == [0, 1]
    assert next(mapped) == 2
    assert consumed == [0, 1, 2]
    assert list(mapped) == [4, 6, 8, 10, 12, 14, 16, 18]


def test_contour_watershed_records_accept_filtered_params_and_stream_tasks(monkeypatch):
    contour = np.zeros((3, 4, 5), dtype=np.float32)
    foreground = np.zeros((3, 4, 5), dtype=np.float32)
    params = [
        ContourWatershedParams(seed_distance=5, noise_scale=0.1),
        ContourWatershedParams(seed_distance=7, noise_scale=0.1),
    ]
    seen = []

    def fake_task(args):
        t, p_idx, param, contour_frame, fg_frame = args
        seen.append((t, p_idx, param.seed_distance))
        return HypothesisRecord(
            t=t,
            p=p_idx,
            labels=np.full((1, *contour_frame.shape), param.seed_distance, dtype=np.uint32),
            params=param,
        )

    monkeypatch.setattr(hypotheses, "_run_watershed_task", fake_task)

    records = iter_contour_watershed_records(
        contour,
        foreground,
        ContourWatershedSweepSpec(),
        n_workers=2,
        params_list=params,
    )

    first = next(records)

    assert first.p == 0
    assert int(first.labels.max()) == 5
    assert seen == [(0, 0, 5)]
    assert [(record.t, record.p, int(record.labels.max())) for record in records] == [
        (0, 1, 7),
        (1, 0, 5),
        (1, 1, 7),
        (2, 0, 5),
        (2, 1, 7),
    ]


def test_contour_watershed_records_reuse_deterministic_frame_cache(monkeypatch):
    contour = np.zeros((1, 10, 10), dtype=np.float32)
    foreground = np.ones((1, 10, 10), dtype=np.float32)
    params = [
        ContourWatershedParams(seed_distance=3, foreground_threshold=0.5, ridge_threshold=0.5),
        ContourWatershedParams(seed_distance=5, foreground_threshold=0.5, ridge_threshold=0.5),
    ]
    prepared = []

    def fake_prepare(boundary, foreground_frame, params):
        prepared.append((params.foreground_threshold, params.ridge_threshold))
        return object()

    def fake_cached_task(args):
        t, p_idx, param, boundary, cached = args
        return HypothesisRecord(
            t=t,
            p=p_idx,
            labels=np.full((1, 10, 10), param.seed_distance, dtype=np.uint32),
            params=param,
        )

    monkeypatch.setattr(hypotheses, "_prepare_contour_watershed_frame", fake_prepare)
    monkeypatch.setattr(hypotheses, "_run_cached_watershed_task", fake_cached_task)

    records = list(
        iter_contour_watershed_records(
            contour,
            foreground,
            ContourWatershedSweepSpec(),
            params_list=params,
        )
    )

    assert [int(record.labels.max()) for record in records] == [3, 5]
    assert prepared == [(0.5, 0.5)]


def test_contour_watershed_cached_records_match_direct_computation():
    y, x = np.mgrid[:24, :24]
    foreground = np.exp(-((y - 8) ** 2 + (x - 8) ** 2) / 24.0)
    foreground += np.exp(-((y - 16) ** 2 + (x - 16) ** 2) / 24.0)
    foreground = (foreground / foreground.max()).astype(np.float32)
    boundary = np.zeros_like(foreground, dtype=np.float32)
    boundary[11:13, :] = 1.0
    boundary[:, 11:13] = 1.0
    params = [
        ContourWatershedParams(seed_distance=3, foreground_threshold=0.2, ridge_threshold=0.5),
        ContourWatershedParams(seed_distance=5, foreground_threshold=0.2, ridge_threshold=0.5),
    ]

    records = list(
        iter_contour_watershed_records(
            boundary[np.newaxis],
            foreground[np.newaxis],
            ContourWatershedSweepSpec(),
            params_list=params,
        )
    )

    direct = [compute_contour_watershed(boundary, foreground, param) for param in params]
    assert [record.p for record in records] == [0, 1]
    assert all(np.array_equal(record.labels[0], expected) for record, expected in zip(records, direct))
