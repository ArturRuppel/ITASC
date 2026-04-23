from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "core" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "ultrack" / "src"))

from cellflow.ultrack.hypotheses import (
    HypothesisRecord,
    NucleusHypothesisParams,
    NucleusHypothesisSweepSpec,
    build_parameter_sets,
    compute_medoid_stack,
    iter_hypothesis_records_from_stacks,
    iter_hypothesis_records,
    load_hypotheses_h5_lazy,
    load_medoid_stack,
    write_hypothesis_sweep_h5,
    write_medoid_stack,
)


def test_build_parameter_sets_cartesian_product_order():
    spec = NucleusHypothesisSweepSpec(
        basins=("prob", "flow_mag"),
        threshold=0.0,
        threshold_min=0.0,
        threshold_max=20.0,
        threshold_step=10.0,
        compactness=0.25,
        compactness_min=0.25,
        compactness_max=0.25,
        compactness_step=0.01,
        smooth_sigma=1.0,
        smooth_min=1.0,
        smooth_max=2.0,
        smooth_step=1.0,
    )

    params = build_parameter_sets(spec)

    assert len(params) == 12
    assert params[0] == NucleusHypothesisParams(
        basin="prob",
        threshold_pct=0.0,
        compactness=0.25,
        smooth_sigma=1.0,
        seed_source="auto",
    )
    assert params[-1] == NucleusHypothesisParams(
        basin="flow_mag",
        threshold_pct=20.0,
        compactness=0.25,
        smooth_sigma=2.0,
        seed_source="auto",
    )


def test_write_and_iter_hypothesis_sweep_h5_round_trip(tmp_path):
    records = [
        HypothesisRecord(
            t=0,
            z=0,
            p=0,
            labels=np.array([[0, 1], [2, 2]], dtype=np.uint32),
            params=NucleusHypothesisParams(
                basin="prob",
                threshold_pct=5.0,
                compactness=0.0,
                smooth_sigma=0.0,
                seed_source="active_layer",
            ),
        ),
        HypothesisRecord(
            t=0,
            z=1,
            p=1,
            labels=np.array([[3, 3], [0, 4]], dtype=np.uint32),
            params=NucleusHypothesisParams(
                basin="flow_mag",
                threshold_pct=10.0,
                compactness=0.5,
                smooth_sigma=1.5,
                seed_source="disk_corrected",
            ),
        ),
    ]

    out = write_hypothesis_sweep_h5(
        tmp_path / "hypotheses.h5",
        records,
        source="unit-test",
        n_t=1,
        n_z=2,
        n_p=2,
    )

    with h5py.File(out, "r") as f:
        assert f.attrs["axes"] == "TZP"
        assert f.attrs["layout"] == "hypotheses/t{t:03d}/z{z:03d}/p{p:03d}/labels"
        labels = f["hypotheses/t000/z001/p001/labels"][:]
        assert np.array_equal(labels, records[1].labels)
        assert f["hypotheses/t000/z001/p001"].attrs["basin"] == "flow_mag"
        assert f["hypotheses/t000/z001/p001"].attrs["threshold_pct"] == 10.0

    loaded = list(iter_hypothesis_records(out))
    assert len(loaded) == 2
    assert loaded[0].params == records[0].params
    assert np.array_equal(loaded[1].labels, records[1].labels)


def test_iter_hypothesis_records_from_stacks_uses_t_z_p_order(tmp_path):
    prob = np.array(
        [
            [0.1, 0.9],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    dp = np.zeros((2, 2, 2), dtype=np.float32)
    seeds = np.array(
        [
            [0, 1],
            [2, 2],
        ],
        dtype=np.int32,
    )

    spec = NucleusHypothesisSweepSpec(
        basins=("prob", "flow_mag"),
        threshold=0.0,
        threshold_min=0.0,
        threshold_max=0.0,
        threshold_step=1.0,
        compactness=0.0,
        compactness_min=0.0,
        compactness_max=0.0,
        compactness_step=0.01,
        smooth_sigma=0.0,
        smooth_min=0.0,
        smooth_max=0.0,
        smooth_step=0.25,
    )

    records = list(iter_hypothesis_records_from_stacks(prob, dp, seeds, spec))

    assert len(records) == 2
    assert [record.t for record in records] == [0, 0]
    assert [record.z for record in records] == [0, 0]
    assert [record.p for record in records] == [0, 1]

    out = write_hypothesis_sweep_h5(tmp_path / "hypotheses.h5", records)
    with h5py.File(out, "r") as f:
        assert "hypotheses/t000/z000/p000/labels" in f
        assert "hypotheses/t000/z000/p001/labels" in f


def test_compute_medoid_stack_shape_and_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    spatial = (4, 4)

    def _make_record(t, z, p):
        return HypothesisRecord(
            t=t,
            z=z,
            p=p,
            labels=rng.integers(0, 5, size=spatial, dtype=np.uint32),
            params=NucleusHypothesisParams(
                basin="prob", threshold_pct=0.0, compactness=0.0, smooth_sigma=0.0
            ),
        )

    # 2 timepoints, 3 z-slices, 2 param sets → 12 records
    records = [_make_record(t, z, p) for t in range(2) for z in range(3) for p in range(2)]

    medoid = compute_medoid_stack(records, n_t=2)
    assert medoid.shape == (*spatial, 2), f"Expected {(*spatial, 2)}, got {medoid.shape}"
    assert medoid.dtype == np.uint32

    # Each medoid slice must be one of the input images for that t
    for t in range(2):
        t_imgs = [rec.labels for rec in records if rec.t == t]
        slice_t = medoid[..., t]
        assert any(np.array_equal(slice_t, img) for img in t_imgs), \
            f"Medoid at t={t} is not one of the input images"

    # Round-trip through HDF5
    out = write_hypothesis_sweep_h5(tmp_path / "hypotheses.h5", records, n_t=2, n_z=3, n_p=2)
    with h5py.File(out, "a") as f:
        write_medoid_stack(f, medoid)

    loaded = load_medoid_stack(out)
    assert np.array_equal(loaded, medoid)

    with h5py.File(out, "r") as f:
        assert f["medoid_stack"].attrs["axes"] == "YXT"


def test_lazy_hdf5_loader_returns_5d_array(tmp_path):
    records = [
        HypothesisRecord(
            t=0,
            z=0,
            p=0,
            labels=np.array([[0, 1], [2, 2]], dtype=np.uint32),
            params=NucleusHypothesisParams(
                basin="prob",
                threshold_pct=5.0,
                compactness=0.0,
                smooth_sigma=0.0,
                seed_source="active_layer",
            ),
        ),
    ]

    out = write_hypothesis_sweep_h5(
        tmp_path / "hypotheses.h5",
        records,
        n_t=1,
        n_z=1,
        n_p=1,
    )

    lazy = load_hypotheses_h5_lazy(out)
    assert lazy.shape == (1, 1, 1, 2, 2)
    assert np.array_equal(lazy[0, 0, 0].compute(), records[0].labels)
