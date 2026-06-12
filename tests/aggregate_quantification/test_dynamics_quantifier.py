"""Registry wiring + end-to-end build/read for the dynamics quantifiers."""
import numpy as np
import tifffile
from skimage.draw import disk

from cellflow.aggregate_quantification.dynamics import read_track_dynamics
from cellflow.aggregate_quantification.quantifier import (
    PositionInputs,
    available_quantifiers,
)
from cellflow.aggregate_quantification.quantifiers.cell_dynamics import (
    CellDynamicsQuantifier,
)
from cellflow.aggregate_quantification.quantifiers.nucleus_dynamics import (
    NucleusDynamicsQuantifier,
)


def _moving_disk_stack(centers, shape=(80, 80), radius=6, label=1):
    frames = []
    for row, col in centers:
        frame = np.zeros(shape, dtype=np.uint16)
        rr, cc = disk((row, col), radius, shape=shape)
        frame[rr, cc] = label
        frames.append(frame)
    return np.stack(frames)


def test_available_quantifiers_discovers_dynamics_pair():
    ids = {cls.quantity_id for cls in available_quantifiers()}
    assert {"cell_dynamics", "nucleus_dynamics"} <= ids


def test_dynamics_outputs_nest_under_aggregate_quantification(tmp_path):
    inputs = PositionInputs(position_dir=tmp_path)
    assert CellDynamicsQuantifier().default_output(inputs) == (
        tmp_path / "aggregate_quantification" / "cell_dynamics.h5"
    )
    assert NucleusDynamicsQuantifier().default_output(inputs) == (
        tmp_path / "aggregate_quantification" / "nucleus_dynamics.h5"
    )


def test_dynamics_requires_labels_with_global_pixel_size_and_interval(tmp_path):
    q = CellDynamicsQuantifier()
    # Pixel size + frame interval are global build params now; the per-position
    # file requirement is just the cell labels.
    assert q.requires == ("cell_labels_path",)
    assert q.required_build_params == {
        "pixel_size_um": "pixel size (µm/px)",
        "time_interval_s": "frame interval (s)",
    }
    labels = tmp_path / "cells.tif"
    tifffile.imwrite(labels, _moving_disk_stack([(40, 40), (40, 41)]))
    # The cell labels alone satisfy the per-position file gate; px/Δt gating is the
    # studio's required_build_params, checked separately.
    assert q.can_build(
        PositionInputs(position_dir=tmp_path, cell_labels_path=labels)
    ) is True
    assert q.can_build(PositionInputs(position_dir=tmp_path)) is False


def test_build_read_roundtrip_ballistic_track(tmp_path):
    # One cell moving 2 px/frame in x for 16 frames -> straight, ballistic.
    centers = [(40, 10 + 2 * i) for i in range(16)]
    labels = tmp_path / "cells.tif"
    tifffile.imwrite(labels, _moving_disk_stack(centers))
    out = tmp_path / "aggregate_quantification" / "cell_dynamics.h5"

    q = CellDynamicsQuantifier()
    inputs = PositionInputs(
        position_dir=tmp_path,
        cell_labels_path=labels,
        pixel_size_um=0.5,
        time_interval_s=2.0,
    )
    result = q.build(inputs, out)
    assert result == out and out.exists()

    dyn = read_track_dynamics(out)
    # 1 track, 16 frames in the instantaneous table.
    assert set(dyn.instantaneous) >= {"frame", "cell_id", "speed_um_per_s", "net_disp_um"}
    assert dyn.instantaneous["cell_id"].max() == 1
    assert dyn.instantaneous["frame"].size == 16
    # 2 px/frame · 0.5 µm/px ÷ 2 s/frame = 0.5 µm/s.
    np.testing.assert_allclose(
        np.nanmean(dyn.instantaneous["speed_um_per_s"]), 0.5, atol=1e-6
    )
    # Straight motion: directionality ratio ≈ 1, MSD exponent ≈ 2.
    np.testing.assert_allclose(dyn.tracks["directionality_ratio"], [1.0], atol=1e-6)
    assert abs(dyn.msd_alpha - 2.0) < 0.05

    # object_table serves the tidy contract (frame + cell_id present).
    table = q.object_table(out)
    assert "frame" in table and "cell_id" in table


def test_nucleus_dynamics_reads_nucleus_labels(tmp_path):
    centers = [(40, 10 + 2 * i) for i in range(8)]
    nuclei = tmp_path / "nuclei.tif"
    tifffile.imwrite(nuclei, _moving_disk_stack(centers, radius=4))
    out = tmp_path / "nucleus_dynamics.h5"

    q = NucleusDynamicsQuantifier()
    inputs = PositionInputs(
        position_dir=tmp_path,
        nucleus_labels_path=nuclei,
        pixel_size_um=1.0,
        time_interval_s=1.0,
    )
    q.build(inputs, out)
    dyn = read_track_dynamics(out)
    assert dyn.instantaneous["frame"].size == 8


def test_build_adds_per_track_msd_and_corr_length(tmp_path):
    # One straight, ballistic track: per-track MSD columns land in the tracks
    # table and the pooled correlation length attr round-trips.
    centers = [(40, 10 + 2 * i) for i in range(16)]
    labels = tmp_path / "cells.tif"
    tifffile.imwrite(labels, _moving_disk_stack(centers))
    out = tmp_path / "aggregate_quantification" / "cell_dynamics.h5"

    q = CellDynamicsQuantifier()
    inputs = PositionInputs(
        position_dir=tmp_path, cell_labels_path=labels, pixel_size_um=0.5, time_interval_s=2.0
    )
    q.build(inputs, out)

    dyn = read_track_dynamics(out)
    assert {"msd_D_um2_per_s", "msd_alpha", "msd_r2"} <= set(dyn.tracks)
    # Straight single track is ballistic per-track too: α ≈ 2.
    np.testing.assert_allclose(dyn.tracks["msd_alpha"], [2.0], atol=0.05)
    # A single isolated track has no pair correlation -> NaN ξ; the attr exists.
    assert hasattr(dyn, "corr_length_um")
    assert np.isnan(dyn.corr_length_um) or dyn.corr_length_um >= 0.0


def test_read_track_dynamics_backward_compatible_without_corr_length(tmp_path):
    # An .h5 built before this change lacks the corr_length_um attr -> loads as NaN.
    import h5py

    centers = [(40, 10 + 2 * i) for i in range(16)]
    labels = tmp_path / "cells.tif"
    tifffile.imwrite(labels, _moving_disk_stack(centers))
    out = tmp_path / "cell_dynamics.h5"
    q = CellDynamicsQuantifier()
    inputs = PositionInputs(
        position_dir=tmp_path, cell_labels_path=labels, pixel_size_um=0.5, time_interval_s=2.0
    )
    q.build(inputs, out)
    with h5py.File(out, "a") as h5:
        del h5["corr_curve/table"].attrs["corr_length_um"]

    dyn = read_track_dynamics(out)
    assert np.isnan(dyn.corr_length_um)
