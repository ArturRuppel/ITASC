"""Unit tests for the headless dynamics core (no Qt, exact synthetic inputs)."""
import math

import numpy as np

from cellflow.aggregate_quantification.dynamics.collective import (
    _one_over_e_length,
    _order_parameter,
    collective_tables,
)
from cellflow.aggregate_quantification.dynamics.kinematics import (
    fit_persistence_time,
    instantaneous_table,
    track_summary_table,
)
from cellflow.aggregate_quantification.dynamics.msd import (
    ensemble_msd,
    fit_msd_power_law,
)
from cellflow.aggregate_quantification.dynamics.trajectories import Trajectory


def _straight_track(track_id=1, n=6, step=(1.0, 0.0), start_frame=0):
    frames = np.arange(start_frame, start_frame + n, dtype=np.int64)
    xy = np.array([(i * step[0], i * step[1]) for i in range(n)], dtype=float)
    return Trajectory(track_id=track_id, frames=frames, xy=xy)


# ------------------------------------------------------------------ kinematics
def test_instantaneous_velocity_uses_real_elapsed_time():
    traj = _straight_track(n=4, step=(1.0, 0.0))  # 1 µm/frame in x
    table = instantaneous_table([traj], time_interval_s=2.0)  # 2 s/frame -> 0.5 µm/s
    assert table["frame"].tolist() == [0, 1, 2, 3]
    np.testing.assert_allclose(table["speed_um_per_s"][:3], 0.5)
    # Last present frame has no forward step.
    assert math.isnan(table["speed_um_per_s"][-1])
    np.testing.assert_allclose(table["net_disp_um"], [0.0, 1.0, 2.0, 3.0])


def test_instantaneous_velocity_spans_gap_by_elapsed_time():
    # Frames 0,1,3 -> the 1->3 step spans a gap (Δf=2); speed divides by 2·dt.
    traj = Trajectory(
        track_id=1,
        frames=np.array([0, 1, 3], dtype=np.int64),
        xy=np.array([(0.0, 0.0), (1.0, 0.0), (3.0, 0.0)]),
    )
    table = instantaneous_table([traj], time_interval_s=1.0)
    np.testing.assert_allclose(table["speed_um_per_s"][0], 1.0)  # 1 µm over 1·dt
    np.testing.assert_allclose(table["speed_um_per_s"][1], 1.0)  # 2 µm over 2·dt


def test_track_summary_ratios_and_gap_count():
    traj = Trajectory(
        track_id=7,
        frames=np.array([0, 1, 3], dtype=np.int64),
        xy=np.array([(0.0, 0.0), (1.0, 0.0), (3.0, 0.0)]),
    )
    table = track_summary_table([traj], time_interval_s=1.0, min_track_frames=3)
    assert table["cell_id"].tolist() == [7]
    assert table["n_gaps"].tolist() == [1]
    np.testing.assert_allclose(table["path_length_um"], [3.0])
    np.testing.assert_allclose(table["net_displacement_um"], [3.0])
    np.testing.assert_allclose(table["directionality_ratio"], [1.0])  # perfectly straight
    np.testing.assert_allclose(table["duration_s"], [3.0])
    np.testing.assert_allclose(table["curvilinear_speed_um_per_s"], [1.0])


def test_track_summary_drops_short_tracks():
    short = _straight_track(track_id=1, n=2)
    ok = _straight_track(track_id=2, n=5)
    table = track_summary_table([short, ok], time_interval_s=1.0, min_track_frames=3)
    assert table["cell_id"].tolist() == [2]


def test_fit_persistence_time_recovers_decay_constant():
    lags_s = np.arange(1, 8, dtype=float)
    P_true = 5.0
    C = np.exp(-lags_s / P_true)
    assert abs(fit_persistence_time(lags_s, C) - P_true) < 1e-6


def test_fit_persistence_time_nan_without_decay():
    lags_s = np.arange(1, 5, dtype=float)
    assert math.isnan(fit_persistence_time(lags_s, np.ones_like(lags_s)))  # C≡1, no decay


# -------------------------------------------------------------------------- MSD
def test_ensemble_msd_ballistic_alpha_two():
    # A long straight, constant-velocity track is purely ballistic: MSD ∝ τ².
    traj = _straight_track(n=21, step=(0.7, 0.0))
    table = ensemble_msd([traj], time_interval_s=1.0)
    fit = fit_msd_power_law(table["lag_s"], table["msd_um2"])
    assert abs(fit.alpha - 2.0) < 1e-6
    assert fit.r2 > 0.999


def test_fit_msd_power_law_recovers_D_and_alpha():
    # Construct MSD = 2·d·D·τ^α with d=2, D=0.25, α=1 -> MSD = τ.
    lag_s = np.array([1.0, 2.0, 4.0, 8.0])
    msd = lag_s.copy()
    fit = fit_msd_power_law(lag_s, msd)
    assert abs(fit.alpha - 1.0) < 1e-9
    assert abs(fit.D_um2_per_s - 0.25) < 1e-9


# ------------------------------------------------------------------- collective
def test_order_parameter_aligned_vs_opposed():
    aligned = np.array([(1.0, 0.0), (2.0, 0.0), (0.5, 0.0)])
    assert abs(_order_parameter(aligned) - 1.0) < 1e-9
    opposed = np.array([(1.0, 0.0), (-1.0, 0.0)])
    assert abs(_order_parameter(opposed)) < 1e-9


def test_one_over_e_length_interpolates():
    centers = np.array([0.5, 1.5, 2.5])
    C = np.array([1.0, math.exp(-1.0) - 0.05, 0.1])  # crosses 1/e in the first bin gap
    xi = _one_over_e_length(centers, C)
    assert 0.5 < xi <= 1.5


def test_one_over_e_length_nan_when_no_decay():
    centers = np.array([0.5, 1.5, 2.5])
    assert math.isnan(_one_over_e_length(centers, np.array([1.0, 0.9, 0.8])))


def test_collective_tables_uniform_drift_is_aligned():
    # 9 cells on a grid, all moving identically (pure drift, no fluctuation):
    # order parameter = 1, and with zero fluctuation the correlation length is NaN.
    xs, ys = np.meshgrid(np.arange(3.0), np.arange(3.0))
    n = xs.size
    inst = {
        "frame": np.zeros(n, dtype=np.int64),
        "cell_id": np.arange(n, dtype=np.int64),
        "x_um": xs.ravel(),
        "y_um": ys.ravel(),
        "vx_um_per_s": np.ones(n),
        "vy_um_per_s": np.zeros(n),
    }
    collective, _ = collective_tables(inst, min_cells=3)
    assert collective["frame"].tolist() == [0]
    np.testing.assert_allclose(collective["order_param"], [1.0])
    assert math.isnan(collective["corr_length_um"][0])
    np.testing.assert_allclose(collective["nn_distance_um"], [1.0])
