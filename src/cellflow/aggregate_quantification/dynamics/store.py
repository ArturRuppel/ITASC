"""Persist / read the track-dynamics artifact and orchestrate the build.

One **HDF5** per substrate (``cell_dynamics.h5`` / ``nucleus_dynamics.h5``) — the
quantity is several heterogeneous tables, so a single flat CSV (as the shape
family uses) does not fit; this mirrors how contacts stores ``cells/`` /
``edges/`` / ``t1_events/``. Groups:

``instantaneous/table``  per ``(frame, cell_id)`` positions + velocities (the
                         ``object_table`` for the generic plotting layer)
``tracks/table``         per-track motility summary
``msd/table``            ensemble MSD curve (+ ``D_um2_per_s`` / ``alpha`` / ``r2`` attrs)
``dac/table``            ensemble directional autocorrelation (+ ``persistence_time_s`` attr)
``collective/table``     per-frame order parameter / correlation length / NN distance
``corr_curve/table``     pooled velocity-correlation curve
``provenance``           run metadata as group attrs

:func:`build_track_dynamics` runs the headless core and writes the file;
:func:`read_track_dynamics` loads it back; :func:`read_instantaneous_table`
serves the tidy-table contract. Backend-only (no Qt / napari).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .collective import collective_tables
from .kinematics import ensemble_dac, instantaneous_table, track_summary_table
from .msd import ensemble_msd, fit_msd_power_law
from .trajectories import extract_trajectories

#: Build params and their defaults (a position with no override uses these).
DEFAULT_PARAMS = {
    "min_track_frames": 3,   # drop shorter tracks from the per-track summary
    "min_msd_samples": 10,   # lags with fewer samples are excluded from the fit
    "corr_bin_um": None,     # None → global median nearest-neighbour distance
    "min_corr_cells": 5,     # frames with fewer velocity-bearing cells → NaN
}


@dataclass(frozen=True)
class TrackDynamics:
    """In-memory view of a dynamics ``.h5`` — every table plus the fit scalars."""

    instantaneous: dict[str, np.ndarray]
    tracks: dict[str, np.ndarray]
    msd: dict[str, np.ndarray]
    dac: dict[str, np.ndarray]
    collective: dict[str, np.ndarray]
    corr_curve: dict[str, np.ndarray]
    msd_D_um2_per_s: float
    msd_alpha: float
    msd_r2: float
    dac_persistence_time_s: float


def build_track_dynamics(
    label_path: str | Path,
    output_path: str | Path,
    *,
    pixel_size_um: float,
    time_interval_s: float,
    source_path: str | Path | None = None,
    params: dict | None = None,
    quantity_id: str = "",
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Compute every dynamics table from a tracked label TIFF and write the ``.h5``."""
    pixel_size_um = float(pixel_size_um)
    time_interval_s = float(time_interval_s)
    if not pixel_size_um > 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um!r}")
    if not time_interval_s > 0:
        raise ValueError(f"time_interval_s must be positive, got {time_interval_s!r}")
    p = {**DEFAULT_PARAMS, **(params or {})}
    label_path = Path(label_path)
    output_path = Path(output_path)

    def step(done: int, total: int, message: str) -> None:
        if progress_cb is not None:
            progress_cb(done, total, message)

    trajectories = extract_trajectories(label_path, pixel_size_um=pixel_size_um)
    step(1, 5, "trajectories")

    instantaneous = instantaneous_table(trajectories, time_interval_s=time_interval_s)
    tracks = track_summary_table(
        trajectories,
        time_interval_s=time_interval_s,
        min_track_frames=int(p["min_track_frames"]),
    )
    step(2, 5, "kinematics")

    msd = ensemble_msd(trajectories, time_interval_s=time_interval_s)
    msd_fit = fit_msd_power_law(
        msd["lag_s"], msd["msd_um2"],
        min_samples_mask=msd["n_samples"] >= int(p["min_msd_samples"]),
    )
    step(3, 5, "msd")

    dac, dac_persistence = ensemble_dac(trajectories, time_interval_s=time_interval_s)
    step(4, 5, "persistence")

    collective, corr_curve = collective_tables(
        instantaneous,
        corr_bin_um=p["corr_bin_um"],
        min_cells=int(p["min_corr_cells"]),
    )
    step(5, 5, "collective")

    _write_h5(
        output_path,
        tables={
            "instantaneous": instantaneous,
            "tracks": tracks,
            "msd": msd,
            "dac": dac,
            "collective": collective,
            "corr_curve": corr_curve,
        },
        msd_fit=msd_fit,
        dac_persistence_time_s=dac_persistence,
        provenance={
            "quantity_id": quantity_id,
            "source_position_path": str(source_path) if source_path else "",
            "label_path": str(label_path),
            "pixel_size_um": pixel_size_um,
            "time_interval_s": time_interval_s,
            "params": {k: ("" if v is None else v) for k, v in p.items()},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cellflow_version": _cellflow_version(),
        },
    )
    return output_path


def read_track_dynamics(path: str | Path) -> TrackDynamics:
    """Load a dynamics ``.h5`` into a :class:`TrackDynamics`."""
    path = Path(path)
    with h5py.File(path, "r") as h5:
        tables = {name: _read_table(h5[f"{name}/table"]) for name in (
            "instantaneous", "tracks", "msd", "dac", "collective", "corr_curve"
        )}
        msd_attrs = h5["msd/table"].attrs
        dac_attrs = h5["dac/table"].attrs
        return TrackDynamics(
            instantaneous=tables["instantaneous"],
            tracks=tables["tracks"],
            msd=tables["msd"],
            dac=tables["dac"],
            collective=tables["collective"],
            corr_curve=tables["corr_curve"],
            msd_D_um2_per_s=float(msd_attrs["D_um2_per_s"]),
            msd_alpha=float(msd_attrs["alpha"]),
            msd_r2=float(msd_attrs["r2"]),
            dac_persistence_time_s=float(dac_attrs["persistence_time_s"]),
        )


def read_instantaneous_table(path: str | Path) -> dict[str, np.ndarray]:
    """The ``(frame, cell_id, …)`` instantaneous table — the tidy-table contract."""
    path = Path(path)
    with h5py.File(path, "r") as h5:
        return _read_table(h5["instantaneous/table"])


# --------------------------------------------------------------------- h5 I/O
def _write_h5(output_path, *, tables, msd_fit, dac_persistence_time_s, provenance) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        for name, table in tables.items():
            group = h5.create_group(f"{name}/table")
            for column, values in table.items():
                group.create_dataset(column, data=np.asarray(values))
        msd_table = h5["msd/table"]
        msd_table.attrs["D_um2_per_s"] = float(msd_fit.D_um2_per_s)
        msd_table.attrs["alpha"] = float(msd_fit.alpha)
        msd_table.attrs["r2"] = float(msd_fit.r2)
        h5["dac/table"].attrs["persistence_time_s"] = float(dac_persistence_time_s)
        prov = h5.create_group("provenance")
        _write_provenance_attrs(prov, provenance)


def _write_provenance_attrs(group: h5py.Group, provenance: dict) -> None:
    for key, value in provenance.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                group.attrs[f"{key}.{sub_key}"] = sub_value
        else:
            group.attrs[key] = value


def _read_table(group: h5py.Group) -> dict[str, np.ndarray]:
    return {name: dataset[:] for name, dataset in group.items()}


def _cellflow_version() -> str:
    try:
        from importlib.metadata import version

        return version("cellflow")
    except Exception:
        return "unknown"
