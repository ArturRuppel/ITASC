"""Nucleus-dynamics quantifier — the registry adapter over the dynamics core.

The nucleus twin of :mod:`.cell_dynamics`: it runs the same label-agnostic
:func:`build_track_dynamics` over the **nucleus** label stack instead of the cell
one, persisting ``aggregate_quantification/nucleus_dynamics.h5``. Nuclei are
compact, point-like centroids — the robust default for motility / MSD. The
object-key column stays ``cell_id`` (a nucleus is nucleus-seeded so it carries
its cell's shared track id), so the pooling / plotting layer treats it no
differently.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from cellflow.contact_analysis.dynamics import (
    TrackDynamics,
    build_track_dynamics,
    read_instantaneous_table,
    read_track_dynamics,
)
from cellflow.contact_analysis.quantifier import PositionInputs, Quantifier


class NucleusDynamicsQuantifier(Quantifier):
    """Quantifies per-nucleus motion (speed, persistence, MSD, collective) from nucleus labels."""

    quantity_id = "nucleus_dynamics"
    display_name = "Nucleus dynamics"
    requires = ("nucleus_labels_path",)
    # Pixel size + frame interval are global build params (see CellDynamics).
    required_build_params = {
        "pixel_size_um": "pixel size (µm/px)",
        "time_interval_s": "frame interval (s)",
    }

    default_output_name = "nucleus_dynamics.h5"
    # Instantaneous nucleus motion is keyed on the shared cell track id
    # (frame, cell_id).
    table_keys = ("frame", "cell_id")

    def build(
        self,
        inputs: PositionInputs,
        output_path: Path,
        *,
        params: dict | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> Path:
        return build_track_dynamics(
            inputs.nucleus_labels_path,
            output_path,
            pixel_size_um=inputs.pixel_size_um,
            time_interval_s=inputs.time_interval_s,
            source_path=inputs.position_dir,
            params=params,
            quantity_id=self.quantity_id,
            progress_cb=progress_cb,
        )

    def read(self, output_path: Path) -> TrackDynamics:
        return read_track_dynamics(output_path)

    def object_table(self, output_path: Path) -> Mapping[str, np.ndarray]:
        return read_instantaneous_table(output_path)
