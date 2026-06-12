"""Cell-dynamics quantifier — the registry adapter over the dynamics core.

Wraps :mod:`cellflow.aggregate_quantification.dynamics` so the studio can build
and read per-cell motion through the generic :class:`Quantifier` interface. Its
persistence is a multi-table ``aggregate_quantification/cell_dynamics.h5``;
:meth:`object_table` exposes the per-frame instantaneous table (the only
``(frame, cell_id)`` table) to the plotting backend. The cell twin of
:mod:`.nucleus_dynamics` — they share the label-agnostic core and differ only by
which label field they read and their output filename.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from cellflow.aggregate_quantification.dynamics import (
    TrackDynamics,
    build_track_dynamics,
    read_instantaneous_table,
    read_track_dynamics,
)
from cellflow.aggregate_quantification.quantifier import PositionInputs, Quantifier


class CellDynamicsQuantifier(Quantifier):
    """Quantifies per-cell motion (speed, persistence, MSD, collective) from cell labels."""

    quantity_id = "cell_dynamics"
    display_name = "Cell dynamics"
    requires = ("cell_labels_path",)
    # Pixel size (µm/px) + frame interval (s/frame) are global build params, set
    # once in the Parameters panel and applied to every position, so each output
    # lands in physical units. They gate the build (unset ⇒ not buildable) and
    # reach :meth:`build` via the stamped ``PositionInputs``, not the params dict.
    required_build_params = {
        "pixel_size_um": "pixel size (µm/px)",
        "time_interval_s": "frame interval (s)",
    }

    default_output_name = "cell_dynamics.h5"

    def build(
        self,
        inputs: PositionInputs,
        output_path: Path,
        *,
        params: dict | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> Path:
        return build_track_dynamics(
            inputs.cell_labels_path,
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
