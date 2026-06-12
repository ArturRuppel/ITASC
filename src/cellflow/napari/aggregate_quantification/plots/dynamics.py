"""Dynamics family plots — four views per dynamics product.

The old Track Dynamics *plugin* carried a scope dropdown (cell / nucleus) and a
view dropdown (per-frame / per-track / per-tissue / curves). Under the
producer/consumer model the scope dropdown dissolves into two products
(``cell_dynamics`` / ``nucleus_dynamics``) and each view becomes its own
:class:`~cellflow.napari.aggregate_quantification.plots.Plot`, all listing under
the **Dynamics** family and lit only when their product is built:

* **per-frame** / **per-track** / **per-tissue** are statistical
  :class:`~cellflow.napari.aggregate_quantification.plots._pool_plot.PoolPlot`\\s
  over the same product, differing only in which table they pool and which group
  axes they offer;
* **curves** is bespoke — it builds the
  :class:`~cellflow.napari.aggregate_quantification.dynamics_curves_panel.DynamicsCurvesPanel`
  (MSD / DAC / velocity correlation) rather than a ``PlotPanel``.

Each view is defined once as an abstract base (no ``plot_id``) and bound to a
product by a thin cell / nucleus subclass.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cellflow.aggregate_quantification.dynamics import read_track_dynamics
from cellflow.aggregate_quantification.quantifier import Quantifier
from cellflow.napari.aggregate_quantification.plots import Plot, PlotContext, PlotParams
from cellflow.napari.aggregate_quantification.plots._pool_plot import PoolPlot
from cellflow.napari.aggregate_quantification.plots._pooling import (
    METADATA_GROUPS,
    CLASS_COLUMN,
    iter_built,
    position_metadata,
)

_FAMILY = "Dynamics"
_CELL_FIELD = "cell_tracked_labels_path"
_NUCLEUS_FIELD = "nucleus_tracked_labels_path"

#: Per-frame (instantaneous) values + group axes (carries a frame axis).
_FRAME_VALUES = ("speed_um_per_s", "vx_um_per_s", "vy_um_per_s", "net_disp_um")
_FRAME_GROUPS = (*METADATA_GROUPS, CLASS_COLUMN, "frame")
#: Per-track (summary) values + group axes (no frame axis). ``msd_*`` are the
#: per-track MSD fit (``msd_r2`` rides in the table but is not a plotting value).
_TRACK_VALUES = (
    "curvilinear_speed_um_per_s",
    "net_speed_um_per_s",
    "directionality_ratio",
    "persistence_time_s",
    "path_length_um",
    "net_displacement_um",
    "duration_s",
    "msd_D_um2_per_s",
    "msd_alpha",
)
_TRACK_GROUPS = (*METADATA_GROUPS, CLASS_COLUMN)
#: Per-tissue (one row per position) ensemble scalars + group axes. Position-level,
#: so no ``class_label`` / ``frame`` axis.
_TISSUE_VALUES = (
    "msd_D_um2_per_s",
    "msd_alpha",
    "persistence_time_s",
    "corr_length_um",
    "order_param",
)
_TISSUE_GROUPS = METADATA_GROUPS


# --------------------------------------------------------------------- per-frame
class _FrameDynamicsPlot(PoolPlot):
    """Per-frame instantaneous motion distributions."""

    family = _FAMILY
    value_columns = _FRAME_VALUES
    group_columns = _FRAME_GROUPS


class CellFrameDynamicsPlot(_FrameDynamicsPlot):
    plot_id = "cell_dynamics_frame"
    display_name = "Cell · per-frame motion"
    consumes = ("cell_dynamics",)
    label_field = _CELL_FIELD


class NucleusFrameDynamicsPlot(_FrameDynamicsPlot):
    plot_id = "nucleus_dynamics_frame"
    display_name = "Nucleus · per-frame motion"
    consumes = ("nucleus_dynamics",)
    label_field = _NUCLEUS_FIELD


# --------------------------------------------------------------------- per-track
class _TrackDynamicsPlot(PoolPlot):
    """Per-track summary distributions (one row per track)."""

    family = _FAMILY
    value_columns = _TRACK_VALUES
    group_columns = _TRACK_GROUPS

    def _read_table(self, quantifier: Quantifier, path: Path) -> Any:
        return read_track_dynamics(path).tracks


class CellTrackDynamicsPlot(_TrackDynamicsPlot):
    plot_id = "cell_dynamics_track"
    display_name = "Cell · per-track summary"
    consumes = ("cell_dynamics",)
    label_field = _CELL_FIELD


class NucleusTrackDynamicsPlot(_TrackDynamicsPlot):
    plot_id = "nucleus_dynamics_track"
    display_name = "Nucleus · per-track summary"
    consumes = ("nucleus_dynamics",)
    label_field = _NUCLEUS_FIELD


# -------------------------------------------------------------------- per-tissue
class _TissueDynamicsPlot(PoolPlot):
    """Per-tissue ensemble scalars — one row per built position, no per-cell join."""

    family = _FAMILY
    value_columns = _TISSUE_VALUES
    group_columns = _TISSUE_GROUPS
    join_class = False

    def pool(self, records: list[dict]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for record, path in iter_built(self.quantity_id, records):
            dyn = read_track_dynamics(path)
            rows.append(
                {
                    **position_metadata(record),
                    "msd_D_um2_per_s": dyn.msd_D_um2_per_s,
                    "msd_alpha": dyn.msd_alpha,
                    "persistence_time_s": dyn.dac_persistence_time_s,
                    "corr_length_um": dyn.corr_length_um,
                    "order_param": _nan_safe_median(dyn.collective["order_param"]),
                }
            )
        return pd.DataFrame(rows)


class CellTissueDynamicsPlot(_TissueDynamicsPlot):
    plot_id = "cell_dynamics_tissue"
    display_name = "Cell · per-tissue ensemble"
    consumes = ("cell_dynamics",)


class NucleusTissueDynamicsPlot(_TissueDynamicsPlot):
    plot_id = "nucleus_dynamics_tissue"
    display_name = "Nucleus · per-tissue ensemble"
    consumes = ("nucleus_dynamics",)


# ------------------------------------------------------------------------ curves
class _CurvesDynamicsPlot(Plot):
    """Bespoke MSD / DAC / velocity-correlation curves (not a ``PlotPanel``)."""

    family = _FAMILY

    def prepare(self, records: list[dict], params: PlotParams = PlotParams()) -> list:
        from cellflow.napari.aggregate_quantification.dynamics_curves_panel import CurveSet

        curves: list[CurveSet] = []
        for record, path in iter_built(self.consumes[0], records):
            dyn = read_track_dynamics(path)
            curves.append(
                CurveSet(
                    group=str(record.get("condition", "") or record.get("id", "")),
                    msd_lag_s=dyn.msd["lag_s"],
                    msd_um2=dyn.msd["msd_um2"],
                    msd_D_um2_per_s=dyn.msd_D_um2_per_s,
                    msd_alpha=dyn.msd_alpha,
                    dac_lag_s=dyn.dac["lag_s"],
                    dac=dyn.dac["dac"],
                    dac_persistence_time_s=dyn.dac_persistence_time_s,
                    corr_separation_um=dyn.corr_curve.get("separation_um", np.asarray([])),
                    corr=dyn.corr_curve.get("corr", np.asarray([])),
                )
            )
        return curves

    def create_panel(self, ctx: PlotContext, prepared: list | None = None) -> Any:
        from cellflow.napari.aggregate_quantification.dynamics_curves_panel import (
            DynamicsCurvesPanel,
        )

        curves = self.prepare(ctx.records) if prepared is None else prepared
        return DynamicsCurvesPanel(curves)


class CellCurvesDynamicsPlot(_CurvesDynamicsPlot):
    plot_id = "cell_dynamics_curves"
    display_name = "Cell · curves (MSD / DAC / C(r))"
    consumes = ("cell_dynamics",)


class NucleusCurvesDynamicsPlot(_CurvesDynamicsPlot):
    plot_id = "nucleus_dynamics_curves"
    display_name = "Nucleus · curves (MSD / DAC / C(r))"
    consumes = ("nucleus_dynamics",)


def _nan_safe_median(values) -> float:
    """Median over the finite entries; NaN when none are finite (avoids the
    all-NaN ``np.nanmedian`` warning)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")
