"""Plotting backend for Aggregate Quantification — the layer that turns tidy
per-object tables into pooled DataFrames, aggregated summaries, figures, and CSV.

Quantity-agnostic and **headless** (no Qt / napari): it consumes tables that have
already been read via :meth:`Quantifier.object_table`, so it never resolves
catalogue paths itself and runs unchanged in scripts, notebooks, and the
standalone wheel. The Cell Shape group plugin is the first UI consumer; contacts
and NLS can reuse it as-is.

Pipeline::

    PositionSource(metadata, table[, join_table, join_columns])  one per position
        │  pool_object_tables(...)
        ▼
    pooled DataFrame   (condition · date · position_id · frame · cell_id · descriptors [· class_label])
        │  aggregate(df, spec)              build_figure(df, spec)        write_csv(df, path)
        ▼                                   ▼                             ▼
    grouped summary DataFrame           matplotlib Figure              CSV file
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

__all__ = [
    "PlotSpec",
    "StyleSpec",
    "PickPoint",
    "PositionSource",
    "pool_object_tables",
    "aggregate",
    "build_figure",
    "pickable_points",
    "write_csv",
    "potential_landscape",
    "effective_barrier",
    "potential_table",
    "DISTRIBUTION_PLOTS",
    "CURVE_PLOTS",
]

#: The tidy keys every per-object table carries; joins and per-position grouping
#: are keyed on these.
KEY_COLUMNS = ("frame", "cell_id")
#: Bucket label for cells with no classification (no contacts artifact, or the
#: position was never classified).
UNCLASSIFIED = "unclassified"

#: The independent unit a comparison aggregates over, coarse→fine. ``cell`` now
#: means the cell *track* (one value per cell, frames collapsed), not the cell
#: *frame*; ``position`` and ``date`` climb to the field-of-view and the
#: biological replicate.
_LEVELS = ("cell", "position", "date")
#: Biological nesting that sits below the comparison groups, coarse→fine. The
#: per-frame ``frame`` axis is always collapsed first (it is never a unit key);
#: ``level`` then chooses how far down this chain the independent unit sits, and
#: the reduction equal-weights each parent's children at every step (a long-lived
#: cell counts once, a crowded position counts once). A column absent from the
#: pooled table is simply skipped.
_NESTING = ("date", "position_id", "cell_id")
#: level -> the nesting column whose distinct values are the independent unit.
_LEVEL_ENTITY = {"cell": "cell_id", "position": "position_id", "date": "date"}
#: Distribution-family plots — rendered through seaborn, with the group-by /
#: ``class_label`` model mapped onto its ``hue=``. ``strip``/``swarm`` are the
#: new per-cell scatter members.
DISTRIBUTION_PLOTS = ("hist", "box", "violin", "strip", "swarm")
#: Curve-family plots — a distribution Boltzmann-inverted into a ``U(x) = −ln P``
#: curve. Unlike the distribution family these pool **raw** samples (no
#: ``reduce_to_units``); see :func:`_plot_potential`.
CURVE_PLOTS = ("potential",)
_PLOTS = (*DISTRIBUTION_PLOTS, "bar", "line", *CURVE_PLOTS)
_STATS = ("mean", "median", "count")
_ERRORS = ("sd", "sem", "none")


@dataclass(frozen=True)
class PositionSource:
    """One position's contribution to a pooled table.

    ``metadata`` (condition, date, position_id, …) is broadcast onto every row.
    ``table`` is the quantity's :meth:`Quantifier.object_table`. ``join_table``
    (optional) supplies ``join_columns`` to left-join *within this position* (so
    cell ids never collide across positions) on whichever key columns it carries:
    a per-frame source joins on ``(frame, cell_id)``; a per-track source — e.g.
    the NLS sidecar CSV's ``{cell_id, class_label}`` — joins on ``cell_id`` alone
    and broadcasts its label across every frame of that cell.
    """

    metadata: Mapping[str, Any]
    table: Mapping[str, np.ndarray]
    join_table: Mapping[str, np.ndarray] | None = None
    join_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlotSpec:
    """What to plot / aggregate over a pooled table."""

    value: str
    group_by: tuple[str, ...] = ()
    # The independent unit: "cell" (per track) | "position" | "date" (replicate).
    # Frames are always collapsed to one value per track first, so no level ever
    # treats a frame as an independent datapoint.
    level: str = "cell"
    plot: str = "hist"  # hist | box | violin | bar | line
    stat: str = "mean"  # mean | median | count
    error: str = "sd"  # sd | sem | none
    bins: int = 30

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("level", self.level, _LEVELS),
            ("plot", self.plot, _PLOTS),
            ("stat", self.stat, _STATS),
            ("error", self.error, _ERRORS),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


@dataclass(frozen=True)
class StyleSpec:
    """Presentation knobs threaded through :func:`build_figure`.

    Every field defaults to today's look, so ``build_figure(df, spec)`` (no
    ``style_spec``) reproduces the previous output; touching a field is the only
    way to change it. The dataclass is plain data — Qt-free and unit-testable —
    so the panel builds it from its styling controls and hands it down.
    """

    #: Named qualitative palette applied across groups (any seaborn/matplotlib
    #: palette name, e.g. ``tab10``, ``Set2``, ``colorblind``).
    palette: str = "tab10"
    #: Optional text overrides; blank keeps the auto label.
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    #: Matplotlib style sheet (``default``, ``ggplot``, ``seaborn-v0_8`` …).
    style: str = "default"
    #: Figure dimensions in inches.
    width: float = 6.0
    height: float = 4.0
    grid: bool = False
    legend: bool = True
    legend_loc: str = "best"
    #: Base font size; titles/labels/ticks scale off it.
    font_size: float = 10.0
    #: Optional axis limits; ``None`` keeps matplotlib's autoscaled bound. Each
    #: side is independent, so e.g. only ``ymin`` may be pinned.
    xmin: float | None = None
    xmax: float | None = None
    ymin: float | None = None
    ymax: float | None = None

    # -- Box-plot knobs (ignored by every other plot type) -------------------
    #: Whisker reach as a multiple of the IQR (matplotlib/seaborn ``whis``):
    #: 1.5 is the Tukey default; larger values push the whiskers toward the
    #: full data range, leaving fewer points flagged as outliers.
    box_whis: float = 1.5
    #: Draw points beyond the whiskers as individual outlier markers.
    box_showfliers: bool = True
    #: Notch the box around the median to show its ~95% confidence interval.
    box_notch: bool = False


def pool_object_tables(sources: Iterable[PositionSource]) -> pd.DataFrame:
    """Concatenate per-position tables into one annotated, tidy DataFrame.

    Each source's table becomes a frame with its metadata columns prepended; an
    optional per-position left-join attaches classification columns. Returns an
    empty DataFrame when there are no sources.
    """
    frames: list[pd.DataFrame] = []
    join_columns: list[str] = []
    for source in sources:
        frame = pd.DataFrame({k: np.asarray(v) for k, v in source.table.items()})
        if source.join_columns:
            join_columns.extend(source.join_columns)
            frame = _join_position(frame, source.join_table or {}, source.join_columns)
        for key, value in source.metadata.items():
            frame.insert(0, key, value)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    pooled = pd.concat(frames, ignore_index=True)
    # A column declared by only some positions is NaN for the others after the
    # concat; treat those (and any blank label) as the unclassified bucket.
    for column in dict.fromkeys(join_columns):
        if column not in pooled.columns:
            pooled[column] = UNCLASSIFIED
        else:
            pooled[column] = pooled[column].replace("", UNCLASSIFIED).fillna(UNCLASSIFIED)
    return pooled


def _join_position(
    frame: pd.DataFrame,
    join_table: Mapping[str, np.ndarray],
    join_columns: tuple[str, ...],
) -> pd.DataFrame:
    join_df = pd.DataFrame({k: np.asarray(v) for k, v in join_table.items()})
    # Join on whichever key columns the join table carries — but it must at least
    # key on ``cell_id``. A per-frame source keys on ``(frame, cell_id)``; a
    # per-track source (e.g. the NLS sidecar CSV, one row per cell) keys on
    # ``cell_id`` alone and broadcasts its label across every frame of that cell.
    keys = [key for key in KEY_COLUMNS if key in join_df.columns]
    # A position without a usable join source (no artifact / CSV) yields no
    # ``cell_id`` key; leave its rows unjoined (filled to UNCLASSIFIED after the
    # concat).
    if "cell_id" not in keys:
        return frame
    keep = [c for c in (*keys, *join_columns) if c in join_df.columns]
    join_df = join_df[keep].drop_duplicates(subset=keys)
    return frame.merge(join_df, on=keys, how="left")


def aggregate(df: pd.DataFrame, spec: PlotSpec) -> pd.DataFrame:
    """Tidy summary keyed by ``spec.group_by`` with ``n``, ``value``, ``error``.

    The *unit* of aggregation is the cell track (``level="cell"``), the position
    (``level="position"``), or the biological replicate (``level="date"``).
    Whatever the level, frames are first collapsed to one value per track — no
    level ever treats a frame as an independent datapoint — and each step up the
    nesting (track→position→date) equal-weights its children, so a comparison
    aggregates the chosen unit and not its (correlated) sub-samples. ``value``
    and ``error`` are the central tendency and spread *across the units* at
    ``level``; ``n`` is how many of them there are.

    The within-unit collapse follows ``spec.stat`` end to end: ``mean`` averages,
    ``median`` takes medians, at every level. ``count`` is special — it counts
    *cells* (distinct tracks), never cell-frames: at cell level a pooled tally
    per group (no spread); at position/date level the per-unit cell count becomes
    a datapoint and the cohort is summarised across units (e.g. cells/position).

    Columns: the group keys, ``n`` (units aggregated), ``value`` (the central
    number), ``error`` (sd/sem across units; NaN/0 when undefined).
    """
    group = list(spec.group_by)
    if df.empty:
        return pd.DataFrame(columns=[*group, "n", "value", "error"])

    if spec.stat == "count":
        return _aggregate_count(df, spec)

    units = reduce_to_units(df, spec)
    central = "median" if spec.stat == "median" else "mean"
    rows: list[dict[str, Any]] = []
    grouped = units.groupby(group, dropna=False) if group else [(None, units)]
    for key, chunk in grouped:
        values = pd.to_numeric(chunk[spec.value], errors="coerce").dropna()
        rows.append({
            **_group_key_to_dict(group, key),
            "n": int(len(values)),
            "value": float(getattr(values, central)()) if len(values) else float("nan"),
            "error": _spread(values, spec.error),
        })
    return pd.DataFrame(rows, columns=[*group, "n", "value", "error"])


def _nesting_keys(df: pd.DataFrame) -> list[str]:
    """The :data:`_NESTING` columns actually present in *df*, coarse→fine."""
    return [k for k in _NESTING if k in df.columns]


def _unit_keys(df: pd.DataFrame, spec: PlotSpec) -> list[str]:
    """Nesting columns retained at ``spec.level`` — the prefix of the present
    nesting down to (and including) the level's entity. When that entity column
    is missing, fall back to the finest nesting available."""
    present = _nesting_keys(df)
    target = _LEVEL_ENTITY[spec.level]
    if target in present:
        return present[: present.index(target) + 1]
    return present


def reduce_to_units(df: pd.DataFrame, spec: PlotSpec) -> pd.DataFrame:
    """Collapse the pooled per-frame table to one row per independent unit.

    Reduces frames → track → … → ``spec.level`` along :data:`_NESTING`, applying
    ``spec.stat`` (``median``, else ``mean``) at every collapse so each parent
    equal-weights its children. Returns the group keys, the retained nesting
    keys, and ``spec.value`` — one row per unit. When the table carries none of
    the nesting columns (no cell_id/position_id/date), each row is already its
    own unit and the frame is returned unchanged.
    """
    present = _nesting_keys(df)
    if not present:
        return df
    group = list(spec.group_by)
    unit = _unit_keys(df, spec)
    agg = "median" if spec.stat == "median" else "mean"
    work = df
    levels = present
    # First pass collapses the frame axis (``frame`` is never a key, so grouping
    # on group+nesting averages a track's frames to one value); each later pass
    # drops the finest nesting key and climbs one level toward ``unit``.
    while True:
        work = work.groupby(group + levels, dropna=False)[spec.value].agg(agg).reset_index()
        if len(levels) <= len(unit):
            return work
        levels = levels[:-1]


def _aggregate_count(df: pd.DataFrame, spec: PlotSpec) -> pd.DataFrame:
    """Count distinct cells (tracks), never cell-frames. See :func:`aggregate`."""
    group = list(spec.group_by)
    present = _nesting_keys(df)
    # One row per distinct cell within each group; without nesting keys every row
    # already stands for a distinct unit.
    cells = df[group + present].drop_duplicates() if present else df

    if spec.level == "cell" or not present:
        # Pooled distinct-cell tally per group; no spread.
        counts = (
            cells.groupby(group, dropna=False).size().reset_index(name="value")
            if group
            else pd.DataFrame([{"value": len(cells)}])
        )
        counts["n"] = counts["value"].astype(int)
        counts["error"] = float("nan")
        return counts[[*group, "n", "value", "error"]]

    # Per-unit cell counts, then summarise across units (mean ± spread).
    unit = _unit_keys(df, spec)
    per_unit = cells.groupby(group + unit, dropna=False).size().reset_index(name="_count")
    rows: list[dict[str, Any]] = []
    grouped = per_unit.groupby(group, dropna=False) if group else [(None, per_unit)]
    for key, chunk in grouped:
        values = chunk["_count"].astype(float)
        rows.append({
            **_group_key_to_dict(group, key),
            "n": int(len(values)),
            "value": float(values.mean()) if len(values) else float("nan"),
            "error": _spread(values, spec.error),
        })
    return pd.DataFrame(rows, columns=[*group, "n", "value", "error"])


def _spread(values: pd.Series, error: str) -> float:
    if error == "none" or len(values) < 2:
        return 0.0 if error == "none" else float("nan")
    sd = float(values.std(ddof=1))
    return sd if error == "sd" else sd / float(np.sqrt(len(values)))


def _group_key_to_dict(group: list[str], key: Any) -> dict[str, Any]:
    if not group:
        return {}
    # pandas yields a 1-tuple for a single-column groupby key; normalize both.
    if not isinstance(key, tuple):
        key = (key,)
    return dict(zip(group, key))


def potential_landscape(
    values: np.ndarray,
    *,
    bins: int,
    value_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boltzmann-invert a 1-D sample into an effective potential ``U = −ln P``.

    Bins *values* into ``P(x) = counts / N`` and returns ``(centers, U, counts)``
    over the **occupied** bins only — an empty bin has ``P = 0`` ⇒ ``U = ∞`` and
    is dropped (the reference's ``probabilities > 0`` mask). ``U`` is in units of
    kT: with the natural log, ``P ∝ exp(−U/kT)`` ⇒ ``U/kT = −ln P + const``.
    *value_range* pins the histogram extent so several groups share one binning;
    ``None`` uses the sample's own min/max. Returns three empty arrays when no
    finite sample survives.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        empty = np.empty(0, dtype=float)
        return empty, empty, np.empty(0, dtype=np.int64)
    counts, edges = np.histogram(finite, bins=bins, range=value_range)
    centers = (edges[:-1] + edges[1:]) / 2.0
    occupied = counts > 0
    u = -np.log(counts[occupied] / finite.size)
    return centers[occupied], u, counts[occupied]


def effective_barrier(centers: np.ndarray, u: np.ndarray) -> float:
    """``ΔE_eff`` [kT] = ``U`` at the bin nearest ``x = 0`` minus ``min(U)``.

    Operates on the occupied ``(centers, U)`` from :func:`potential_landscape`.
    The transition state is ``x = 0`` (a junction length → 0, the four-fold
    vertex); the well is the curve minimum (the most-probable value). Returns NaN
    when there are fewer than two occupied bins or ``0`` is not bracketed by the
    occupied range — i.e. the data never reached the transition state.
    """
    centers = np.asarray(centers, dtype=float)
    u = np.asarray(u, dtype=float)
    if u.size < 2 or centers.min() > 0.0 or centers.max() < 0.0:
        return float("nan")
    transition = float(u[int(np.argmin(np.abs(centers)))])
    return transition - float(u.min())


def _shared_range(df: pd.DataFrame, spec: PlotSpec) -> tuple[float, float] | None:
    """Common ``(min, max)`` over ``spec.value`` so every group bins identically;
    ``None`` (let numpy pick per call) when there is no finite spread."""
    values = pd.to_numeric(df[spec.value], errors="coerce").to_numpy()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    lo, hi = float(values.min()), float(values.max())
    return (lo, hi) if lo < hi else None


def potential_table(df: pd.DataFrame, spec: PlotSpec) -> pd.DataFrame:
    """The plotted potential curve(s) as a tidy table, for CSV export.

    One block of rows per group (``group`` is the ``" · "``-joined group-by, or
    ``"all"`` when none): ``group, center, U, counts, delta_e_eff``. The barrier
    ``delta_e_eff`` is repeated down a group's rows and matches the per-curve
    legend annotation in :func:`build_figure`. Empty when *df* lacks
    ``spec.value`` or has no finite samples.
    """
    columns = ["group", "center", "U", "counts", "delta_e_eff"]
    if df.empty or spec.value not in df.columns:
        return pd.DataFrame(columns=columns)
    value_range = _shared_range(df, spec)
    blocks: list[pd.DataFrame] = []
    for label, chunk in _group_series(df, list(spec.group_by)):
        values = pd.to_numeric(chunk[spec.value], errors="coerce").to_numpy()
        centers, u, counts = potential_landscape(values, bins=spec.bins, value_range=value_range)
        if centers.size == 0:
            continue
        blocks.append(pd.DataFrame({
            "group": label,
            "center": centers,
            "U": u,
            "counts": counts,
            "delta_e_eff": effective_barrier(centers, u),
        }))
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(columns=columns)


def build_figure(
    df: pd.DataFrame, spec: PlotSpec, style_spec: StyleSpec | None = None
) -> Figure:
    """Render *spec* over *df* with *style_spec*; return a `Figure` with an Agg
    canvas attached so ``fig.savefig(...)`` works headlessly (the Qt frontend
    reattaches its own).

    Distribution-family plots (``DISTRIBUTION_PLOTS``) route through seaborn,
    whose ``hue=`` maps onto the group-by / ``class_label`` model — but over the
    per-unit table from :func:`reduce_to_units`, not raw frames, so they carry the
    same pseudoreplication guard as the others. ``bar``/``line`` stay custom
    matplotlib driven by :func:`aggregate`."""
    style_spec = style_spec or StyleSpec()
    with mpl.style.context([style_spec.style, _rc_overrides(style_spec)]):
        fig = Figure(figsize=(style_spec.width, style_spec.height), tight_layout=True)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        # A count bar/line tallies tracks and needs no value column; every other
        # path reads ``spec.value``. Guard a column the table doesn't carry (e.g.
        # a stale build missing a newer quantity) with a legible placeholder
        # rather than a KeyError from deep in the aggregation.
        needs_value = not (spec.stat == "count" and spec.plot not in DISTRIBUTION_PLOTS)
        if df.empty or (needs_value and spec.value not in df.columns):
            ax.set_title(style_spec.title or "No data in scope")
            return fig

        if spec.plot in DISTRIBUTION_PLOTS:
            _plot_distribution(ax, df, spec, style_spec)
        elif spec.plot == "potential":
            _plot_potential(ax, df, spec, style_spec)
        elif spec.plot == "bar":
            _plot_bar(ax, df, spec, style_spec)
        else:  # line
            _plot_line(ax, df, spec, style_spec)
        _apply_style(ax, spec, style_spec)
    return fig


def _rc_overrides(style_spec: StyleSpec) -> dict[str, Any]:
    """rcParams derived from the base font size, applied within the style sheet
    context so every text artist created inside picks them up."""
    base = style_spec.font_size
    return {
        "font.size": base,
        "axes.titlesize": base * 1.2,
        "axes.labelsize": base,
        "xtick.labelsize": base * 0.9,
        "ytick.labelsize": base * 0.9,
        "legend.fontsize": base * 0.9,
    }


def _apply_style(ax, spec: PlotSpec, style_spec: StyleSpec) -> None:
    """Apply text overrides, grid, and legend after the plot is drawn. A blank
    text override keeps the auto label (title defaults to the value column)."""
    ax.set_title(style_spec.title or spec.value)
    if style_spec.xlabel:
        ax.set_xlabel(style_spec.xlabel)
    if style_spec.ylabel:
        ax.set_ylabel(style_spec.ylabel)
    ax.grid(style_spec.grid)
    # Axis limits before the legend branch's early return so they always apply.
    if style_spec.xmin is not None or style_spec.xmax is not None:
        ax.set_xlim(left=style_spec.xmin, right=style_spec.xmax)
    if style_spec.ymin is not None or style_spec.ymax is not None:
        ax.set_ylim(bottom=style_spec.ymin, top=style_spec.ymax)

    existing = ax.get_legend()
    if not style_spec.legend:
        if existing is not None:
            existing.remove()
        return
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc=style_spec.legend_loc, fontsize="small")
    elif existing is not None:  # seaborn-managed legend (e.g. histplot)
        existing.set_loc(style_spec.legend_loc)


def _group_label_column(df: pd.DataFrame, group: list[str]) -> tuple[pd.DataFrame, str | None]:
    """Add a single combined group-label column (``" · "``-joined) when there is
    a group-by, so seaborn's single-column ``hue=``/``x=`` can carry a multi-axis
    grouping. Returns the (copied) frame and the hue column name (None when no
    group-by)."""
    if not group:
        return df, None
    data = df.copy()
    data["_group"] = data[group].astype(str).agg(" · ".join, axis=1)
    return data, "_group"


@dataclass(frozen=True)
class PickPoint:
    """One clickable plotted point mapped back to its source row.

    ``category`` is the hue label the point sits under ("" when there is no
    group-by); ``value`` is the plotted measurement; ``row_index`` is the index
    into the DataFrame ``build_figure`` was given.
    """

    category: str
    value: float
    row_index: int


def pickable_points(df: pd.DataFrame, spec: PlotSpec, style_spec: StyleSpec) -> list[PickPoint]:
    """Plotted points mapped back to a representative source row, for click-to-select.

    Each drawn point is one independent unit (per ``spec.level``), matching what
    :func:`_plot_distribution` renders. ``strip``/``swarm`` expose every
    finite-value unit; ``box`` exposes only the Tukey outliers (``whis`` from
    *style_spec*); all other plots expose none. The category string matches the
    x-axis tick label seaborn draws (see :func:`_group_label_column`), so the UI
    can scope a click to one category.

    ``row_index`` is positional (``.iloc``) into the **original pooled** *df*: a
    coarse unit (e.g. a position) resolves to one representative source row, whose
    identity columns load that unit. The pooled tables from
    ``pool_object_tables`` carry a default ``RangeIndex``.
    """
    if spec.plot not in ("strip", "swarm", "box") or df.empty:
        return []
    units = reduce_to_units(df, spec)
    row_for = _representative_row(df, spec)
    data, hue = _group_label_column(units, list(spec.group_by))
    values = pd.to_numeric(data[spec.value], errors="coerce")
    cats = (
        data[hue].astype(str)
        if hue is not None
        else pd.Series([""] * len(data), index=data.index)
    )
    finite = values.notna()
    if spec.plot in ("strip", "swarm"):
        return [
            PickPoint(str(cats[i]), float(values[i]), row_for(data.loc[i]))
            for i in values.index[finite]
        ]
    # box: outliers only, per category (matplotlib's Tukey flier rule).
    out: list[PickPoint] = []
    for cat, members in values[finite].groupby(cats[finite]):
        q1, q3 = members.quantile(0.25), members.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - style_spec.box_whis * iqr, q3 + style_spec.box_whis * iqr
        for i, v in members.items():
            if v < lo or v > hi:
                out.append(PickPoint(str(cat), float(v), row_for(data.loc[i])))
    return out


def _representative_row(df: pd.DataFrame, spec: PlotSpec):
    """A callable mapping a reduced unit row to a positional index into *df*.

    With nesting keys, each unit maps to the first original row carrying its
    (group + unit) key tuple. Without them, the reduction is the identity, so a
    unit row *is* an original row and its own index (positional on a default
    ``RangeIndex``) is used.
    """
    present = _nesting_keys(df)
    if not present:
        return lambda row: int(row.name)
    keys = list(spec.group_by) + _unit_keys(df, spec)
    first = (
        df.reset_index(drop=True).reset_index().groupby(keys, dropna=False)["index"].first()
    )

    def row_for(row: pd.Series) -> int:
        key = tuple(row[k] for k in keys)
        return int(first.loc[key[0] if len(keys) == 1 else key])

    return row_for


def _plot_distribution(ax, df: pd.DataFrame, spec: PlotSpec, style_spec: StyleSpec) -> None:
    group = list(spec.group_by)
    # Distributions show one point per independent unit (per ``spec.level``), not
    # per frame: a histogram of cell areas has one observation per track, so a
    # long-lived cell can't dominate the shape it draws.
    units = reduce_to_units(df, spec)
    data, hue = _group_label_column(units, group)
    data = data.assign(**{spec.value: pd.to_numeric(data[spec.value], errors="coerce")})
    data = data.dropna(subset=[spec.value])
    if data.empty:
        ax.set_title("No data in scope")
        return
    palette = style_spec.palette if hue is not None else None
    if spec.plot == "hist":
        sns.histplot(
            data=data, x=spec.value, hue=hue, bins=spec.bins,
            alpha=0.5, palette=palette, ax=ax,
        )
        ax.set_ylabel("count")
    elif spec.plot == "box":
        sns.boxplot(
            data=data, x=hue, y=spec.value, hue=hue, palette=palette, ax=ax, legend=False,
            whis=style_spec.box_whis, showfliers=style_spec.box_showfliers,
            notch=style_spec.box_notch,
        )
    elif spec.plot == "violin":
        sns.violinplot(
            data=data, x=hue, y=spec.value, hue=hue, palette=palette, ax=ax, legend=False,
        )
    else:  # strip | swarm
        plot = sns.stripplot if spec.plot == "strip" else sns.swarmplot
        plot(data=data, x=hue, y=spec.value, hue=hue, palette=palette, ax=ax, legend=False)


def _plot_potential(ax, df: pd.DataFrame, spec: PlotSpec, style_spec: StyleSpec) -> None:
    """Render the effective potential ``U(x) = −ln P(x)`` curve per group.

    Unlike the distribution plots this pools **raw** samples (no
    :func:`reduce_to_units`): the landscape is the shape of the within-sample
    fluctuation distribution, not a per-unit comparison, so collapsing to one
    value per track would be wrong — and degenerate for a table that carries no
    ``cell_id`` nesting (e.g. the contacts signed-junction-length table). Every
    group shares one binning (``_shared_range``) so curves are comparable, and
    each curve's effective barrier ``ΔE_eff`` rides in its legend label."""
    group = list(spec.group_by)
    value_range = _shared_range(df, spec)
    series_list = list(_group_series(df, group))
    colors = sns.color_palette(style_spec.palette, n_colors=max(len(series_list), 1))
    spans_zero = value_range is not None and value_range[0] <= 0.0 <= value_range[1]
    drew = False
    for color, (label, chunk) in zip(colors, series_list):
        values = pd.to_numeric(chunk[spec.value], errors="coerce").to_numpy()
        centers, u, _ = potential_landscape(values, bins=spec.bins, value_range=value_range)
        if centers.size == 0:
            continue
        barrier = effective_barrier(centers, u)
        barrier_txt = f"{barrier:.2f} kT" if np.isfinite(barrier) else "n/a"
        ax.plot(
            centers, u, marker="o", linestyle="-", markersize=4, color=color,
            label=f"{label} (ΔE_eff={barrier_txt})",
        )
        drew = True
    if not drew:
        ax.set_title("No data in scope")
        return
    if spans_zero:
        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel(spec.value)
    ax.set_ylabel("U = −ln P  [kT]")


def _plot_bar(ax, df: pd.DataFrame, spec: PlotSpec, style_spec: StyleSpec) -> None:
    summary = aggregate(df, spec)
    group = list(spec.group_by)
    labels = [
        " · ".join(str(summary.iloc[i][g]) for g in group) if group else "all"
        for i in range(len(summary))
    ]
    x = np.arange(len(summary))
    errors = summary["error"].fillna(0.0).to_numpy()
    colors = sns.color_palette(style_spec.palette, n_colors=max(len(summary), 1))
    ax.bar(x, summary["value"].to_numpy(), yerr=errors, capsize=4, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(_bar_ylabel(spec))


def _bar_ylabel(spec: PlotSpec) -> str:
    if spec.stat == "count":
        if spec.level == "position":
            return "cells per position"
        if spec.level == "date":
            return "cells per date"
        return "cell count"
    return f"{spec.stat} {spec.value}"


def _plot_line(ax, df: pd.DataFrame, spec: PlotSpec, style_spec: StyleSpec) -> None:
    group = list(spec.group_by)
    series_list = list(_group_series(df, group))
    colors = sns.color_palette(style_spec.palette, n_colors=max(len(series_list), 1))
    for color, (label, chunk) in zip(colors, series_list):
        if spec.stat == "count":
            series = chunk.groupby("frame", dropna=False).size()
        else:
            agg = "mean" if spec.stat == "mean" else "median"
            series = chunk.groupby("frame", dropna=False)[spec.value].agg(agg)
        series = series.sort_index()
        ax.plot(series.index.to_numpy(), series.to_numpy(), marker="o", label=label, color=color)
    ax.set_xlabel("frame")
    ax.set_ylabel("count" if spec.stat == "count" else f"{spec.stat} {spec.value}")


def _group_series(df: pd.DataFrame, group: list[str]):
    """Yield ``(label, sub_df)`` per group combination (one group when none)."""
    if not group:
        yield "all", df
        return
    for key, chunk in df.groupby(group, dropna=False):
        parts = key if isinstance(key, tuple) else (key,)
        yield " · ".join(str(p) for p in parts), chunk


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write *df* to CSV, ensuring a ``.csv`` suffix (Qt's save dialog often omits
    it). Returns the final path."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        path = path.with_name(path.name + ".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
