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
    "PositionSource",
    "pool_object_tables",
    "aggregate",
    "build_figure",
    "write_csv",
    "DISTRIBUTION_PLOTS",
]

#: The tidy keys every per-object table carries; joins and per-position grouping
#: are keyed on these.
KEY_COLUMNS = ("frame", "cell_id")
#: Bucket label for cells with no classification (no contacts artifact, or the
#: position was never classified).
UNCLASSIFIED = "unclassified"

_LEVELS = ("cell", "position")
#: Distribution-family plots — rendered through seaborn, with the group-by /
#: ``class_label`` model mapped onto its ``hue=``. ``strip``/``swarm`` are the
#: new per-cell scatter members.
DISTRIBUTION_PLOTS = ("hist", "box", "violin", "strip", "swarm")
_PLOTS = (*DISTRIBUTION_PLOTS, "bar", "line")
_STATS = ("mean", "median", "count")
_ERRORS = ("sd", "sem", "none")


@dataclass(frozen=True)
class PositionSource:
    """One position's contribution to a pooled table.

    ``metadata`` (condition, date, position_id, …) is broadcast onto every row.
    ``table`` is the quantity's :meth:`Quantifier.object_table`. ``join_table``
    (optional) is another quantity's object_table — e.g. the contacts ``cells``
    table — from which ``join_columns`` are left-joined on ``(frame, cell_id)``
    *within this position*, so cell ids never collide across positions.
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
    level: str = "cell"  # "cell" (pooled) | "position" (per-position summary)
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
    # A position without the join source (no contacts artifact) yields no usable
    # join keys; leave its rows unjoined (filled to UNCLASSIFIED after the concat).
    if not all(key in join_df.columns for key in KEY_COLUMNS):
        return frame
    keep = [c for c in (*KEY_COLUMNS, *join_columns) if c in join_df.columns]
    join_df = join_df[keep].drop_duplicates(subset=list(KEY_COLUMNS))
    return frame.merge(join_df, on=list(KEY_COLUMNS), how="left")


def aggregate(df: pd.DataFrame, spec: PlotSpec) -> pd.DataFrame:
    """Tidy summary keyed by ``spec.group_by`` with ``n``, ``value``, ``error``.

    The *unit* of aggregation is the cell (``level="cell"``) or the position
    (``level="position"``). At position level each position is first reduced to
    one number — its per-position cell ``count``, ``mean``, or ``median`` — and
    the cohort is summarised *across positions* (always a mean ± spread), so a
    comparison aggregates tissues, not cells (pseudoreplication guard).

    Columns: the group keys, ``n`` (units aggregated), ``value`` (the central
    number), ``error`` (sd/sem across units; NaN/0 when undefined).
    """
    group = list(spec.group_by)
    if df.empty:
        return pd.DataFrame(columns=[*group, "n", "value", "error"])

    # Pooled cell count: one tally per group, no spread.
    if spec.level == "cell" and spec.stat == "count":
        counts = (
            df.groupby(group, dropna=False).size().reset_index(name="value")
            if group
            else pd.DataFrame([{"value": len(df)}])
        )
        counts["n"] = counts["value"].astype(int)
        counts["error"] = float("nan")
        return counts[[*group, "n", "value", "error"]]

    units, value_col = _reduction_units(df, spec)
    # Across-unit reduction: median only when the unit is the cell and the chosen
    # stat is median; otherwise (and always across positions) a mean.
    central = "median" if (spec.level == "cell" and spec.stat == "median") else "mean"
    rows: list[dict[str, Any]] = []
    grouped = units.groupby(group, dropna=False) if group else [(None, units)]
    for key, chunk in grouped:
        values = pd.to_numeric(chunk[value_col], errors="coerce").dropna()
        rows.append({
            **_group_key_to_dict(group, key),
            "n": int(len(values)),
            "value": float(getattr(values, central)()) if len(values) else float("nan"),
            "error": _spread(values, spec.error),
        })
    return pd.DataFrame(rows, columns=[*group, "n", "value", "error"])


def _reduction_units(df: pd.DataFrame, spec: PlotSpec) -> tuple[pd.DataFrame, str]:
    """Per-unit values to reduce over a group: per-position summaries (position
    level) or the pooled per-cell values themselves (cell level)."""
    group = list(spec.group_by)
    if spec.level == "position":
        keys = list(dict.fromkeys([*group, "position_id"]))
        if spec.stat == "count":
            units = df.groupby(keys, dropna=False).size().reset_index(name="_unit")
        else:
            agg = "mean" if spec.stat == "mean" else "median"
            units = (
                df.groupby(keys, dropna=False)[spec.value].agg(agg).reset_index(name="_unit")
            )
        return units, "_unit"
    return df, spec.value


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


def build_figure(
    df: pd.DataFrame, spec: PlotSpec, style_spec: StyleSpec | None = None
) -> Figure:
    """Render *spec* over *df* with *style_spec*; return a `Figure` with an Agg
    canvas attached so ``fig.savefig(...)`` works headlessly (the Qt frontend
    reattaches its own).

    Distribution-family plots (``DISTRIBUTION_PLOTS``) route through seaborn,
    whose ``hue=`` maps onto the group-by / ``class_label`` model. ``bar``/``line``
    stay custom matplotlib driven by :func:`aggregate`, preserving the
    position-level pseudoreplication guard (seaborn must not silently re-aggregate
    raw cells there)."""
    style_spec = style_spec or StyleSpec()
    with mpl.style.context([style_spec.style, _rc_overrides(style_spec)]):
        fig = Figure(figsize=(style_spec.width, style_spec.height), tight_layout=True)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        if df.empty:
            ax.set_title(style_spec.title or "No data in scope")
            return fig

        if spec.plot in DISTRIBUTION_PLOTS:
            _plot_distribution(ax, df, spec, style_spec)
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


def _plot_distribution(ax, df: pd.DataFrame, spec: PlotSpec, style_spec: StyleSpec) -> None:
    group = list(spec.group_by)
    data, hue = _group_label_column(df, group)
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
        sns.boxplot(data=data, x=hue, y=spec.value, hue=hue, palette=palette, ax=ax, legend=False)
    elif spec.plot == "violin":
        sns.violinplot(
            data=data, x=hue, y=spec.value, hue=hue, palette=palette, ax=ax, legend=False,
        )
    else:  # strip | swarm
        plot = sns.stripplot if spec.plot == "strip" else sns.swarmplot
        plot(data=data, x=hue, y=spec.value, hue=hue, palette=palette, ax=ax, legend=False)


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
        return "cells per position" if spec.level == "position" else "cell count"
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
