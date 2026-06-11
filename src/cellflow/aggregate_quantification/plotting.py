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

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

__all__ = [
    "PlotSpec",
    "PositionSource",
    "pool_object_tables",
    "aggregate",
    "build_figure",
    "write_csv",
]

#: The tidy keys every per-object table carries; joins and per-position grouping
#: are keyed on these.
KEY_COLUMNS = ("frame", "cell_id")
#: Bucket label for cells with no classification (no contacts artifact, or the
#: position was never classified).
UNCLASSIFIED = "unclassified"

_LEVELS = ("cell", "position")
_PLOTS = ("hist", "box", "violin", "bar", "line")
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


def build_figure(df: pd.DataFrame, spec: PlotSpec) -> Figure:
    """Render *spec* over *df*; return a `Figure` with an Agg canvas attached so
    ``fig.savefig(...)`` works headlessly (the Qt frontend reattaches its own)."""
    fig = Figure(figsize=(6, 4), tight_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    if df.empty:
        ax.set_title("No data in scope")
        return fig

    if spec.plot in ("hist", "box", "violin"):
        _plot_distribution(ax, df, spec)
    elif spec.plot == "bar":
        _plot_bar(ax, df, spec)
    else:  # line
        _plot_line(ax, df, spec)
    ax.set_title(spec.value)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize="small")
    return fig


def _group_series(df: pd.DataFrame, group: list[str]):
    """Yield ``(label, sub_df)`` per group combination (one group when none)."""
    if not group:
        yield "all", df
        return
    for key, chunk in df.groupby(group, dropna=False):
        parts = key if isinstance(key, tuple) else (key,)
        yield " · ".join(str(p) for p in parts), chunk


def _plot_distribution(ax, df: pd.DataFrame, spec: PlotSpec) -> None:
    group = list(spec.group_by)
    labels, arrays = [], []
    for label, chunk in _group_series(df, group):
        values = pd.to_numeric(chunk[spec.value], errors="coerce").dropna().to_numpy()
        if values.size:
            labels.append(label)
            arrays.append(values)
    if not arrays:
        ax.set_title("No data in scope")
        return
    if spec.plot == "hist":
        for label, values in zip(labels, arrays):
            ax.hist(values, bins=spec.bins, alpha=0.5, label=label)
        ax.set_xlabel(spec.value)
        ax.set_ylabel("count")
    elif spec.plot == "box":
        ax.boxplot(arrays, tick_labels=labels)
        ax.set_ylabel(spec.value)
    else:  # violin
        ax.violinplot(arrays, showmedians=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel(spec.value)


def _plot_bar(ax, df: pd.DataFrame, spec: PlotSpec) -> None:
    summary = aggregate(df, spec)
    group = list(spec.group_by)
    labels = [
        " · ".join(str(summary.iloc[i][g]) for g in group) if group else "all"
        for i in range(len(summary))
    ]
    x = np.arange(len(summary))
    errors = summary["error"].fillna(0.0).to_numpy()
    ax.bar(x, summary["value"].to_numpy(), yerr=errors, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(_bar_ylabel(spec))


def _bar_ylabel(spec: PlotSpec) -> str:
    if spec.stat == "count":
        return "cells per position" if spec.level == "position" else "cell count"
    return f"{spec.stat} {spec.value}"


def _plot_line(ax, df: pd.DataFrame, spec: PlotSpec) -> None:
    group = list(spec.group_by)
    for label, chunk in _group_series(df, group):
        if spec.stat == "count":
            series = chunk.groupby("frame", dropna=False).size()
        else:
            agg = "mean" if spec.stat == "mean" else "median"
            series = chunk.groupby("frame", dropna=False)[spec.value].agg(agg)
        series = series.sort_index()
        ax.plot(series.index.to_numpy(), series.to_numpy(), marker="o", label=label)
    ax.set_xlabel("frame")
    ax.set_ylabel("count" if spec.stat == "count" else f"{spec.stat} {spec.value}")


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write *df* to CSV, ensuring a ``.csv`` suffix (Qt's save dialog often omits
    it). Returns the final path."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        path = path.with_name(path.name + ".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
