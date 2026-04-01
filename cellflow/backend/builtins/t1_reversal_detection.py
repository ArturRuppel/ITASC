"""T1 reversal detection module — find T1 events that undo a previous T1."""
from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..api import Source, get_t1_events
from ..analysis_modules import (
    AnalysisModule,
    AnalysisResult,
    ParamType,
    Parameter,
)


class T1ReversalDetection(AnalysisModule):
    """Detect T1 events that reverse a previous T1 (same 4 cells, swapped pairs)."""

    @property
    def name(self) -> str:
        return "t1_reversal_detection"

    @property
    def description(self) -> str:
        return "Detect T1 events that reverse a previous T1 (same 4 cells, swapped pairs)"

    def parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="max_reversal_lag",
                label="Max reversal lag (frames)",
                type=ParamType.INT,
                default=50,
                min=1,
                max=500,
                description="Maximum frames between a T1 and its reversal",
            ),
        ]

    def compute(self, source: Source, **params: Any) -> AnalysisResult:
        params = self.validate_params(**params)
        max_lag = params["max_reversal_lag"]

        events_df = get_t1_events(source)
        metadata: dict = {}

        if events_df.empty:
            metadata["n_total_t1"] = 0
            return AnalysisResult(
                tables={"main": pd.DataFrame(), "reversals": pd.DataFrame()},
                metadata=metadata,
            )

        multi_tissue = "tissue_id" in events_df.columns

        # Process each tissue independently
        all_reversals: list = []
        if multi_tissue:
            groups = events_df.groupby("tissue_id")
        else:
            groups = [(None, events_df)]

        for tissue_id, group_df in groups:
            group_df = group_df.sort_values("frame").reset_index(drop=True)
            reversals = _find_reversals(group_df, max_lag)
            if tissue_id is not None:
                for r in reversals:
                    r["tissue_id"] = tissue_id
            all_reversals.extend(reversals)

        n_total = len(events_df)
        n_reversed = len(all_reversals)

        metadata["n_total_t1"] = n_total
        metadata["n_reversed"] = n_reversed
        metadata["n_unreversed"] = n_total - 2 * n_reversed  # each reversal pair uses 2 events
        metadata["reversal_fraction"] = n_reversed / n_total if n_total > 0 else 0.0

        if all_reversals:
            rev_df = pd.DataFrame(all_reversals)
            lags = rev_df["lag_frames"].values
            metadata["mean_lag"] = float(np.mean(lags))
            metadata["median_lag"] = float(np.median(lags))
        else:
            rev_df = pd.DataFrame()

        # Summary table
        summary = pd.DataFrame([{
            "n_total_t1": n_total,
            "n_reversed": n_reversed,
            "reversal_fraction": metadata["reversal_fraction"],
            "mean_lag": metadata.get("mean_lag", np.nan),
            "median_lag": metadata.get("median_lag", np.nan),
        }])

        return AnalysisResult(
            tables={"main": summary, "reversals": rev_df},
            metadata=metadata,
        )

    def visualize(self, result: AnalysisResult, **params: Any) -> List[go.Figure]:
        params = self.validate_params(**params)
        meta = result.metadata
        rev_df = result.tables.get("reversals", pd.DataFrame())

        if rev_df.empty:
            return []

        figs: list = []
        lags = rev_df["lag_frames"].values

        # Figure 1: Histogram of reversal time lags
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(
            x=lags,
            marker_color="#89b4fa",
            opacity=0.7,
        ))
        annotation_text = (
            f"n<sub>reversed</sub>={meta.get('n_reversed', '?')}/{meta.get('n_total_t1', '?')}<br>"
            f"fraction={meta.get('reversal_fraction', 0):.2f}<br>"
            f"mean lag={meta.get('mean_lag', 0):.1f}<br>"
            f"median lag={meta.get('median_lag', 0):.1f}"
        )
        fig1.update_layout(
            title="T1 reversal time lag distribution",
            xaxis_title="Lag (frames)",
            yaxis_title="Count",
            annotations=[dict(
                x=0.95, y=0.95, xref="paper", yref="paper",
                text=annotation_text, showarrow=False,
                bgcolor="rgba(49,50,68,0.85)", bordercolor="#585b70",
                font=dict(size=11, color="#cdd6f4"), align="left",
            )],
        )
        figs.append(fig1)

        # Figure 2: ECDF of reversal lags
        sorted_lags = np.sort(lags)
        ecdf_y = np.arange(1, len(sorted_lags) + 1) / len(sorted_lags)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=sorted_lags,
            y=ecdf_y,
            mode="lines",
            line=dict(color="#a6e3a1", width=2),
            name="ECDF",
        ))
        fig2.update_layout(
            title="Cumulative distribution of T1 reversal times",
            xaxis_title="Lag (frames)",
            yaxis_title="Cumulative fraction",
        )
        figs.append(fig2)

        return figs


def _find_reversals(events_df: pd.DataFrame, max_lag: int) -> list:
    """Find reversal pairs within a single tissue's events.

    A reversal occurs when event B's losing pair matches event A's gaining
    pair and they involve the same 4 cells.

    Uses greedy first-match: each event participates in at most one reversal.
    """
    n = len(events_df)
    used = set()  # indices that already participated in a reversal
    reversals = []

    frames = events_df["frame"].values
    losing_a = events_df["losing_a"].values
    losing_b = events_df["losing_b"].values
    gaining_a = events_df["gaining_a"].values
    gaining_b = events_df["gaining_b"].values

    for i in range(n):
        if i in used:
            continue
        gaining_set_i = frozenset((int(gaining_a[i]), int(gaining_b[i])))
        all_cells_i = frozenset((
            int(losing_a[i]), int(losing_b[i]),
            int(gaining_a[i]), int(gaining_b[i]),
        ))

        for j in range(i + 1, n):
            if j in used:
                continue
            lag = int(frames[j] - frames[i])
            if lag > max_lag:
                break  # events are sorted by frame, no point looking further
            if lag <= 0:
                continue

            losing_set_j = frozenset((int(losing_a[j]), int(losing_b[j])))
            all_cells_j = frozenset((
                int(losing_a[j]), int(losing_b[j]),
                int(gaining_a[j]), int(gaining_b[j]),
            ))

            if all_cells_i == all_cells_j and gaining_set_i == losing_set_j:
                used.add(i)
                used.add(j)
                reversals.append({
                    "event_a_frame": int(frames[i]),
                    "event_b_frame": int(frames[j]),
                    "lag_frames": lag,
                    "losing_a_orig": int(losing_a[i]),
                    "losing_b_orig": int(losing_b[i]),
                    "gaining_a_orig": int(gaining_a[i]),
                    "gaining_b_orig": int(gaining_b[i]),
                })
                break  # move on to next event

    return reversals
