"""T1 transition rate analysis module."""
from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..api import Source, get_t1_events, get_t1_rate
from ..analysis_modules import (
    AnalysisModule,
    AnalysisResult,
    ParamType,
    Parameter,
)


class T1TransitionRate(AnalysisModule):
    """T1 transition rate as a function of time."""

    @property
    def name(self) -> str:
        return "t1_transition_rate"

    @property
    def description(self) -> str:
        return "T1 event rate over time with smoothing window"

    def parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="window",
                label="Smoothing window (frames)",
                type=ParamType.INT,
                default=1,
                min=1,
                max=50,
                description="Number of frames for rolling average of T1 rate",
            ),
            Parameter(
                name="cumulative",
                label="Show cumulative",
                type=ParamType.BOOL,
                default=False,
                description="Also show cumulative T1 count",
            ),
        ]

    def compute(self, source: Source, **params: Any) -> AnalysisResult:
        params = self.validate_params(**params)
        window = params["window"]

        events = get_t1_events(source)
        rate = get_t1_rate(source, window=window)

        metadata: dict = {
            "total_t1_events": len(events),
            "n_frames": len(rate),
        }

        if not events.empty:
            metadata["mean_rate"] = float(rate["t1_rate"].mean())
            metadata["peak_frame"] = int(rate.loc[rate["t1_rate"].idxmax(), "frame"])

        return AnalysisResult(
            tables={"events": events, "rate": rate},
            metadata=metadata,
        )

    def visualize(self, result: AnalysisResult, **params: Any) -> List[go.Figure]:
        params = self.validate_params(**params)
        rate = result.tables.get("rate", pd.DataFrame())
        if rate.empty:
            return []

        figs: list = []

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=rate["frame"], y=rate["n_t1_events"],
            name="Events per frame", opacity=0.3,
            marker_color="#89b4fa",
        ))
        fig.add_trace(go.Scatter(
            x=rate["frame"], y=rate["t1_rate"],
            name=f"Rate (window={params['window']})",
            mode="lines", line=dict(color="#f38ba8", width=2),
        ))

        if params.get("cumulative", False):
            fig.add_trace(go.Scatter(
                x=rate["frame"], y=rate["n_t1_events"].cumsum(),
                name="Cumulative",
                mode="lines", line=dict(color="#a6e3a1", dash="dash", width=1.5),
                yaxis="y2",
            ))
            fig.update_layout(
                yaxis2=dict(
                    title="Cumulative T1 events",
                    overlaying="y", side="right",
                ),
            )

        fig.update_layout(
            title="T1 transition rate",
            xaxis_title="Frame",
            yaxis_title="T1 events",
            barmode="overlay",
        )
        figs.append(fig)

        return figs
