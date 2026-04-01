"""Junction length distribution analysis module."""
from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..api import Source, get_junctions
from ..analysis_modules import (
    AnalysisModule,
    AnalysisResult,
    ParamType,
    Parameter,
)


class JunctionLengthDistribution(AnalysisModule):
    """Histogram of junction lengths, filterable by tag and neighbor count."""

    @property
    def name(self) -> str:
        return "junction_length_distribution"

    @property
    def description(self) -> str:
        return "Distribution of junction lengths across frames"

    def parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="n_bins",
                label="Number of bins",
                type=ParamType.INT,
                default=30,
                min=5,
                max=200,
                description="Number of histogram bins",
            ),
            Parameter(
                name="include_tags",
                label="Include tags",
                type=ParamType.TAG,
                default=None,
                description="Only include junctions with these tags (comma-separated, or None for all)",
            ),
            Parameter(
                name="exclude_tags",
                label="Exclude tags",
                type=ParamType.TAG,
                default="edge_border",
                description="Exclude junctions with these tags (comma-separated)",
            ),
            Parameter(
                name="per_frame",
                label="Per-frame breakdown",
                type=ParamType.BOOL,
                default=False,
                description="Show separate distributions for each frame",
            ),
            Parameter(
                name="normalize",
                label="Normalize",
                type=ParamType.BOOL,
                default=True,
                description="Normalize histogram to density",
            ),
        ]

    def compute(self, source: Source, **params: Any) -> AnalysisResult:
        params = self.validate_params(**params)

        include = _parse_tags(params["include_tags"])
        exclude = _parse_tags(params["exclude_tags"])

        df = get_junctions(source, tags=include, exclude_tags=exclude)

        metadata: dict = {
            "n_junctions": len(df),
            "n_frames": df["frame"].nunique() if not df.empty else 0,
        }

        if df.empty:
            return AnalysisResult(tables={"main": df}, metadata=metadata)

        metadata["mean_length"] = float(df["length"].mean())
        metadata["std_length"] = float(df["length"].std())
        metadata["median_length"] = float(df["length"].median())

        # Per-frame summary
        summary = (
            df.groupby("frame")["length"]
            .agg(["mean", "std", "median", "count"])
            .reset_index()
        )
        summary.columns = ["frame", "mean_length", "std_length", "median_length", "count"]

        return AnalysisResult(
            tables={"main": df, "per_frame_summary": summary},
            metadata=metadata,
        )

    def visualize(self, result: AnalysisResult, **params: Any) -> List[go.Figure]:
        params = self.validate_params(**params)
        df = result.tables.get("main", pd.DataFrame())
        if df.empty:
            return []

        figs: list = []
        n_bins = params.get("n_bins", 30)
        density = params.get("normalize", True)

        # Overall distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["length"],
            nbinsx=n_bins,
            histnorm="probability density" if density else None,
            marker_color="#89b4fa",
            opacity=0.7,
        ))
        meta = result.metadata
        annotation = (
            f"n={meta.get('n_junctions', '?')}<br>"
            f"\u03bc={meta.get('mean_length', 0):.1f}<br>"
            f"\u03c3={meta.get('std_length', 0):.1f}"
        )
        fig.update_layout(
            title="Junction length distribution",
            xaxis_title="Junction length (px)",
            yaxis_title="Density" if density else "Count",
            annotations=[dict(
                x=0.95, y=0.95, xref="paper", yref="paper",
                text=annotation, showarrow=False,
                bgcolor="rgba(49,50,68,0.85)", bordercolor="#585b70",
                font=dict(size=11, color="#cdd6f4"), align="left",
            )],
        )
        figs.append(fig)

        # Per-frame breakdown
        if params.get("per_frame", False):
            summary = result.tables.get("per_frame_summary")
            if summary is not None and not summary.empty and summary["frame"].nunique() > 1:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=summary["frame"],
                    y=summary["mean_length"],
                    error_y=dict(type="data", array=summary["std_length"], visible=True),
                    mode="lines+markers",
                    marker_color="#89b4fa",
                ))
                fig2.update_layout(
                    title="Junction length over time",
                    xaxis_title="Frame",
                    yaxis_title="Mean junction length (px)",
                )
                figs.append(fig2)

        return figs


def _parse_tags(value: Any) -> set | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return {t.strip() for t in value.split(",") if t.strip()}
    return set(value)
