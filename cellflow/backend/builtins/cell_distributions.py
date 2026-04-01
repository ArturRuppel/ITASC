"""Cell area, shape index, and coordination number distributions."""
from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..api import Source, get_cells
from ..analysis_modules import (
    AnalysisModule,
    AnalysisResult,
    ParamType,
    Parameter,
)


class CellDistributions(AnalysisModule):
    """Distributions of cell area, shape index, and coordination number."""

    @property
    def name(self) -> str:
        return "cell_distributions"

    @property
    def description(self) -> str:
        return "Cell area, shape index, and coordination number distributions"

    def parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="metric",
                label="Metric",
                type=ParamType.CHOICE,
                default="area",
                choices=["area", "shape_index", "num_neighbors", "perimeter"],
                description="Which cell property to analyze",
            ),
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
                name="per_frame",
                label="Per-frame breakdown",
                type=ParamType.BOOL,
                default=False,
                description="Show mean over time",
            ),
            Parameter(
                name="min_neighbors",
                label="Min neighbors",
                type=ParamType.INT,
                default=None,
                min=0,
                max=20,
                description="Only include cells with at least this many neighbors",
            ),
        ]

    def compute(self, source: Source, **params: Any) -> AnalysisResult:
        params = self.validate_params(**params)
        metric = params["metric"]

        df = get_cells(source, min_neighbors=params.get("min_neighbors"))

        metadata: dict = {
            "n_cells": len(df),
            "n_frames": df["frame"].nunique() if not df.empty else 0,
            "metric": metric,
        }

        if df.empty:
            return AnalysisResult(tables={"main": df}, metadata=metadata)

        col = df[metric].dropna()
        metadata["mean"] = float(col.mean())
        metadata["std"] = float(col.std())
        metadata["median"] = float(col.median())

        # Per-frame summary
        summary = (
            df.groupby("frame")[metric]
            .agg(["mean", "std", "median", "count"])
            .reset_index()
        )
        summary.columns = ["frame", "mean", "std", "median", "count"]

        return AnalysisResult(
            tables={"main": df, "per_frame_summary": summary},
            metadata=metadata,
        )

    def visualize(self, result: AnalysisResult, **params: Any) -> List[go.Figure]:
        params = self.validate_params(**params)
        df = result.tables.get("main", pd.DataFrame())
        metric = result.metadata.get("metric", "area")
        if df.empty:
            return []

        figs: list = []
        n_bins = params.get("n_bins", 30)

        labels = {
            "area": "Cell area (px\u00b2)",
            "shape_index": "Shape index (p / \u221aa)",
            "num_neighbors": "Coordination number",
            "perimeter": "Cell perimeter (px)",
        }
        xlabel = labels.get(metric, metric)

        # Histogram
        fig = go.Figure()
        col = df[metric].dropna()
        fig.add_trace(go.Histogram(
            x=col,
            nbinsx=n_bins,
            histnorm="probability density",
            marker_color="#89b4fa",
            opacity=0.7,
        ))
        annotation = (
            f"n={len(col)}<br>"
            f"\u03bc={result.metadata.get('mean', 0):.2f}<br>"
            f"\u03c3={result.metadata.get('std', 0):.2f}"
        )
        fig.update_layout(
            title=f"{xlabel} distribution",
            xaxis_title=xlabel,
            yaxis_title="Density",
            annotations=[dict(
                x=0.95, y=0.95, xref="paper", yref="paper",
                text=annotation, showarrow=False,
                bgcolor="rgba(49,50,68,0.85)", bordercolor="#585b70",
                font=dict(size=11, color="#cdd6f4"), align="left",
            )],
        )
        figs.append(fig)

        # Time series
        if params.get("per_frame", False):
            summary = result.tables.get("per_frame_summary")
            if summary is not None and not summary.empty and summary["frame"].nunique() > 1:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=summary["frame"],
                    y=summary["mean"],
                    error_y=dict(type="data", array=summary["std"], visible=True),
                    mode="lines+markers",
                    marker_color="#89b4fa",
                ))
                fig2.update_layout(
                    title=f"{xlabel} over time",
                    xaxis_title="Frame",
                    yaxis_title=f"Mean {xlabel.lower()}",
                )
                figs.append(fig2)

        return figs
