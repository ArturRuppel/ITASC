"""Property correlation analysis module — scatter plot with linear regression."""
from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import linregress

from ..api import Source, get_cells, get_junctions, get_trajectories
from ..analysis_modules import (
    AnalysisModule,
    AnalysisResult,
    ParamType,
    Parameter,
)


class PropertyCorrelation(AnalysisModule):
    """Scatter plot with linear regression between two data properties."""

    @property
    def name(self) -> str:
        return "property_correlation"

    @property
    def description(self) -> str:
        return "Scatter plot with linear regression between two data properties"

    def parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="data_source",
                label="Data source",
                type=ParamType.CHOICE,
                default="cells",
                choices=["cells", "junctions", "trajectories"],
                description="Which data to query",
            ),
            Parameter(
                name="x_property",
                label="X property",
                type=ParamType.STR,
                default="area",
                description="Column name for x-axis",
            ),
            Parameter(
                name="y_property",
                label="Y property",
                type=ParamType.STR,
                default="shape_index",
                description="Column name for y-axis",
            ),
            Parameter(
                name="include_tags",
                label="Include tags",
                type=ParamType.TAG,
                default=None,
                description="Only include items with these tags (junctions/trajectories only)",
            ),
            Parameter(
                name="exclude_tags",
                label="Exclude tags",
                type=ParamType.TAG,
                default="edge_border",
                description="Exclude items with these tags",
            ),
            Parameter(
                name="show_regression",
                label="Show regression",
                type=ParamType.BOOL,
                default=True,
                description="Overlay linear regression line with statistics",
            ),
        ]

    def compute(self, source: Source, **params: Any) -> AnalysisResult:
        params = self.validate_params(**params)
        data_source = params["data_source"]
        x_prop = params["x_property"]
        y_prop = params["y_property"]
        include = _parse_tags(params["include_tags"])
        exclude = _parse_tags(params["exclude_tags"])

        # Query the appropriate API
        if data_source == "cells":
            df = get_cells(source)
        elif data_source == "junctions":
            df = get_junctions(source, tags=include, exclude_tags=exclude)
        else:
            df = get_trajectories(source, tags=include, exclude_tags=exclude)

        metadata: dict = {"data_source": data_source, "x_property": x_prop, "y_property": y_prop}

        if df.empty:
            metadata["error"] = "No data returned from query"
            return AnalysisResult(tables={"main": df}, metadata=metadata)

        # Validate columns exist
        for col in (x_prop, y_prop):
            if col not in df.columns:
                metadata["error"] = (
                    f"Column '{col}' not found. "
                    f"Available: {sorted(c for c in df.columns if not c.startswith('_'))}"
                )
                return AnalysisResult(tables={"main": pd.DataFrame()}, metadata=metadata)

        # Keep only the relevant columns, drop NaN rows
        subset = df[[x_prop, y_prop]].dropna()
        metadata["n_points"] = len(subset)

        if len(subset) < 2:
            return AnalysisResult(tables={"main": subset}, metadata=metadata)

        # Linear regression
        if params["show_regression"]:
            x = subset[x_prop].values
            y = subset[y_prop].values
            result = linregress(x, y)
            metadata["slope"] = float(result.slope)
            metadata["intercept"] = float(result.intercept)
            metadata["r_squared"] = float(result.rvalue ** 2)
            metadata["p_value"] = float(result.pvalue)
            metadata["std_err"] = float(result.stderr)

        return AnalysisResult(tables={"main": subset}, metadata=metadata)

    def visualize(self, result: AnalysisResult, **params: Any) -> List[go.Figure]:
        params = self.validate_params(**params)
        df = result.tables.get("main", pd.DataFrame())
        meta = result.metadata

        if "error" in meta:
            return []
        if df.empty:
            return []

        x_prop = meta.get("x_property", "x")
        y_prop = meta.get("y_property", "y")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[x_prop],
            y=df[y_prop],
            mode="markers",
            marker=dict(color="#89b4fa", size=4, opacity=0.5),
            name="Data",
        ))

        annotation_text = f"n={meta.get('n_points', '?')}"

        if params.get("show_regression") and "slope" in meta:
            x_range = np.linspace(df[x_prop].min(), df[x_prop].max(), 100)
            y_hat = meta["slope"] * x_range + meta["intercept"]
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_hat,
                mode="lines",
                line=dict(color="#f38ba8", width=2),
                name="Regression",
            ))
            annotation_text = (
                f"n={meta.get('n_points', '?')}<br>"
                f"R\u00b2={meta['r_squared']:.3f}<br>"
                f"slope={meta['slope']:.4g}<br>"
                f"p={meta['p_value']:.2e}"
            )

        fig.update_layout(
            title=f"{y_prop} vs {x_prop}",
            xaxis_title=x_prop,
            yaxis_title=y_prop,
            annotations=[dict(
                x=0.95, y=0.95, xref="paper", yref="paper",
                text=annotation_text, showarrow=False,
                bgcolor="rgba(49,50,68,0.85)", bordercolor="#585b70",
                font=dict(size=11, color="#cdd6f4"), align="left",
            )],
        )

        return [fig]


def _parse_tags(value: Any) -> set | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return {t.strip() for t in value.split(",") if t.strip()}
    return set(value)
