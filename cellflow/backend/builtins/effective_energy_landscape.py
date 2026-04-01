"""Effective energy landscape analysis module — -log(P(L)) from signed junction lengths."""
from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..api import Source, get_trajectories
from ..analysis_modules import (
    AnalysisModule,
    AnalysisResult,
    ParamType,
    Parameter,
)


class EffectiveEnergyLandscape(AnalysisModule):
    """Effective energy landscape -log(P(L)) from signed junction lengths."""

    @property
    def name(self) -> str:
        return "effective_energy_landscape"

    @property
    def description(self) -> str:
        return "Effective energy landscape -log(P(L)) from signed junction lengths"

    def parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="n_bins",
                label="Number of bins",
                type=ParamType.INT,
                default=50,
                min=10,
                max=200,
                description="Number of histogram bins",
            ),
            Parameter(
                name="bin_strategy",
                label="Bin strategy",
                type=ParamType.CHOICE,
                default="linear",
                choices=["linear", "sinh"],
                description=(
                    "'linear' for uniform bins, 'sinh' for denser bins near L=0 "
                    "(better resolution at the energy barrier)"
                ),
            ),
            Parameter(
                name="include_tags",
                label="Include tags",
                type=ParamType.TAG,
                default=None,
                description="Only include trajectories with these tags (comma-separated, or None for all)",
            ),
            Parameter(
                name="exclude_tags",
                label="Exclude tags",
                type=ParamType.TAG,
                default="edge_border",
                description="Exclude trajectories with these tags (comma-separated)",
            ),
        ]

    def compute(self, source: Source, **params: Any) -> AnalysisResult:
        params = self.validate_params(**params)
        include = _parse_tags(params["include_tags"])
        exclude = _parse_tags(params["exclude_tags"])
        n_bins = params["n_bins"]
        strategy = params["bin_strategy"]

        df = get_trajectories(source, tags=include, exclude_tags=exclude)
        metadata: dict = {}

        if df.empty or "signed_length" not in df.columns:
            return AnalysisResult(tables={"main": pd.DataFrame()}, metadata=metadata)

        lengths = df["signed_length"].dropna().values
        if len(lengths) < 4:
            metadata["error"] = f"Not enough data points ({len(lengths)})"
            return AnalysisResult(tables={"main": pd.DataFrame()}, metadata=metadata)

        metadata["n_lengths"] = int(len(lengths))
        metadata["n_trajectories"] = int(df["trajectory_id"].nunique())
        metadata["mean_signed_length"] = float(np.mean(lengths))
        metadata["std_signed_length"] = float(np.std(lengths))

        # Build bin edges
        l_min, l_max = float(lengths.min()), float(lengths.max())
        if strategy == "sinh":
            # sinh-spaced bins: denser near zero
            t = np.linspace(np.arcsinh(l_min), np.arcsinh(l_max), n_bins + 1)
            bin_edges = np.sinh(t)
        else:
            bin_edges = np.linspace(l_min, l_max, n_bins + 1)

        # Histogram
        counts, actual_edges = np.histogram(lengths, bins=bin_edges)
        bin_centers = (actual_edges[:-1] + actual_edges[1:]) / 2.0
        total = counts.sum()
        prob = counts / total

        # -log(P(L)), replacing zeros with NaN
        neg_log_p = np.full_like(prob, np.nan, dtype=float)
        valid = prob > 0
        neg_log_p[valid] = -np.log(prob[valid])

        # Build result table
        main_df = pd.DataFrame({
            "L": bin_centers,
            "P": prob,
            "neg_log_P": neg_log_p,
        })

        # Energy barrier: ΔE_eff = value_at_zero - minimum_left_of_zero
        delta_e = _compute_energy_barrier(bin_centers, neg_log_p)
        if delta_e is not None:
            metadata["delta_E_eff"] = float(delta_e["delta_E_eff"])
            metadata["neg_log_P_at_zero"] = float(delta_e["at_zero"])
            metadata["neg_log_P_min_left"] = float(delta_e["min_left"])
            metadata["L_at_min_left"] = float(delta_e["L_at_min_left"])

        return AnalysisResult(tables={"main": main_df}, metadata=metadata)

    def visualize(self, result: AnalysisResult, **params: Any) -> List[go.Figure]:
        params = self.validate_params(**params)
        df = result.tables.get("main", pd.DataFrame())
        meta = result.metadata

        if df.empty or "neg_log_P" not in df.columns:
            return []

        valid = df["neg_log_P"].notna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.loc[valid, "L"],
            y=df.loc[valid, "neg_log_P"],
            mode="lines+markers",
            line=dict(color="#89b4fa", width=2),
            marker=dict(size=4),
            name="-log P(L)",
        ))

        # Vertical line at L=0
        fig.add_vline(x=0, line_dash="dash", line_color="#f38ba8", line_width=1.5)

        annotation_text = f"n={meta.get('n_lengths', '?')}"

        if "delta_E_eff" in meta:
            de = meta["delta_E_eff"]
            at_zero = meta["neg_log_P_at_zero"]
            min_left = meta["neg_log_P_min_left"]
            l_min_left = meta["L_at_min_left"]

            # Horizontal line at the minimum (well) on the left
            fig.add_hline(
                y=min_left, line_dash="dot", line_color="#a6e3a1",
                line_width=1, opacity=0.6,
            )
            # Horizontal line at zero crossing
            fig.add_hline(
                y=at_zero, line_dash="dot", line_color="#f38ba8",
                line_width=1, opacity=0.6,
            )
            # Arrow showing the barrier
            fig.add_annotation(
                x=l_min_left, y=min_left,
                ax=0, ay=at_zero,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.5,
                arrowcolor="#fab387", arrowwidth=2,
            )

            annotation_text = (
                f"n={meta.get('n_lengths', '?')}<br>"
                f"\u0394E<sub>eff</sub> = {de:.2f}"
            )

        fig.update_layout(
            title="Effective energy landscape",
            xaxis_title="Signed junction length L (px)",
            yaxis_title="-log P(L)",
            annotations=[dict(
                x=0.95, y=0.05, xref="paper", yref="paper",
                text=annotation_text, showarrow=False,
                bgcolor="rgba(49,50,68,0.85)", bordercolor="#585b70",
                font=dict(size=12, color="#cdd6f4"), align="left",
            )],
        )

        return [fig]


def _compute_energy_barrier(bin_centers: np.ndarray, neg_log_p: np.ndarray) -> dict | None:
    """Compute the effective energy barrier ΔE_eff.

    The barrier is defined as the difference between -log(P) interpolated at
    L=0 and the minimum -log(P) for L < 0 (the "well" to the left of zero).

    Returns a dict with keys: delta_E_eff, at_zero, min_left, L_at_min_left,
    or None if computation is not possible.
    """
    valid = ~np.isnan(neg_log_p)
    if valid.sum() < 3:
        return None

    centers_v = bin_centers[valid]
    values_v = neg_log_p[valid]

    # Need data on both sides of zero
    left_mask = centers_v < 0
    right_mask = centers_v > 0
    if not np.any(left_mask) or not np.any(right_mask):
        return None

    # Interpolate -log(P) at L=0
    # np.interp requires monotonically increasing x
    sort_idx = np.argsort(centers_v)
    at_zero = float(np.interp(0.0, centers_v[sort_idx], values_v[sort_idx]))

    # Find minimum of -log(P) for L < 0
    left_values = values_v[left_mask]
    left_centers = centers_v[left_mask]
    min_idx = np.argmin(left_values)
    min_left = float(left_values[min_idx])
    l_at_min_left = float(left_centers[min_idx])

    delta_e = at_zero - min_left
    if delta_e <= 0:
        return None  # No meaningful barrier

    return {
        "delta_E_eff": delta_e,
        "at_zero": at_zero,
        "min_left": min_left,
        "L_at_min_left": l_at_min_left,
    }


def _parse_tags(value: Any) -> set | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return {t.strip() for t in value.split(",") if t.strip()}
    return set(value)
