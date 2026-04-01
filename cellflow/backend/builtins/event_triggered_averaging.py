"""Event-triggered averaging around T1 transitions."""
from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..api import Source, get_trajectories, get_t1_events
from ..analysis_modules import (
    AnalysisModule,
    AnalysisResult,
    ParamType,
    Parameter,
)


class EventTriggeredAveraging(AnalysisModule):
    """Average junction length trajectory aligned to T1 events."""

    @property
    def name(self) -> str:
        return "event_triggered_averaging"

    @property
    def description(self) -> str:
        return "Average junction length aligned to T1 transition events"

    def parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="window_before",
                label="Frames before event",
                type=ParamType.INT,
                default=10,
                min=1,
                max=100,
                description="Number of frames to include before the T1 event",
            ),
            Parameter(
                name="window_after",
                label="Frames after event",
                type=ParamType.INT,
                default=10,
                min=1,
                max=100,
                description="Number of frames to include after the T1 event",
            ),
            Parameter(
                name="include_tags",
                label="Include tags",
                type=ParamType.TAG,
                default=None,
                description="Only include trajectories with these tags (comma-separated)",
            ),
            Parameter(
                name="exclude_tags",
                label="Exclude tags",
                type=ParamType.TAG,
                default="edge_border",
                description="Exclude trajectories with these tags",
            ),
            Parameter(
                name="use_signed_length",
                label="Use signed length",
                type=ParamType.BOOL,
                default=True,
                description="Use signed length (shows sign flip at T1) vs absolute length",
            ),
        ]

    def compute(self, source: Source, **params: Any) -> AnalysisResult:
        params = self.validate_params(**params)
        w_before = params["window_before"]
        w_after = params["window_after"]
        signed = params["use_signed_length"]
        include = _parse_tags(params.get("include_tags"))
        exclude = _parse_tags(params.get("exclude_tags"))

        # Get trajectories that have T1 events
        traj_df = get_trajectories(
            source, has_t1=True, tags=include, exclude_tags=exclude,
        )
        events_df = get_t1_events(source)

        metadata: dict = {
            "n_t1_events": len(events_df),
            "n_trajectories": traj_df["trajectory_id"].nunique() if not traj_df.empty else 0,
            "window_before": w_before,
            "window_after": w_after,
        }

        if traj_df.empty or events_df.empty:
            return AnalysisResult(
                tables={"main": pd.DataFrame(), "aligned": pd.DataFrame()},
                metadata=metadata,
            )

        # Build aligned traces: for each trajectory with T1 events,
        # extract a window around each T1 event frame
        length_col = "signed_length" if signed else "abs_length"
        aligned_rows: list = []
        n_events_used = 0

        for traj_id, traj_group in traj_df.groupby("trajectory_id"):
            traj_group = traj_group.sort_values("frame")
            frame_to_length = dict(zip(traj_group["frame"], traj_group[length_col]))

            # Find T1 event frames for this trajectory by checking
            # which T1 events involve cells from this trajectory
            traj_cells = set()
            for _, row in traj_group.iterrows():
                traj_cells.add(row["cell_a"])
                traj_cells.add(row["cell_b"])

            # Match T1 events to this trajectory
            for _, evt in events_df.iterrows():
                evt_cells = {evt["losing_a"], evt["losing_b"], evt["gaining_a"], evt["gaining_b"]}
                if not (traj_cells & evt_cells):
                    continue

                t1_frame = evt["frame"]
                n_events_used += 1

                for dt in range(-w_before, w_after + 1):
                    f = t1_frame + dt
                    if f in frame_to_length:
                        aligned_rows.append({
                            "trajectory_id": traj_id,
                            "t1_frame": t1_frame,
                            "dt": dt,
                            "length": frame_to_length[f],
                        })

        aligned = pd.DataFrame(aligned_rows)
        metadata["n_events_used"] = n_events_used

        # Compute ensemble average
        if not aligned.empty:
            ensemble = (
                aligned.groupby("dt")["length"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            ensemble.columns = ["dt", "mean_length", "std_length", "count"]
            ensemble["sem"] = ensemble["std_length"] / np.sqrt(ensemble["count"])
        else:
            ensemble = pd.DataFrame()

        return AnalysisResult(
            tables={"main": ensemble, "aligned": aligned},
            metadata=metadata,
        )

    def visualize(self, result: AnalysisResult, **params: Any) -> List[go.Figure]:
        params = self.validate_params(**params)
        ensemble = result.tables.get("main", pd.DataFrame())
        aligned = result.tables.get("aligned", pd.DataFrame())
        if ensemble.empty:
            return []

        figs: list = []
        length_label = "Signed length" if params.get("use_signed_length", True) else "Length"

        # Ensemble average with SEM band
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ensemble["dt"], y=ensemble["mean_length"] + ensemble["sem"],
            mode="lines", line=dict(width=0), showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=ensemble["dt"], y=ensemble["mean_length"] - ensemble["sem"],
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(137,180,250,0.25)", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=ensemble["dt"], y=ensemble["mean_length"],
            mode="lines", line=dict(color="#cdd6f4", width=2),
            name="Mean",
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="#f38ba8", annotation_text="T1 event")
        fig.add_hline(y=0, line_dash="dot", line_color="#585b70", line_width=0.5)
        fig.update_layout(
            title=f"Event-triggered average (n={result.metadata.get('n_events_used', '?')} events)",
            xaxis_title="Frames relative to T1 event",
            yaxis_title=f"Mean {length_label.lower()} (px)",
        )
        figs.append(fig)

        # Individual traces (if not too many)
        if not aligned.empty:
            n_traces = aligned.groupby(["trajectory_id", "t1_frame"]).ngroups
            if n_traces <= 50:
                fig2 = go.Figure()
                for (tid, t1f), grp in aligned.groupby(["trajectory_id", "t1_frame"]):
                    fig2.add_trace(go.Scatter(
                        x=grp["dt"], y=grp["length"],
                        mode="lines", opacity=0.3, line=dict(width=0.8),
                        showlegend=False,
                        hovertemplate=f"traj {tid}, T1@{t1f}<br>dt=%{{x}}, len=%{{y:.1f}}",
                    ))
                fig2.add_trace(go.Scatter(
                    x=ensemble["dt"], y=ensemble["mean_length"],
                    mode="lines", line=dict(color="#cdd6f4", width=2),
                    name="Mean",
                ))
                fig2.add_vline(x=0, line_dash="dash", line_color="#f38ba8")
                fig2.add_hline(y=0, line_dash="dot", line_color="#585b70", line_width=0.5)
                fig2.update_layout(
                    title=f"Individual traces (n={n_traces})",
                    xaxis_title="Frames relative to T1 event",
                    yaxis_title=f"{length_label} (px)",
                )
                figs.append(fig2)

        return figs


def _parse_tags(value: Any) -> set | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return {t.strip() for t in value.split(",") if t.strip()}
    return set(value)
