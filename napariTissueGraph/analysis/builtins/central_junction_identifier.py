"""Central junction identifier module — find the central junction in 4-cell quadruplets."""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ...core.api import Source, _resolve_source
from ...structures import EdgeTrajectory, TissueGraphTimeSeries
from ..modules import (
    AnalysisModule,
    AnalysisResult,
    ParamType,
    Parameter,
)


class CentralJunctionIdentifier(AnalysisModule):
    """Identify central junctions in 4-cell quadruplet configurations."""

    @property
    def name(self) -> str:
        return "central_junction_identifier"

    @property
    def description(self) -> str:
        return "Identify central junctions in 4-cell quadruplet configurations"

    def parameters(self) -> List[Parameter]:
        return [
            Parameter(
                name="tag_name",
                label="Central tag",
                type=ParamType.STR,
                default="central",
                description="Tag to assign to central junction trajectories",
            ),
            Parameter(
                name="peripheral_tag_name",
                label="Peripheral tag",
                type=ParamType.STR,
                default="peripheral",
                description="Tag to assign to non-central junction trajectories",
            ),
            Parameter(
                name="exclude_background",
                label="Exclude background cell",
                type=ParamType.BOOL,
                default=True,
                description="Exclude cell_id=0 (typically the background)",
            ),
        ]

    def compute(self, source: Source, **params: Any) -> AnalysisResult:
        params = self.validate_params(**params)
        exclude_bg = params["exclude_background"]

        items, multi = _resolve_source(source)

        all_central_rows: list = []
        all_traj_rows: list = []
        n_frames_analyzed = 0
        n_frames_skipped = 0

        for tissue_id, series in items:
            # Per-frame: identify central junctions
            central_pairs_by_frame: Dict[int, FrozenSet[int]] = {}

            for frame_idx, frame in series.frames.items():
                cell_ids = set(frame.cells.keys())
                if exclude_bg:
                    cell_ids.discard(0)

                if len(cell_ids) != 4:
                    n_frames_skipped += 1
                    continue

                n_frames_analyzed += 1
                central_pair = _find_central_pair(cell_ids, frame.graph)
                if central_pair is None:
                    n_frames_skipped += 1
                    continue

                central_pairs_by_frame[frame_idx] = central_pair

                # Record the central junction data
                jdata = frame.junctions.get(central_pair)
                row = {
                    "frame": frame_idx,
                    "central_cell_a": min(central_pair),
                    "central_cell_b": max(central_pair),
                    "junction_length": jdata.length if jdata else np.nan,
                }
                if multi:
                    row["tissue_id"] = tissue_id
                all_central_rows.append(row)

            # Match trajectories to central/peripheral classification
            for traj in series.edge_trajectories.values():
                n_central = 0
                n_peripheral = 0
                for fi, cp in zip(traj.frames, traj.cell_pairs):
                    pair_set = frozenset(cp)
                    if fi in central_pairs_by_frame:
                        if pair_set == central_pairs_by_frame[fi]:
                            n_central += 1
                        else:
                            n_peripheral += 1

                total = n_central + n_peripheral
                if total == 0:
                    classification = "unclassified"
                elif n_central > n_peripheral:
                    classification = "central"
                elif n_peripheral > n_central:
                    classification = "peripheral"
                else:
                    classification = "mixed"

                traj_row = {
                    "trajectory_id": traj.trajectory_id,
                    "n_central_frames": n_central,
                    "n_peripheral_frames": n_peripheral,
                    "classification": classification,
                }
                if multi:
                    traj_row["tissue_id"] = tissue_id
                all_traj_rows.append(traj_row)

        main_df = pd.DataFrame(all_central_rows) if all_central_rows else pd.DataFrame()
        traj_df = pd.DataFrame(all_traj_rows) if all_traj_rows else pd.DataFrame()

        n_central_traj = int((traj_df["classification"] == "central").sum()) if not traj_df.empty else 0
        n_peripheral_traj = int((traj_df["classification"] == "peripheral").sum()) if not traj_df.empty else 0

        metadata = {
            "n_frames_analyzed": n_frames_analyzed,
            "n_frames_skipped": n_frames_skipped,
            "n_central_trajectories": n_central_traj,
            "n_peripheral_trajectories": n_peripheral_traj,
            "tag_name": params["tag_name"],
            "peripheral_tag_name": params["peripheral_tag_name"],
        }

        return AnalysisResult(
            tables={"main": main_df, "trajectory_mapping": traj_df},
            metadata=metadata,
        )

    def visualize(self, result: AnalysisResult, **params: Any) -> List[go.Figure]:
        params = self.validate_params(**params)
        meta = result.metadata
        main_df = result.tables.get("main", pd.DataFrame())
        traj_df = result.tables.get("trajectory_mapping", pd.DataFrame())

        figs: list = []

        # Figure 1: Trajectory classification bar chart
        if not traj_df.empty and "classification" in traj_df.columns:
            counts = traj_df["classification"].value_counts()
            colors = {
                "central": "#f38ba8",
                "peripheral": "#89b4fa",
                "mixed": "#fab387",
                "unclassified": "#6c7086",
            }
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=list(counts.index),
                y=list(counts.values),
                marker_color=[colors.get(c, "#cdd6f4") for c in counts.index],
            ))
            fig1.update_layout(
                title="Trajectory classification",
                xaxis_title="Classification",
                yaxis_title="Count",
                annotations=[dict(
                    x=0.95, y=0.95, xref="paper", yref="paper",
                    text=(
                        f"analyzed={meta.get('n_frames_analyzed', '?')} frames<br>"
                        f"skipped={meta.get('n_frames_skipped', '?')} frames"
                    ),
                    showarrow=False,
                    bgcolor="rgba(49,50,68,0.85)", bordercolor="#585b70",
                    font=dict(size=11, color="#cdd6f4"), align="left",
                )],
            )
            figs.append(fig1)

        # Figure 2: Central junction length over time
        if not main_df.empty and "junction_length" in main_df.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=main_df["frame"],
                y=main_df["junction_length"],
                mode="lines+markers",
                marker=dict(size=3, color="#f38ba8"),
                line=dict(color="#f38ba8", width=1.5),
                name="Central junction",
            ))
            fig2.update_layout(
                title="Central junction length over time",
                xaxis_title="Frame",
                yaxis_title="Junction length (px)",
            )
            figs.append(fig2)

        return figs


def _find_central_pair(
    cell_ids: set, graph,
) -> Optional[FrozenSet[int]]:
    """Find the central junction pair in a 4-cell quadruplet.

    The central edge connects the two cells that each have 3 neighbors
    within the quadruplet (i.e., they contact all other 3 cells).
    """
    # Count neighbors within the quadruplet for each cell
    cells_with_3 = []
    for cell in cell_ids:
        if cell not in graph:
            continue
        n_neighbors_in_quad = len(set(graph.neighbors(cell)) & cell_ids)
        if n_neighbors_in_quad == 3:
            cells_with_3.append(cell)

    if len(cells_with_3) != 2:
        return None

    # Verify they are actually connected
    pair = frozenset(cells_with_3)
    if graph.has_edge(cells_with_3[0], cells_with_3[1]):
        return pair
    return None


def apply_central_tags(
    source: Source,
    result: AnalysisResult,
    tag_name: str = "central",
    peripheral_tag_name: str = "peripheral",
) -> int:
    """Apply central/peripheral tags to EdgeTrajectory objects based on module results.

    This is a utility function that mutates the source data. Call it after
    ``CentralJunctionIdentifier.compute()`` to tag the trajectories.

    Parameters
    ----------
    source : Source
        The original data source (must be a TissueGraphTimeSeries or Dataset).
    result : AnalysisResult
        Output from ``CentralJunctionIdentifier.compute()``.
    tag_name : str
        Tag for central trajectories.
    peripheral_tag_name : str
        Tag for peripheral trajectories.

    Returns
    -------
    int
        Number of trajectories tagged.
    """
    traj_df = result.tables.get("trajectory_mapping", pd.DataFrame())
    if traj_df.empty:
        return 0

    items, _ = _resolve_source(source)
    # Build lookup: (tissue_id, trajectory_id) -> classification
    lookup: dict = {}
    multi = "tissue_id" in traj_df.columns
    for _, row in traj_df.iterrows():
        key = (row["tissue_id"], row["trajectory_id"]) if multi else (0, row["trajectory_id"])
        lookup[key] = row["classification"]

    n_tagged = 0
    for tissue_id, series in items:
        for traj in series.edge_trajectories.values():
            key = (tissue_id, traj.trajectory_id)
            cls = lookup.get(key)
            if cls == "central":
                traj.tags.add(tag_name)
                traj.tags.discard(peripheral_tag_name)
                n_tagged += 1
            elif cls == "peripheral":
                traj.tags.add(peripheral_tag_name)
                traj.tags.discard(tag_name)
                n_tagged += 1

    return n_tagged
