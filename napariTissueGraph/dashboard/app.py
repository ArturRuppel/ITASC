"""TissueGraph analysis dashboard built with Dash + Plotly.

Launch standalone::

    python -m napariTissueGraph.dashboard          # serves at localhost:8050
    python -m napariTissueGraph.dashboard /path/to/dataset

Use in Jupyter::

    from napariTissueGraph.dashboard import create_app
    app = create_app()
    app.run(jupyter_mode="inline")
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Type

import pandas as pd
from dash import Dash, Input, Output, State, callback_context, clientside_callback, dash_table, dcc, html, no_update
import plotly.graph_objects as go
import plotly.io as pio

from ..analysis.modules import (
    AnalysisModule,
    AnalysisResult,
    ParamType,
    Parameter,
    discover_modules,
)
from ..analysis.tagging import (
    tag_junction,
    tag_trajectory,
    untag_junction,
    untag_trajectory,
    get_all_tags,
)
from ..core.io import load_dataset, save_dataset
from ..structures import TissueGraphDataset
from .themes import (
    DEFAULT_THEME,
    FONT_STACK,
    THEMES,
    build_plotly_template,
    css_variables,
    get_theme,
)

# Server-side dataset cache (avoids reloading from disk on every callback)
_dataset_cache: Dict[str, TissueGraphDataset] = {}

logger = logging.getLogger(__name__)

# Register all Plotly templates at import time
for tname, tdef in THEMES.items():
    pio.templates[f"tg-{tname}"] = build_plotly_template(tdef)
pio.templates.default = f"tg-{DEFAULT_THEME}"

# ------------------------------------------------------------------
# Static CSS (uses CSS custom properties set by theme)
# ------------------------------------------------------------------

STYLESHEET = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* {{ box-sizing: border-box; }}
body {{
    margin: 0; padding: 0;
    font-family: {FONT_STACK};
    background: var(--tg-crust);
    color: var(--tg-text);
}}

/* Sidebar */
.tg-sidebar {{
    width: 340px; padding: 24px;
    background: var(--tg-mantle);
    overflow-y: auto; height: 100vh; flex-shrink: 0;
    border-right: 1px solid var(--tg-surface0);
}}

/* Main */
.tg-main {{
    flex: 1; padding: 28px 32px;
    overflow-y: auto; height: 100vh;
    background: var(--tg-base);
}}

/* Section headers */
.tg-section-title {{
    margin: 0 0 12px 0; font-size: 0.8em;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--tg-overlay0); font-weight: 600;
}}

/* Buttons */
.tg-btn {{
    width: 100%; padding: 10px 16px; border: none; border-radius: 8px;
    font-weight: 600; font-size: 0.9em; cursor: pointer;
    font-family: {FONT_STACK}; letter-spacing: 0.01em;
    transition: opacity 0.15s, filter 0.15s;
}}
.tg-btn:hover {{ filter: brightness(1.1); }}
.tg-btn:disabled {{ opacity: 0.5; cursor: not-allowed; filter: none; }}
.tg-btn-primary {{ background: var(--tg-blue); color: var(--tg-crust); }}
.tg-btn-success {{ background: var(--tg-green); color: var(--tg-crust); }}

/* Divider */
.tg-divider {{ border-top: 1px solid var(--tg-surface0); margin: 0 0 24px 0; }}

/* Inputs */
input[type="text"] {{
    background: var(--tg-surface0) !important;
    border: 1px solid var(--tg-surface1) !important;
    color: var(--tg-text) !important;
    border-radius: 6px !important; padding: 8px 12px !important;
    font-family: {FONT_STACK} !important;
}}
input[type="text"]::placeholder {{ color: var(--tg-overlay0) !important; }}
input[type="text"]:focus {{ border-color: var(--tg-blue) !important; outline: none !important; }}

/* Dropdown overrides */
.Select-control, .Select-menu-outer {{
    background: var(--tg-surface0) !important;
    border-color: var(--tg-surface1) !important;
    color: var(--tg-text) !important;
}}
.Select-value-label, .Select-placeholder {{ color: var(--tg-text) !important; }}
.Select-option {{ background: var(--tg-surface0) !important; color: var(--tg-text) !important; }}
.Select-option.is-focused {{ background: var(--tg-surface1) !important; }}

/* Slider overrides */
.rc-slider-track {{ background: var(--tg-blue) !important; }}
.rc-slider-handle {{ border-color: var(--tg-blue) !important; background: var(--tg-blue) !important; }}
.rc-slider-rail {{ background: var(--tg-surface1) !important; }}

/* Param labels */
.tg-param-label {{ font-weight: 500; font-size: 0.85em; color: var(--tg-text); margin-bottom: 4px; display: block; }}
.tg-param-help {{ font-size: 0.75em; color: var(--tg-overlay0); margin: 4px 0 14px 0; line-height: 1.3; }}

/* Status bar — uses pre-computed rgba vars from theme */
.tg-status {{
    padding: 12px 16px; margin-bottom: 20px; border-radius: 8px;
    font-size: 0.9em; font-family: {FONT_STACK};
}}
.tg-status-info {{ background: var(--tg-status-info-bg); color: var(--tg-blue); border: 1px solid var(--tg-blue); }}
.tg-status-success {{ background: var(--tg-status-success-bg); color: var(--tg-green); border: 1px solid var(--tg-green); }}
.tg-status-warning {{ background: var(--tg-status-warning-bg); color: var(--tg-yellow); border: 1px solid var(--tg-yellow); }}
.tg-status-danger {{ background: var(--tg-status-danger-bg); color: var(--tg-red); border: 1px solid var(--tg-red); }}

/* Metadata cards */
.tg-meta-card {{
    padding: 10px 16px; background: var(--tg-surface0);
    border-radius: 8px; min-width: 120px;
}}
.tg-meta-label {{ color: var(--tg-overlay0); font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.04em; }}
.tg-meta-value {{ color: var(--tg-text); font-size: 1.1em; font-weight: 600; }}

/* Table header label */
.tg-table-title {{
    margin-top: 28px; margin-bottom: 10px; color: var(--tg-subtext);
    font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;
}}

/* Info rows */
.tg-info-label {{ color: var(--tg-overlay0); }}
.tg-info-value {{ color: var(--tg-text); }}

/* Dataset info */
.tg-dataset-name {{ font-weight: 600; color: var(--tg-text); margin-bottom: 6px; }}

/* All labels and text inside sidebar */
.tg-sidebar label {{ color: var(--tg-text); }}

/* Theme picker */
.tg-theme-row {{
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 24px;
}}
.tg-theme-row label {{
    font-size: 0.78em; color: var(--tg-overlay0);
    font-weight: 500; white-space: nowrap;
}}

/* ---- Dash DataTable overrides ---- */
.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table {{
    border-collapse: collapse !important;
}}
.dash-table-container .cell-table,
.dash-table-container .dash-cell-value {{
    font-family: {FONT_STACK} !important;
    color: var(--tg-subtext) !important;
}}
/* Table header cells */
.dash-table-container th {{
    background-color: var(--tg-surface0) !important;
    color: var(--tg-text) !important;
    border-color: var(--tg-surface1) !important;
}}
/* Table data cells */
.dash-table-container td {{
    background-color: var(--tg-base) !important;
    color: var(--tg-subtext) !important;
    border-color: var(--tg-surface0) !important;
}}
/* Table filter row inputs */
.dash-table-container input.dash-filter--case--sensitive,
.dash-table-container input {{
    background-color: var(--tg-surface0) !important;
    color: var(--tg-text) !important;
    border-color: var(--tg-surface1) !important;
}}
/* Table active cell */
.dash-table-container td.focused {{
    background-color: var(--tg-surface1) !important;
    outline-color: var(--tg-blue) !important;
}}
/* Table pagination */
.dash-table-container .previous-next-container button,
.dash-table-container .page-number {{
    color: var(--tg-text) !important;
    background-color: var(--tg-surface0) !important;
    border-color: var(--tg-surface1) !important;
    font-family: {FONT_STACK} !important;
}}
.dash-table-container .current-page-container input {{
    background-color: var(--tg-surface0) !important;
    color: var(--tg-text) !important;
    border-color: var(--tg-surface1) !important;
}}
.dash-table-container .current-page-container,
.dash-table-container .page-number {{
    color: var(--tg-subtext) !important;
}}

/* ---- Dash Dropdown deep overrides ---- */
.Select-arrow-zone,
.Select-clear-zone {{ color: var(--tg-overlay0) !important; }}
.Select-input input {{ color: var(--tg-text) !important; }}
.Select-noresults {{ color: var(--tg-overlay0) !important; background: var(--tg-surface0) !important; }}

/* ---- Slider tooltip ---- */
.rc-slider-tooltip-inner {{
    background-color: var(--tg-surface0) !important;
    color: var(--tg-text) !important;
    border-color: var(--tg-surface1) !important;
    font-family: {FONT_STACK} !important;
    box-shadow: none !important;
}}
.rc-slider-tooltip-arrow {{
    border-top-color: var(--tg-surface0) !important;
}}

/* ---- Checklist ---- */
.tg-sidebar input[type="checkbox"] {{
    accent-color: var(--tg-blue);
}}

/* ---- Plotly modebar ---- */
.modebar-btn {{ color: var(--tg-overlay0) !important; }}
.modebar-btn:hover {{ color: var(--tg-text) !important; }}
.modebar {{ background: transparent !important; }}

/* ---- Tabs ---- */
.tg-tabs .tab {{
    background: var(--tg-mantle) !important;
    border: 1px solid var(--tg-surface0) !important;
    border-bottom: none !important;
    color: var(--tg-overlay0) !important;
    padding: 10px 20px !important;
    font-family: {FONT_STACK} !important;
    font-weight: 500 !important;
    font-size: 0.9em !important;
}}
.tg-tabs .tab--selected {{
    background: var(--tg-base) !important;
    color: var(--tg-text) !important;
    border-bottom: 2px solid var(--tg-blue) !important;
}}

/* ---- Edge Viewer ---- */
.ev-controls {{
    display: flex; gap: 16px; align-items: flex-end;
    margin-bottom: 16px; flex-wrap: wrap;
}}
.ev-controls label {{ font-size: 0.82em; color: var(--tg-overlay0); font-weight: 500; display: block; margin-bottom: 4px; }}
.ev-split {{
    display: flex; gap: 20px; margin-top: 12px;
}}
.ev-map {{ flex: 1; min-width: 0; }}
.ev-panel {{ flex: 1; min-width: 0; }}
.ev-details {{
    background: var(--tg-surface0); border-radius: 8px;
    padding: 14px 18px; margin-top: 16px; font-size: 0.88em;
    color: var(--tg-subtext); line-height: 1.6;
}}
.ev-details strong {{ color: var(--tg-text); }}
.ev-ops {{
    display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap;
}}
.ev-ops input {{
    width: 70px;
}}
.tg-btn-sm {{
    padding: 7px 14px; border: none; border-radius: 6px;
    font-weight: 500; font-size: 0.82em; cursor: pointer;
    font-family: {FONT_STACK}; letter-spacing: 0.01em;
    transition: opacity 0.15s, filter 0.15s;
}}
.tg-btn-sm:hover {{ filter: brightness(1.1); }}
.tg-btn-danger {{ background: var(--tg-red); color: var(--tg-crust); }}
.tg-btn-secondary {{ background: var(--tg-surface1); color: var(--tg-text); }}
"""


# ------------------------------------------------------------------
# Parameter → Dash component mapping
# ------------------------------------------------------------------


def _make_input(p: Parameter) -> html.Div:
    """Create a Dash input component from an analysis module Parameter."""
    input_id = {"type": "param-input", "name": p.name}

    if p.type == ParamType.INT:
        component = dcc.Slider(
            id=input_id,
            min=int(p.min) if p.min is not None else 0,
            max=int(p.max) if p.max is not None else 100,
            step=int(p.step) if p.step is not None else 1,
            value=p.default or 0,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": True},
        )
    elif p.type == ParamType.FLOAT:
        component = dcc.Slider(
            id=input_id,
            min=float(p.min) if p.min is not None else 0.0,
            max=float(p.max) if p.max is not None else 1.0,
            step=float(p.step) if p.step is not None else 0.01,
            value=p.default or 0.0,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": True},
        )
    elif p.type == ParamType.BOOL:
        component = dcc.Checklist(
            id=input_id,
            options=[{"label": " Yes", "value": "on"}],
            value=["on"] if p.default else [],
            inline=True,
        )
    elif p.type == ParamType.CHOICE:
        component = dcc.Dropdown(
            id=input_id,
            options=[{"label": c, "value": c} for c in (p.choices or [])],
            value=p.default,
            clearable=False,
        )
    elif p.type in (ParamType.STR, ParamType.TAG):
        component = dcc.Input(
            id=input_id,
            type="text",
            value=str(p.default) if p.default is not None else "",
            placeholder=p.description,
            style={"width": "100%"},
        )
    else:
        component = dcc.Input(
            id=input_id,
            type="text",
            value=str(p.default) if p.default is not None else "",
            style={"width": "100%"},
        )

    return html.Div([
        html.Label(p.label, className="tg-param-label"),
        component,
        html.P(p.description, className="tg-param-help"),
    ])


def _read_widget_value(p: Parameter, raw_value: Any) -> Any:
    """Convert a raw Dash widget value back to the type expected by the module."""
    if p.type == ParamType.BOOL:
        return bool(raw_value and "on" in raw_value)
    if p.type == ParamType.INT and raw_value is not None:
        return int(raw_value)
    if p.type == ParamType.FLOAT and raw_value is not None:
        return float(raw_value)
    return raw_value


# ------------------------------------------------------------------
# Edge Viewer helpers
# ------------------------------------------------------------------


def _get_cached_dataset(path: str) -> Optional[TissueGraphDataset]:
    """Return dataset from cache, loading if needed."""
    if not path:
        return None
    if path not in _dataset_cache:
        try:
            _dataset_cache[path] = load_dataset(Path(path))
        except Exception:
            return None
    return _dataset_cache[path]


_EDGE_COLORWAY = [
    "#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8",
    "#cba6f7", "#94e2d5", "#fab387", "#74c7ec",
    "#f5c2e7", "#b4befe", "#eba0ac", "#89dceb",
]

_TAG_COLORWAY = [
    "#f38ba8", "#a6e3a1", "#89b4fa", "#fab387",
    "#cba6f7", "#94e2d5", "#f9e2af", "#f5c2e7",
]


def _build_tissue_map(ds: TissueGraphDataset, tissue_id: int, frame_idx: int,
                      selected_edge: Optional[Tuple[int, int]] = None,
                      color_mode: str = "uniform",
                      theme_name: str = DEFAULT_THEME) -> go.Figure:
    """Build a Plotly figure showing cell polygons and junction lines for one frame."""
    theme = get_theme(theme_name)
    fig = go.Figure()

    if tissue_id not in ds.tissues:
        return fig

    series = ds.tissues[tissue_id]
    if frame_idx not in series.frames:
        return fig

    frame = series.frames[frame_idx]

    # Draw cell polygons (if vertices available)
    for cell in frame.cells.values():
        if cell.vertices is not None and len(cell.vertices) >= 3:
            verts = cell.vertices
            ys = list(verts[:, 0]) + [verts[0, 0]]
            xs = list(verts[:, 1]) + [verts[0, 1]]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=theme["surface2"], width=1),
                hoverinfo="skip", showlegend=False,
            ))
        fig.add_trace(go.Scatter(
            x=[cell.position[1]], y=[cell.position[0]],
            mode="text", text=[str(cell.cell_id)],
            textfont=dict(size=9, color=theme["overlay0"]),
            hoverinfo="skip", showlegend=False,
        ))

    # Build color assignments based on mode
    # Build trajectory lookup for color-by-trajectory
    traj_lookup: Dict[FrozenSet[int], int] = {}
    for traj in series.edge_trajectories.values():
        for i, fi in enumerate(traj.frames):
            if fi == frame_idx:
                traj_lookup[frozenset(traj.cell_pairs[i])] = traj.trajectory_id

    # Collect all tags for color-by-tag
    all_tags_sorted: List[str] = []
    if color_mode == "tag":
        tag_set: Set[str] = set()
        for jd in frame.junctions.values():
            tag_set.update(jd.tags)
            traj_id = traj_lookup.get(frozenset(jd.cell_pair))
            if traj_id is not None and traj_id in series.edge_trajectories:
                tag_set.update(series.edge_trajectories[traj_id].tags)
        all_tags_sorted = sorted(tag_set)
    tag_color_map = {t: _TAG_COLORWAY[i % len(_TAG_COLORWAY)] for i, t in enumerate(all_tags_sorted)}

    # Unique trajectory IDs for coloring
    unique_traj_ids = sorted(set(traj_lookup.values()))
    traj_color_map = {tid: _EDGE_COLORWAY[i % len(_EDGE_COLORWAY)] for i, tid in enumerate(unique_traj_ids)}

    # Number junctions for click identification
    n_bg_traces = len(fig.data)  # cell polygons + labels drawn above

    for jd in frame.junctions.values():
        if len(jd.coordinates) < 2:
            continue
        coords = jd.coordinates
        is_selected = (selected_edge is not None
                       and tuple(sorted(selected_edge)) == tuple(sorted(jd.cell_pair)))

        # Determine color
        edge_key_fs = frozenset(jd.cell_pair)
        merged_tags: Set[str] = set(jd.tags)
        traj_id = traj_lookup.get(edge_key_fs)
        if traj_id is not None and traj_id in series.edge_trajectories:
            merged_tags |= series.edge_trajectories[traj_id].tags

        if is_selected:
            color = "#ffffff"
            width = 5
        elif color_mode == "trajectory" and traj_id is not None:
            color = traj_color_map.get(traj_id, theme["accent"])
            width = 2.5
        elif color_mode == "tag" and merged_tags:
            first_tag = sorted(merged_tags)[0]
            color = tag_color_map.get(first_tag, theme["accent"])
            width = 2.5
        else:
            color = theme["accent"]
            width = 2

        tag_str = ", ".join(sorted(merged_tags)) if merged_tags else ""
        hover = (f"Edge ({jd.cell_pair[0]}, {jd.cell_pair[1]})<br>"
                 f"Length: {jd.length:.1f}<br>"
                 f"Tags: {tag_str or 'none'}")

        fig.add_trace(go.Scatter(
            x=coords[:, 1].tolist(), y=coords[:, 0].tolist(),
            mode="lines", line=dict(color=color, width=width),
            opacity=1.0 if is_selected else 0.8,
            hovertext=hover, hoverinfo="text",
            showlegend=False,
            customdata=[[jd.cell_pair[0], jd.cell_pair[1]]],
        ))

    fig.update_layout(
        xaxis=dict(scaleanchor="y", constrain="domain", showgrid=False,
                   zeroline=False, showticklabels=False),
        yaxis=dict(autorange="reversed", showgrid=False,
                   zeroline=False, showticklabels=False),
        margin=dict(l=5, r=5, t=5, b=5),
        height=500,
        dragmode="pan",
        clickmode="event",
        paper_bgcolor=theme["base"],
        plot_bgcolor=theme["base"],
    )
    return fig, n_bg_traces


def _build_edge_table_data(ds: TissueGraphDataset, tissue_id: int,
                           frame_idx: int) -> List[Dict[str, Any]]:
    """Build edge table rows for one frame."""
    if tissue_id not in ds.tissues:
        return []
    series = ds.tissues[tissue_id]
    if frame_idx not in series.frames:
        return []

    frame = series.frames[frame_idx]

    # Build trajectory lookup for this frame
    traj_lookup: Dict[FrozenSet[int], Tuple[int, str]] = {}
    for traj in series.edge_trajectories.values():
        for i, fi in enumerate(traj.frames):
            if fi == frame_idx:
                key = frozenset(traj.cell_pairs[i])
                traj_lookup[key] = (traj.trajectory_id, traj.name or "")

    rows = []
    for edge_key, jd in frame.junctions.items():
        traj_id, traj_name = traj_lookup.get(frozenset(jd.cell_pair), (-1, ""))
        merged_tags: Set[str] = set(jd.tags)
        if traj_id != -1 and traj_id in series.edge_trajectories:
            merged_tags |= series.edge_trajectories[traj_id].tags
        rows.append({
            "cell_a": jd.cell_pair[0],
            "cell_b": jd.cell_pair[1],
            "length": round(jd.length, 2),
            "tags": ", ".join(sorted(merged_tags)) if merged_tags else "",
            "traj_id": traj_id if traj_id != -1 else "",
            "name": traj_name,
            "n_coords": len(jd.coordinates),
        })
    return rows


def _build_edge_viewer_layout() -> html.Div:
    """Build the Edge Viewer tab layout."""
    arrow_btn_style = {
        "padding": "4px 10px", "border": "1px solid var(--tg-surface1)",
        "borderRadius": "6px", "background": "var(--tg-surface0)",
        "color": "var(--tg-text)", "cursor": "pointer", "fontSize": "1.1em",
        "lineHeight": "1", "fontFamily": FONT_STACK,
    }
    return html.Div([
        # Hidden stores for tissue navigation
        dcc.Store(id="ev-tissue-index", data=0),
        dcc.Store(id="ev-tissue-ids", data=[]),
        dcc.Store(id="ev-n-bg-traces", data=0),

        # Navigation controls
        html.Div([
            # Tissue arrows
            html.Div([
                html.Label("Tissue", style={"marginBottom": "4px"}),
                html.Div([
                    html.Button("\u25c0", id="ev-tissue-prev", n_clicks=0, style=arrow_btn_style),
                    html.Span(id="ev-tissue-label", children="--",
                              style={"minWidth": "80px", "textAlign": "center",
                                     "fontWeight": "600", "color": "var(--tg-text)",
                                     "fontSize": "0.92em"}),
                    html.Button("\u25b6", id="ev-tissue-next", n_clicks=0, style=arrow_btn_style),
                ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
            ]),
            # Frame arrows + slider
            html.Div([
                html.Label("Frame", style={"marginBottom": "4px"}),
                html.Div([
                    html.Button("\u25c0", id="ev-frame-prev", n_clicks=0, style=arrow_btn_style),
                    html.Div([
                        dcc.Slider(id="ev-frame-slider", min=0, max=0, step=1, value=0,
                                   marks=None, updatemode="drag",
                                   tooltip={"placement": "bottom", "always_visible": True}),
                    ], style={"flex": "1", "minWidth": "180px"}),
                    html.Button("\u25b6", id="ev-frame-next", n_clicks=0, style=arrow_btn_style),
                ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
            ], style={"flex": "1"}),
            # Color mode
            html.Div([
                html.Label("Color", style={"marginBottom": "4px"}),
                dcc.Dropdown(id="ev-color-mode", options=[
                    {"label": "Uniform", "value": "uniform"},
                    {"label": "By trajectory", "value": "trajectory"},
                    {"label": "By tag", "value": "tag"},
                ], value="uniform", clearable=False, style={"width": "150px"}),
            ]),
        ], className="ev-controls"),

        # Info bar
        html.Div(id="ev-info", style={
            "fontSize": "0.85em", "color": "var(--tg-subtext)", "marginBottom": "12px",
        }),

        # Split: map + panel
        html.Div([
            # Left: tissue map
            html.Div([
                dcc.Graph(id="ev-tissue-map", figure=go.Figure(),
                          config={"displayModeBar": True, "displaylogo": False,
                                  "scrollZoom": True},
                          style={"height": "500px"}),
            ], className="ev-map"),

            # Right: edge table + details + operations
            html.Div([
                dash_table.DataTable(
                    id="ev-edge-table",
                    columns=[
                        {"name": "Cell A", "id": "cell_a"},
                        {"name": "Cell B", "id": "cell_b"},
                        {"name": "Length", "id": "length"},
                        {"name": "Tags", "id": "tags"},
                        {"name": "Traj ID", "id": "traj_id"},
                        {"name": "Name", "id": "name"},
                    ],
                    data=[],
                    row_selectable="multi",
                    selected_rows=[],
                    sort_action="native",
                    filter_action="native",
                    page_size=12,
                    style_table={"overflowX": "auto", "borderRadius": "8px"},
                    style_cell={
                        "textAlign": "left", "padding": "6px 10px",
                        "minWidth": "60px", "fontFamily": FONT_STACK,
                        "fontSize": "0.85em",
                    },
                ),

                # Edge details
                html.Div(id="ev-edge-details", className="ev-details",
                         children="Select an edge to see details."),

                # Tag operations
                html.Div([
                    html.Label("Tag operations", style={
                        "fontSize": "0.82em", "color": "var(--tg-overlay0)",
                        "fontWeight": "500", "marginBottom": "6px", "display": "block",
                    }),
                    html.Div([
                        dcc.Input(id="ev-tag-input", type="text",
                                  placeholder="Tag name...",
                                  style={"width": "140px", "marginRight": "6px"}),
                        html.Button("Tag", id="ev-tag-btn", n_clicks=0,
                                    className="tg-btn-sm tg-btn-primary"),
                        html.Button("Untag", id="ev-untag-btn", n_clicks=0,
                                    className="tg-btn-sm tg-btn-secondary",
                                    style={"marginLeft": "4px"}),
                    ], style={"display": "flex", "alignItems": "center", "gap": "4px"}),
                ], style={"marginTop": "14px"}),

                # Save
                html.Div([
                    html.Button("Save to dataset", id="ev-save-btn", n_clicks=0,
                                className="tg-btn-sm tg-btn-primary"),
                ], style={"marginTop": "14px"}),

                # Edge viewer status
                html.Div(id="ev-status", style={
                    "marginTop": "10px", "fontSize": "0.82em",
                    "color": "var(--tg-overlay0)",
                }),
            ], className="ev-panel"),
        ], className="ev-split"),
    ])


# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------


def create_app(dataset_path: Optional[str] = None) -> Dash:
    """Create and return the Dash app."""
    modules = discover_modules()
    module_options = [
        {"label": modules[n]().description, "value": n}
        for n in sorted(modules.keys())
    ]

    # Serialize all theme CSS variable blocks for the clientside callback
    theme_css_map = {name: css_variables(t) for name, t in THEMES.items()}

    app = Dash(
        __name__,
        title="TissueGraph Analysis",
        suppress_callback_exceptions=True,
    )

    default_theme = get_theme(DEFAULT_THEME)
    default_css_vars = css_variables(default_theme)

    app.index_string = f"""<!DOCTYPE html>
<html>
<head>
    {{%metas%}}
    <title>{{%title%}}</title>
    {{%favicon%}}
    {{%css%}}
    <style id="tg-theme-vars">{default_css_vars}</style>
    <style>{STYLESHEET}</style>
</head>
<body>
    {{%app_entry%}}
    <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>"""

    # ----- Layout -----
    app.layout = html.Div([
        dcc.Store(id="dataset-store"),
        dcc.Store(id="theme-store", data=DEFAULT_THEME),
        dcc.Store(id="theme-css-map", data=theme_css_map),

        html.Div([
            # ===== Sidebar =====
            html.Div([
                # Logo
                html.Div([
                    html.H2("TissueGraph", style={
                        "margin": "0", "fontWeight": "700",
                        "color": "var(--tg-accent)", "letterSpacing": "-0.02em",
                    }),
                    html.P("Analysis Dashboard", style={
                        "margin": "2px 0 0 0", "fontSize": "0.8em",
                        "color": "var(--tg-overlay0)", "fontWeight": "400",
                    }),
                ], style={"marginBottom": "20px"}),

                # Theme picker
                html.Div([
                    html.Label("Theme"),
                    dcc.Dropdown(
                        id="theme-select",
                        options=[{"label": n, "value": n} for n in THEMES],
                        value=DEFAULT_THEME,
                        clearable=False,
                        style={"flex": "1"},
                    ),
                ], className="tg-theme-row"),

                # Dataset section
                html.Div([
                    html.H4("Dataset", className="tg-section-title"),
                    dcc.Input(
                        id="dataset-path-input",
                        type="text",
                        placeholder="Path to saved dataset...",
                        value=dataset_path or "",
                        style={"width": "100%", "marginBottom": "10px"},
                    ),
                    html.Button("Load dataset", id="load-btn", n_clicks=0,
                                className="tg-btn tg-btn-primary"),
                    html.Div(id="dataset-info", style={
                        "marginTop": "14px", "fontSize": "0.82em",
                        "color": "var(--tg-subtext)", "lineHeight": "1.5",
                    }),
                ], style={"marginBottom": "24px"}),

                html.Div(className="tg-divider"),

                # Analysis section
                html.Div([
                    html.H4("Analysis", className="tg-section-title"),
                    dcc.Dropdown(
                        id="module-select",
                        options=module_options,
                        value=module_options[0]["value"] if module_options else None,
                        clearable=False,
                        style={"marginBottom": "16px"},
                    ),
                    html.Div(id="param-container"),
                    html.Button("Run analysis", id="run-btn", n_clicks=0, disabled=True,
                                className="tg-btn tg-btn-success",
                                style={"marginTop": "8px"}),
                ]),

            ], className="tg-sidebar"),

            # ===== Main area =====
            html.Div([
                html.Div("Load a dataset to get started.",
                         id="status-bar", className="tg-status tg-status-info"),

                dcc.Tabs(id="main-tabs", value="analysis", className="tg-tabs", children=[
                    dcc.Tab(label="Analysis", value="analysis", className="tab", selected_className="tab--selected", children=[
                        html.Div([
                            html.Div(id="metadata-pane", style={"marginBottom": "20px"}),
                            html.Div(id="results-area"),
                        ], style={"paddingTop": "16px"}),
                    ]),
                    dcc.Tab(label="Edge Viewer", value="edge-viewer", className="tab", selected_className="tab--selected", children=[
                        html.Div([
                            _build_edge_viewer_layout(),
                        ], style={"paddingTop": "16px"}),
                    ]),
                ]),

            ], className="tg-main"),

        ], style={"display": "flex", "height": "100vh"}),
    ], style={"fontFamily": FONT_STACK})

    # ----- Theme switching (clientside for instant response) -----
    clientside_callback(
        """
        function(themeName, cssMap) {
            var css = cssMap[themeName];
            if (css) {
                var el = document.getElementById('tg-theme-vars');
                if (el) el.textContent = css;
            }
            return themeName;
        }
        """,
        Output("theme-store", "data"),
        Input("theme-select", "value"),
        State("theme-css-map", "data"),
    )

    # ----- Callbacks -----

    @app.callback(
        Output("dataset-store", "data"),
        Output("dataset-info", "children"),
        Output("status-bar", "children"),
        Output("status-bar", "className"),
        Output("run-btn", "disabled"),
        Input("load-btn", "n_clicks"),
        State("dataset-path-input", "value"),
        prevent_initial_call=True,
    )
    def load_dataset_cb(n_clicks, path):
        if not path or not path.strip():
            return (
                no_update, no_update,
                "Enter a dataset path first.",
                "tg-status tg-status-warning",
                True,
            )
        path = path.strip()
        try:
            ds = load_dataset(Path(path))
            _dataset_cache[path] = ds
            n_tissues = len(ds.tissue_ids)
            total_frames = sum(ds.tissues[tid].num_frames for tid in ds.tissue_ids)
            info = html.Div([
                html.Div(Path(path).name, className="tg-dataset-name"),
                html.Div([
                    _info_row("Condition", ds.condition or "(none)"),
                    _info_row("Tissues", str(n_tissues)),
                    _info_row("Total frames", str(total_frames)),
                    _info_row("Pixel size", str(ds.pixel_size or "?")),
                    _info_row("Time interval", str(ds.time_interval or "?")),
                ]),
            ])
            return (
                path, info,
                f"Loaded dataset from {Path(path).name}",
                "tg-status tg-status-success",
                False,
            )
        except Exception as exc:
            return (
                no_update,
                html.Span(f"Error: {exc}", style={"color": "var(--tg-red)"}),
                f"Failed to load: {exc}",
                "tg-status tg-status-danger",
                True,
            )

    @app.callback(
        Output("param-container", "children"),
        Input("module-select", "value"),
    )
    def update_params(module_name):
        if not module_name or module_name not in modules:
            return html.P("Select a module", style={"color": "var(--tg-overlay0)"})
        mod = modules[module_name]()
        params = mod.parameters()
        if not params:
            return html.P("No parameters", style={"color": "var(--tg-overlay0)"})
        return html.Div([_make_input(p) for p in params])

    @app.callback(
        Output("results-area", "children"),
        Output("metadata-pane", "children"),
        Output("status-bar", "children", allow_duplicate=True),
        Output("status-bar", "className", allow_duplicate=True),
        Input("run-btn", "n_clicks"),
        State("dataset-store", "data"),
        State("module-select", "value"),
        State("param-container", "children"),
        State("theme-store", "data"),
        prevent_initial_call=True,
    )
    def run_analysis(n_clicks, dataset_path, module_name, param_children, theme_name):
        if not dataset_path or not module_name:
            return no_update, no_update, "Load a dataset first.", "tg-status tg-status-warning"

        try:
            # Apply current theme's Plotly template
            pio.templates.default = f"tg-{theme_name}"

            # Use cached dataset so analysis results can be applied to it
            ds = _get_cached_dataset(dataset_path)
            if ds is None:
                ds = load_dataset(Path(dataset_path))
                _dataset_cache[dataset_path] = ds
            mod = modules[module_name]()
            param_specs = mod.parameters()
            params = _extract_params_from_children(param_specs, param_children)

            result = mod.compute(ds, **params)

            # Apply central junction tags to cached dataset when that module is run
            if module_name == "central_junction_identifier":
                from ..analysis.builtins.central_junction_identifier import apply_central_tags
                n_tagged = apply_central_tags(
                    ds, result,
                    tag_name=params.get("tag_name", "central"),
                    peripheral_tag_name=params.get("peripheral_tag_name", "peripheral"),
                )
                logger.info("Applied central junction tags to %d trajectories", n_tagged)

            figs = mod.visualize(result, **params)

            results_children = []

            # Figures
            for fig in figs:
                results_children.append(
                    dcc.Graph(
                        figure=fig,
                        style={"marginBottom": "24px"},
                        config={"displayModeBar": True, "displaylogo": False},
                    )
                )

            # Tables
            theme = get_theme(theme_name)
            for tname, tdf in result.tables.items():
                if tdf.empty:
                    continue
                display_df = tdf.head(500).copy()
                for col in display_df.select_dtypes(include="float").columns:
                    display_df[col] = display_df[col].round(4)

                results_children.append(
                    html.Div([
                        html.H4(tname, className="tg-table-title"),
                        dash_table.DataTable(
                            data=display_df.to_dict("records"),
                            columns=[{"name": c, "id": c} for c in display_df.columns],
                            page_size=15,
                            sort_action="native",
                            filter_action="native",
                            style_table={"overflowX": "auto", "borderRadius": "8px"},
                            style_cell={
                                "textAlign": "left", "padding": "8px 12px",
                                "minWidth": "80px", "fontFamily": FONT_STACK,
                            },
                            style_header={
                                "backgroundColor": theme["surface0"],
                                "color": theme["text"],
                                "fontWeight": "600",
                                "border": f"1px solid {theme['surface1']}",
                                "fontFamily": FONT_STACK,
                            },
                            style_data={
                                "backgroundColor": theme["base"],
                                "color": theme["subtext"],
                                "border": f"1px solid {theme['surface0']}",
                                "fontSize": "0.85em",
                            },
                            style_filter={
                                "backgroundColor": theme["surface0"],
                                "color": theme["text"],
                                "border": f"1px solid {theme['surface1']}",
                            },
                            style_data_conditional=[{
                                "if": {"state": "active"},
                                "backgroundColor": theme["surface1"],
                                "border": f"1px solid {theme['surface2']}",
                            }],
                        ),
                    ])
                )

            # Metadata cards
            metadata_div = html.Div()
            if result.metadata:
                items = []
                for k, v in result.metadata.items():
                    val_str = f"{v:.4g}" if isinstance(v, float) else str(v)
                    items.append(
                        html.Div([
                            html.Span(k, className="tg-meta-label"),
                            html.Div(val_str, className="tg-meta-value"),
                        ], className="tg-meta-card")
                    )
                metadata_div = html.Div(items, style={
                    "display": "flex", "gap": "12px", "flexWrap": "wrap",
                    "marginBottom": "24px",
                })

            return (
                html.Div(results_children),
                metadata_div,
                "Analysis complete.",
                "tg-status tg-status-success",
            )

        except Exception as exc:
            logger.exception("Analysis failed")
            return (
                no_update, no_update,
                f"Analysis failed: {exc}",
                "tg-status tg-status-danger",
            )

    # ----- Edge Viewer callbacks -----

    # Tissue arrow navigation: update tissue index store
    @app.callback(
        Output("ev-tissue-ids", "data"),
        Input("dataset-store", "data"),
    )
    def ev_load_tissue_ids(dataset_path):
        if not dataset_path:
            return []
        ds = _get_cached_dataset(dataset_path)
        if ds is None:
            return []
        return ds.tissue_ids

    @app.callback(
        Output("ev-tissue-index", "data"),
        Input("ev-tissue-prev", "n_clicks"),
        Input("ev-tissue-next", "n_clicks"),
        Input("ev-tissue-ids", "data"),
        State("ev-tissue-index", "data"),
        prevent_initial_call=True,
    )
    def ev_navigate_tissue(prev_clicks, next_clicks, tissue_ids, current_idx):
        ctx = callback_context
        if not ctx.triggered or not tissue_ids:
            return 0
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if triggered_id == "ev-tissue-prev":
            return max(0, (current_idx or 0) - 1)
        elif triggered_id == "ev-tissue-next":
            return min(len(tissue_ids) - 1, (current_idx or 0) + 1)
        # tissue_ids changed — reset to 0
        return 0

    @app.callback(
        Output("ev-tissue-label", "children"),
        Input("ev-tissue-index", "data"),
        State("ev-tissue-ids", "data"),
    )
    def ev_update_tissue_label(idx, tissue_ids):
        if not tissue_ids or idx is None or idx >= len(tissue_ids):
            return "--"
        return f"Tissue {tissue_ids[idx]} ({idx + 1}/{len(tissue_ids)})"

    # Frame slider range updates when tissue changes
    @app.callback(
        Output("ev-frame-slider", "min"),
        Output("ev-frame-slider", "max"),
        Output("ev-frame-slider", "value"),
        Input("ev-tissue-index", "data"),
        State("ev-tissue-ids", "data"),
        State("dataset-store", "data"),
    )
    def ev_update_frame_slider(idx, tissue_ids, dataset_path):
        if not tissue_ids or idx is None or idx >= len(tissue_ids) or not dataset_path:
            return 0, 0, 0
        tissue_id = tissue_ids[idx]
        ds = _get_cached_dataset(dataset_path)
        if ds is None or tissue_id not in ds.tissues:
            return 0, 0, 0
        frames = ds.tissues[tissue_id].frame_indices
        if not frames:
            return 0, 0, 0
        return frames[0], frames[-1], frames[0]

    # Frame arrow navigation
    @app.callback(
        Output("ev-frame-slider", "value", allow_duplicate=True),
        Input("ev-frame-prev", "n_clicks"),
        Input("ev-frame-next", "n_clicks"),
        State("ev-frame-slider", "value"),
        State("ev-frame-slider", "min"),
        State("ev-frame-slider", "max"),
        prevent_initial_call=True,
    )
    def ev_navigate_frame(prev_clicks, next_clicks, current, fmin, fmax):
        ctx = callback_context
        if not ctx.triggered:
            return no_update
        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if triggered_id == "ev-frame-prev":
            return max(fmin, (current or 0) - 1)
        elif triggered_id == "ev-frame-next":
            return min(fmax, (current or 0) + 1)
        return no_update

    # Main view update: map + table + info
    @app.callback(
        Output("ev-tissue-map", "figure"),
        Output("ev-edge-table", "data"),
        Output("ev-edge-table", "selected_rows"),
        Output("ev-info", "children"),
        Output("ev-n-bg-traces", "data"),
        Input("ev-frame-slider", "value"),
        Input("ev-tissue-index", "data"),
        Input("ev-color-mode", "value"),
        State("ev-tissue-ids", "data"),
        State("dataset-store", "data"),
        State("theme-store", "data"),
    )
    def ev_update_view(frame_idx, tissue_idx, color_mode, tissue_ids, dataset_path, theme_name):
        if not tissue_ids or tissue_idx is None or tissue_idx >= len(tissue_ids) or not dataset_path:
            return go.Figure(), [], [], "", 0
        tissue_id = tissue_ids[tissue_idx]
        ds = _get_cached_dataset(dataset_path)
        if ds is None or tissue_id not in ds.tissues:
            return go.Figure(), [], [], "", 0

        series = ds.tissues[tissue_id]
        valid_frames = series.frame_indices
        if frame_idx not in series.frames and valid_frames:
            frame_idx = min(valid_frames, key=lambda f: abs(f - frame_idx))

        fig, n_bg = _build_tissue_map(
            ds, tissue_id, frame_idx,
            color_mode=color_mode or "uniform",
            theme_name=theme_name or DEFAULT_THEME,
        )
        table_data = _build_edge_table_data(ds, tissue_id, frame_idx)

        n_edges = len(table_data)
        n_tagged = sum(1 for r in table_data if r["tags"])
        info = f"Tissue {tissue_id}, Frame {frame_idx}: {n_edges} edges, {n_tagged} tagged"

        return fig, table_data, [], info, n_bg

    # Click on plot edge -> select in table
    @app.callback(
        Output("ev-edge-table", "selected_rows", allow_duplicate=True),
        Input("ev-tissue-map", "clickData"),
        State("ev-edge-table", "data"),
        State("ev-n-bg-traces", "data"),
        prevent_initial_call=True,
    )
    def ev_click_edge(click_data, table_data, n_bg_traces):
        if not click_data or not table_data:
            return no_update
        points = click_data.get("points", [])
        if not points:
            return no_update
        point = points[0]
        curve_idx = point.get("curveNumber", -1)
        # Edge traces start after n_bg_traces background traces
        edge_idx = curve_idx - (n_bg_traces or 0)
        if edge_idx < 0 or edge_idx >= len(table_data):
            return no_update
        return [edge_idx]

    # Edge details
    @app.callback(
        Output("ev-edge-details", "children"),
        Input("ev-edge-table", "selected_rows"),
        State("ev-edge-table", "data"),
        State("ev-tissue-index", "data"),
        State("ev-tissue-ids", "data"),
        State("dataset-store", "data"),
    )
    def ev_show_details(selected_rows, table_data, tissue_idx, tissue_ids, dataset_path):
        if not selected_rows or not table_data:
            return "Select an edge to see details."
        tissue_id = tissue_ids[tissue_idx] if tissue_ids and tissue_idx is not None and tissue_idx < len(tissue_ids) else None
        ds = _get_cached_dataset(dataset_path) if dataset_path else None
        details = []
        for row_idx in selected_rows:
            if row_idx >= len(table_data):
                continue
            row = table_data[row_idx]
            cell_a, cell_b = row["cell_a"], row["cell_b"]
            parts = [
                html.Div([html.Strong(f"Edge ({cell_a}, {cell_b})")]),
                html.Div(f"Length: {row['length']} px"),
                html.Div(f"Coordinates: {row['n_coords']} points"),
                html.Div(f"Tags: {row['tags'] or 'none'}"),
            ]
            traj_id = row.get("traj_id", "")
            if traj_id != "" and traj_id != -1 and ds is not None and tissue_id is not None and tissue_id in ds.tissues:
                series = ds.tissues[tissue_id]
                if int(traj_id) in series.edge_trajectories:
                    traj = series.edge_trajectories[int(traj_id)]
                    parts.append(html.Div(
                        f"Trajectory: #{traj_id} ({len(traj.frames)} frames, "
                        f"{len(traj.t1_events)} T1 events)"
                    ))
            if row.get("name"):
                parts.append(html.Div(f"Name: {row['name']}"))
            details.append(html.Div(parts, style={"marginBottom": "8px"}))
        return details

    # Tag / untag
    @app.callback(
        Output("ev-edge-table", "data", allow_duplicate=True),
        Output("ev-tissue-map", "figure", allow_duplicate=True),
        Output("ev-n-bg-traces", "data", allow_duplicate=True),
        Output("ev-status", "children", allow_duplicate=True),
        Input("ev-tag-btn", "n_clicks"),
        Input("ev-untag-btn", "n_clicks"),
        State("ev-tag-input", "value"),
        State("ev-edge-table", "selected_rows"),
        State("ev-edge-table", "data"),
        State("ev-tissue-index", "data"),
        State("ev-tissue-ids", "data"),
        State("ev-frame-slider", "value"),
        State("ev-color-mode", "value"),
        State("dataset-store", "data"),
        State("theme-store", "data"),
        prevent_initial_call=True,
    )
    def ev_tag_untag(tag_clicks, untag_clicks, tag_name, selected_rows,
                     table_data, tissue_idx, tissue_ids, frame_idx,
                     color_mode, dataset_path, theme_name):
        ctx = callback_context
        if not ctx.triggered or not selected_rows or not tag_name or not tag_name.strip():
            return no_update, no_update, no_update, "Enter a tag name and select edges first."

        if not tissue_ids or tissue_idx is None or tissue_idx >= len(tissue_ids):
            return no_update, no_update, no_update, "No tissue selected."
        tissue_id = tissue_ids[tissue_idx]

        triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
        is_tag = triggered_id == "ev-tag-btn"
        tag_name = tag_name.strip()

        ds = _get_cached_dataset(dataset_path)
        if ds is None or tissue_id not in ds.tissues:
            return no_update, no_update, no_update, "No dataset loaded."

        series = ds.tissues[tissue_id]
        frame = series.frames.get(frame_idx)
        if frame is None:
            return no_update, no_update, no_update, "Invalid frame."

        count = 0
        for row_idx in selected_rows:
            if row_idx >= len(table_data):
                continue
            row = table_data[row_idx]
            cell_pair = (int(row["cell_a"]), int(row["cell_b"]))
            traj_id = row.get("traj_id", "")

            if is_tag:
                tag_junction(frame, cell_pair, tag_name)
                if traj_id != "" and traj_id != -1 and int(traj_id) in series.edge_trajectories:
                    tag_trajectory(series, int(traj_id), tag_name)
            else:
                untag_junction(frame, cell_pair, tag_name)
                if traj_id != "" and traj_id != -1 and int(traj_id) in series.edge_trajectories:
                    untag_trajectory(series, int(traj_id), tag_name)
            count += 1

        action = "Tagged" if is_tag else "Untagged"
        new_table = _build_edge_table_data(ds, tissue_id, frame_idx)
        new_fig, n_bg = _build_tissue_map(
            ds, tissue_id, frame_idx,
            color_mode=color_mode or "uniform",
            theme_name=theme_name or DEFAULT_THEME,
        )
        return new_table, new_fig, n_bg, f"{action} {count} edge(s) as '{tag_name}'."

    # Save to dataset
    @app.callback(
        Output("ev-status", "children", allow_duplicate=True),
        Input("ev-save-btn", "n_clicks"),
        State("dataset-store", "data"),
        prevent_initial_call=True,
    )
    def ev_save_changes(n_clicks, dataset_path):
        if not dataset_path:
            return "No dataset loaded."
        ds = _get_cached_dataset(dataset_path)
        if ds is None:
            return "No dataset in cache."
        try:
            save_dataset(ds, Path(dataset_path))
            return "Changes saved to disk."
        except Exception as exc:
            return f"Save failed: {exc}"

    # Auto-load on startup
    if dataset_path:
        app.layout.children[0].data = dataset_path
        # Pre-populate cache
        _get_cached_dataset(dataset_path)

    return app


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _info_row(label: str, value: str) -> html.Div:
    return html.Div([
        html.Span(f"{label}: ", className="tg-info-label"),
        html.Span(value, className="tg-info-value"),
    ], style={"marginBottom": "2px"})


def _extract_params_from_children(
    param_specs: List[Parameter],
    children: Any,
) -> Dict[str, Any]:
    """Extract parameter values from Dash component tree."""
    values: Dict[str, Any] = {}
    spec_map = {p.name: p for p in param_specs}

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            props = node.get("props", {})
            node_id = props.get("id")
            if isinstance(node_id, dict) and node_id.get("type") == "param-input":
                pname = node_id["name"]
                if pname in spec_map:
                    raw = props.get("value")
                    values[pname] = _read_widget_value(spec_map[pname], raw)
            children = props.get("children")
            if children is not None:
                _walk(children)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(children)

    for p in param_specs:
        if p.name not in values:
            values[p.name] = p.default

    return values


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def serve(dataset_path: Optional[str] = None, port: int = 8050, show: bool = True) -> None:
    """Launch the dashboard as a standalone web app."""
    import webbrowser
    app = create_app(dataset_path=dataset_path)
    if show:
        import threading
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    app.run(debug=False, port=port)
