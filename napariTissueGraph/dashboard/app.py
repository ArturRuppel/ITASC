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
from typing import Any, Dict, List, Optional, Type

import pandas as pd
from dash import Dash, Input, Output, State, clientside_callback, dash_table, dcc, html, no_update
import plotly.graph_objects as go
import plotly.io as pio

from ..analysis.modules import (
    AnalysisModule,
    AnalysisResult,
    ParamType,
    Parameter,
    discover_modules,
)
from ..core.io import load_dataset
from ..structures import TissueGraphDataset
from .themes import (
    DEFAULT_THEME,
    FONT_STACK,
    THEMES,
    build_plotly_template,
    css_variables,
    get_theme,
)

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
                html.Div(id="metadata-pane", style={"marginBottom": "20px"}),
                html.Div(id="results-area"),
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

            ds = load_dataset(Path(dataset_path))
            mod = modules[module_name]()
            param_specs = mod.parameters()
            params = _extract_params_from_children(param_specs, param_children)

            result = mod.compute(ds, **params)
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

    # Auto-load on startup
    if dataset_path:
        app.layout.children[0].data = dataset_path

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
