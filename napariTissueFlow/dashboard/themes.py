"""Dashboard color themes.

Each theme defines CSS custom properties and a matching Plotly template.
Adding a new theme = adding a new dict here.
"""
from __future__ import annotations

from typing import Dict

import plotly.graph_objects as go

FONT_STACK = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #rrggbb to rgba(r, g, b, alpha)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ------------------------------------------------------------------
# Theme definitions
# ------------------------------------------------------------------

THEMES: Dict[str, dict] = {
    "Midnight": {
        "base": "#1e1e2e",
        "mantle": "#181825",
        "crust": "#11111b",
        "surface0": "#313244",
        "surface1": "#45475a",
        "surface2": "#585b70",
        "text": "#cdd6f4",
        "subtext": "#a6adc8",
        "overlay0": "#6c7086",
        "accent": "#b4befe",
        "blue": "#89b4fa",
        "green": "#a6e3a1",
        "yellow": "#f9e2af",
        "red": "#f38ba8",
        "colorway": ["#89b4fa", "#a6e3a1", "#f9e2af", "#f38ba8",
                      "#cba6f7", "#94e2d5", "#fab387", "#74c7ec"],
        "chart_annotation_bg": "rgba(49,50,68,0.85)",
        "chart_annotation_border": "#585b70",
    },
    "Ocean": {
        "base": "#0f1923",
        "mantle": "#0a1218",
        "crust": "#060d12",
        "surface0": "#1a2d3d",
        "surface1": "#264052",
        "surface2": "#325368",
        "text": "#d4e4f1",
        "subtext": "#94b8d4",
        "overlay0": "#5f87a3",
        "accent": "#5eaad4",
        "blue": "#5eaad4",
        "green": "#6ec89b",
        "yellow": "#e8c76a",
        "red": "#e06c75",
        "colorway": ["#5eaad4", "#6ec89b", "#e8c76a", "#e06c75",
                      "#c678dd", "#56d4c8", "#d19a66", "#61afef"],
        "chart_annotation_bg": "rgba(26,45,61,0.85)",
        "chart_annotation_border": "#325368",
    },
    "Slate": {
        "base": "#1b1f27",
        "mantle": "#161920",
        "crust": "#10131a",
        "surface0": "#272c36",
        "surface1": "#343a47",
        "surface2": "#414858",
        "text": "#dce1ea",
        "subtext": "#a3abbe",
        "overlay0": "#6b7590",
        "accent": "#8fa7e6",
        "blue": "#8fa7e6",
        "green": "#8fbf8f",
        "yellow": "#d4b06a",
        "red": "#cf7d7d",
        "colorway": ["#8fa7e6", "#8fbf8f", "#d4b06a", "#cf7d7d",
                      "#b497d6", "#7dc4c4", "#c9956b", "#7dafd4"],
        "chart_annotation_bg": "rgba(39,44,54,0.85)",
        "chart_annotation_border": "#414858",
    },
    "Light": {
        "base": "#ffffff",
        "mantle": "#f5f5f7",
        "crust": "#ebebef",
        "surface0": "#e4e4e8",
        "surface1": "#d0d0d6",
        "surface2": "#b8b8c0",
        "text": "#1e1e2e",
        "subtext": "#4c4f69",
        "overlay0": "#7c7f93",
        "accent": "#1e66f5",
        "blue": "#1e66f5",
        "green": "#40a02b",
        "yellow": "#df8e1d",
        "red": "#d20f39",
        "colorway": ["#1e66f5", "#40a02b", "#df8e1d", "#d20f39",
                      "#8839ef", "#179299", "#fe640b", "#04a5e5"],
        "chart_annotation_bg": "rgba(228,228,232,0.85)",
        "chart_annotation_border": "#b8b8c0",
    },
}

DEFAULT_THEME = "Midnight"


def get_theme(name: str) -> dict:
    """Return a theme dict by name, falling back to default."""
    return THEMES.get(name, THEMES[DEFAULT_THEME])


def build_plotly_template(theme: dict) -> go.layout.Template:
    """Create a Plotly layout template from a theme dict."""
    return go.layout.Template(
        layout=go.Layout(
            font=dict(family=FONT_STACK, size=13, color=theme["text"]),
            paper_bgcolor=theme["base"],
            plot_bgcolor=theme["base"],
            title=dict(font=dict(size=16, color=theme["text"])),
            xaxis=dict(
                gridcolor=theme["surface0"], zerolinecolor=theme["surface1"],
                title_font=dict(color=theme["subtext"]),
                tickfont=dict(color=theme["subtext"]),
            ),
            yaxis=dict(
                gridcolor=theme["surface0"], zerolinecolor=theme["surface1"],
                title_font=dict(color=theme["subtext"]),
                tickfont=dict(color=theme["subtext"]),
            ),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=theme["text"])),
            colorway=theme["colorway"],
            margin=dict(l=60, r=30, t=50, b=50),
        ),
    )


def css_variables(theme: dict) -> str:
    """Generate a CSS custom property block from a theme dict.

    Includes pre-computed rgba variants for status backgrounds since
    ``rgba(from ...)`` relative color syntax is not widely supported.
    """
    lines = [":root {"]
    for key in ["base", "mantle", "crust", "surface0", "surface1", "surface2",
                "text", "subtext", "overlay0", "accent", "blue", "green",
                "yellow", "red"]:
        lines.append(f"  --tg-{key}: {theme[key]};")

    # Pre-computed status background colors
    for name, key in [("info", "blue"), ("success", "green"),
                      ("warning", "yellow"), ("danger", "red")]:
        lines.append(f"  --tg-status-{name}-bg: {_hex_to_rgba(theme[key], 0.12)};")

    lines.append("}")
    return "\n".join(lines)
