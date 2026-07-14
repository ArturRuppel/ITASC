#!/usr/bin/env python3
"""Generate the ITASC "how it works" stage diagrams and distribution boards.

One shared visual language, emitted in a light and a dark variant so Furo can
swap them with its theme toggle (via .only-light / .only-dark). Writes the SVGs
into docs/_static/diagrams/ and a side-by-side preview.html next to this script.

Run from anywhere: paths are derived from this file's location.
"""
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs/_static/diagrams")
PREVIEW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview.html")

SANS = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', 'Cascadia Code', Menlo, Consolas, monospace"

LIGHT = {
    "card_bg": "#ffffff", "card_ln": "#e4e9eb", "edge": "#8b959c", "edge_off": "#e0e6e8",
    "ghost_ln": "#c7ced2", "ghost_ink": "#a9b1b8", "sub": "#616b72",
    "roles": {
        "seg":     ("#2f6df0", "#e9f0fe", "#1c4fbf"),
        "track":   ("#0d9488", "#e2f4f1", "#0a6f66"),
        "grow":    ("#3f9142", "#e9f4e9", "#2f6f31"),
        "correct": ("#c07a12", "#fbf0db", "#8c580a"),
        "quant":   ("#7c5cd6", "#efeafb", "#5b3fb0"),
        "core":    ("#64707a", "#e9edef", "#414b53"),
        "io":      ("#98a2ab", "#f2f4f6", "#59636b"),
    },
}
DARK = {
    "card_bg": "#17191c", "card_ln": "#2b3036", "edge": "#7a848b", "edge_off": "#363c42",
    "ghost_ln": "#3d454c", "ghost_ink": "#727b82", "sub": "#99a2a9",
    "roles": {
        "seg":     ("#6ea0ff", "#16273f", "#accaff"),
        "track":   ("#2dd4bf", "#0e2d2a", "#82ebde"),
        "grow":    ("#69c06d", "#15291b", "#9edaa1"),
        "correct": ("#e0a94e", "#2d2413", "#f1cd8c"),
        "quant":   ("#b49cf0", "#221b36", "#cebdf8"),
        "core":    ("#8a97a0", "#20272c", "#bcc5cc"),
        "io":      ("#7f8890", "#1c2024", "#a2abb2"),
    },
}

ORIGIN_X, COLGAP, ORIGIN_Y, ROWGAP = 104, 190, 66, 98
NW, NH = 172, 52


def cx(col):
    return ORIGIN_X + col * COLGAP


def cy(row):
    return ORIGIN_Y + row * ROWGAP


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def border(n, tx, ty):
    w, h = n["w"] / 2, n["h"] / 2
    dx, dy = tx - n["x"], ty - n["y"]
    if dx == 0 and dy == 0:
        return n["x"], n["y"]
    s = 1.0 / max(abs(dx) / w, abs(dy) / h)
    L = math.hypot(dx, dy) * s
    g = min(7.0, L)
    return n["x"] + dx * s * (1 - g / L), n["y"] + dy * s * (1 - g / L)


def svg_open(W, H, alt):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" font-family="{SANS}" role="img" '
        f'aria-label="{esc(alt)}">'
    )


def markers(P):
    return (
        '<defs>'
        f'<marker id="ar" markerWidth="8" markerHeight="8" refX="5.6" refY="3" orient="auto">'
        f'<path d="M0,0 L6,3 L0,6 Z" fill="{P["edge"]}"/></marker>'
        f'<marker id="arf" markerWidth="8" markerHeight="8" refX="5.6" refY="3" orient="auto">'
        f'<path d="M0,0 L6,3 L0,6 Z" fill="{P["edge_off"]}"/></marker></defs>'
    )


def card(W, H, P):
    return (
        f'<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="14" '
        f'fill="{P["card_bg"]}" stroke="{P["card_ln"]}" stroke-width="1"/>'
    )


def node_svg(n, ln, bg, ink, sub_col, dash=""):
    x, y = n["x"] - n["w"] / 2, n["y"] - n["h"] / 2
    return (
        f'<rect x="{x:.0f}" y="{y:.0f}" width="{n["w"]}" height="{n["h"]}" rx="10" '
        f'fill="{bg}" stroke="{ln}" stroke-width="1.6"{dash}/>'
        f'<text x="{n["x"]:.0f}" y="{n["y"]-3:.0f}" text-anchor="middle" '
        f'font-size="15" font-weight="600" fill="{ink}">{esc(n["main"])}</text>'
        f'<text x="{n["x"]:.0f}" y="{n["y"]+14:.0f}" text-anchor="middle" '
        f'font-family="{MONO}" font-size="11" fill="{sub_col}">{esc(n["sub"])}</text>'
    )


def render(spec, P):
    W, H = spec["viewBox"]
    nodes = {}
    for nid, n in spec["nodes"].items():
        nodes[nid] = {
            "x": cx(n["col"]), "y": cy(n["row"]), "w": n.get("w", NW), "h": n.get("h", NH),
            "role": n["role"], "main": n["main"], "sub": n["sub"],
        }
    p = [svg_open(W, H, spec["alt"]), markers(P), card(W, H, P)]
    for e in spec["edges"]:
        a, b = nodes[e["a"]], nodes[e["b"]]
        x1, y1 = border(a, b["x"], b["y"])
        x2, y2 = border(b, a["x"], a["y"])
        dash = ' stroke-dasharray="5 4"' if e.get("dashed") else ""
        p.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{P["edge"]}" stroke-width="1.6"{dash} marker-end="url(#ar)"/>'
        )
        if e.get("label"):
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            p.append(
                f'<rect x="{mx-e["lw"]/2:.1f}" y="{my-9:.1f}" width="{e["lw"]}" height="16" '
                f'rx="4" fill="{P["card_bg"]}"/>'
                f'<text x="{mx:.1f}" y="{my+3:.1f}" text-anchor="middle" '
                f'font-family="{MONO}" font-size="10.5" fill="{P["sub"]}">{esc(e["label"])}</text>'
            )
    for n in nodes.values():
        ln, bg, ink = P["roles"][n["role"]]
        p.append(node_svg(n, ln, bg, ink, P["sub"]))
    p.append("</svg>")
    return "".join(p)


# ── Stage diagram specs (geometry + text, palette-independent) ────────────────

DIAGRAMS = {
    "overview": {
        "viewBox": (968, 226),
        "alt": "The ITASC pipeline: input maps, nucleus tracking, cell bodies, "
               "contacts and aggregation, with correction on the tracking and cell stages.",
        "nodes": {
            "input":    {"col": 0, "row": 0, "role": "seg",   "main": "Input maps",      "sub": "Cellpose → fg + contour"},
            "nucleus":  {"col": 1, "row": 0, "role": "track", "main": "Nucleus tracking", "sub": "Ultrack: outline + track"},
            "cell":     {"col": 2, "row": 0, "role": "grow",  "main": "Cell bodies",      "sub": "grow from nuclei"},
            "contact":  {"col": 3, "row": 0, "role": "quant", "main": "Contacts + T1",    "sub": "who touches / swaps"},
            "aggregate":{"col": 4, "row": 0, "role": "quant", "main": "Aggregate",        "sub": "pooled tidy tables"},
            "corr":     {"col": 1.5, "row": 1, "role": "correct", "w": 236, "main": "Correction", "sub": "a person, where it is wrong"},
        },
        "edges": [
            {"a": "input", "b": "nucleus"}, {"a": "nucleus", "b": "cell"},
            {"a": "cell", "b": "contact"}, {"a": "contact", "b": "aggregate"},
            {"a": "nucleus", "b": "corr"}, {"a": "cell", "b": "corr"},
        ],
    },
    "input-maps": {
        "viewBox": (778, 324),
        "alt": "One frame through Cellpose into a brightness output and a direction "
               "output, which become the foreground map and the contour map.",
        "nodes": {
            "raw":      {"col": 0, "row": 1, "role": "io",    "main": "Raw frame",     "sub": "one channel"},
            "cellpose": {"col": 1, "row": 1, "role": "seg",   "main": "Cellpose",      "sub": "reads one image"},
            "bright":   {"col": 2, "row": 0, "role": "io",    "main": "Brightness",    "sub": "cell-like?"},
            "direction":{"col": 2, "row": 2, "role": "io",    "main": "Direction",     "sub": "toward the centre"},
            "fg":       {"col": 3, "row": 0, "role": "seg",   "main": "Foreground map", "sub": "where cells are"},
            "contour":  {"col": 3, "row": 2, "role": "seg",   "main": "Contour map",   "sub": "where boundaries run"},
        },
        "edges": [
            {"a": "raw", "b": "cellpose"},
            {"a": "cellpose", "b": "bright"}, {"a": "cellpose", "b": "direction"},
            {"a": "bright", "b": "fg"}, {"a": "direction", "b": "contour"},
        ],
    },
    "nucleus-tracking": {
        "viewBox": (968, 236),
        "alt": "Nucleus maps become atoms, then candidates, then a solve, then tracked "
               "nuclei; correction feeds back into the solve.",
        "nodes": {
            "maps":       {"col": 0, "row": 0, "role": "seg",     "main": "Nucleus maps", "sub": "foreground + contour"},
            "atoms":      {"col": 1, "row": 0, "role": "track",   "main": "Atoms",        "sub": "fragments < nucleus"},
            "candidates": {"col": 2, "row": 0, "role": "track",   "main": "Candidates",   "sub": "nested merges"},
            "solve":      {"col": 3, "row": 0, "role": "track",   "main": "Solve",        "sub": "ILP: consistent in time"},
            "tracked":    {"col": 4, "row": 0, "role": "track",   "main": "Tracked nuclei", "sub": "outline + identity"},
            "corr":       {"col": 2.5, "row": 1, "role": "correct", "w": 252, "main": "Correction", "sub": "validate · candidates · anchor"},
        },
        "edges": [
            {"a": "maps", "b": "atoms"}, {"a": "atoms", "b": "candidates"},
            {"a": "candidates", "b": "solve"}, {"a": "solve", "b": "tracked"},
            {"a": "tracked", "b": "corr"},
            {"a": "corr", "b": "solve", "dashed": True, "label": "re-solve", "lw": 58},
        ],
    },
    "cell-segmentation": {
        "viewBox": (778, 286),
        "alt": "Cell maps make a cost field; tracked nuclei are the seeds; a geodesic "
               "grow produces cell bodies, then correction.",
        "nodes": {
            "maps":   {"col": 0, "row": 0, "role": "seg",   "main": "Cell maps",     "sub": "foreground + contour"},
            "cost":   {"col": 1, "row": 0, "role": "grow",  "main": "Cost field",    "sub": "ridges cost to cross"},
            "nuclei": {"col": 1, "row": 1, "role": "track", "main": "Tracked nuclei", "sub": "the seeds"},
            "grow":   {"col": 2, "row": 0.5, "role": "grow", "main": "Geodesic grow", "sub": "nearest nucleus wins"},
            "bodies": {"col": 3, "row": 0.5, "role": "grow", "main": "Cell bodies",   "sub": "one per nucleus"},
            "corr":   {"col": 2.5, "row": 1.6, "role": "correct", "w": 214, "main": "Correction", "sub": "EpiCure · fill · clean"},
        },
        "edges": [
            {"a": "maps", "b": "cost"}, {"a": "cost", "b": "grow"},
            {"a": "nuclei", "b": "grow"}, {"a": "grow", "b": "bodies"},
            {"a": "bodies", "b": "corr"},
        ],
    },
    "contact-analysis": {
        "viewBox": (968, 226),
        "alt": "Tracked labels give a contact graph and T1 events, written per position "
               "to an HDF5 file, then pooled by the aggregator into tidy CSVs.",
        "nodes": {
            "labels":   {"col": 0, "row": 0.5, "role": "track", "main": "Tracked labels", "sub": "cells + nuclei"},
            "edges":    {"col": 1, "row": 0,   "role": "quant", "main": "Contact graph", "sub": "who touches whom"},
            "t1":       {"col": 1, "row": 1,   "role": "quant", "main": "T1 events",     "sub": "neighbour swaps"},
            "h5":       {"col": 2, "row": 0.5, "role": "io", "w": 186, "main": "contact_analysis.h5", "sub": "per position"},
            "aggregate":{"col": 3, "row": 0.5, "role": "quant", "main": "Aggregator",    "sub": "pool across positions"},
            "csv":      {"col": 4, "row": 0.5, "role": "io",    "main": "Tidy CSVs",     "sub": "one per quantity"},
        },
        "edges": [
            {"a": "labels", "b": "edges"}, {"a": "labels", "b": "t1"},
            {"a": "edges", "b": "h5"}, {"a": "t1", "b": "h5"},
            {"a": "h5", "b": "aggregate"}, {"a": "aggregate", "b": "csv"},
        ],
    },
}

# ── Distribution "parts bin" board ────────────────────────────────────────────

BOARD_NODES = {
    "maps":       (112, 54,  "seg",     "Cellpose → maps",   "foreground + contour"),
    "ultrack":    (308, 54,  "track",   "Ultrack",           "segment + track"),
    "cellseg":    (500, 54,  "grow",    "Cell bodies",       "grow from nucleus"),
    "contact":    (706, 54,  "quant",   "Contact analysis",  "edges · T1 events"),
    "aggregator": (890, 54,  "quant",   "Aggregator",        "pool → CSV"),
    "ultrackcorr":(308, 140, "correct", "Ultrack correction", "candidates · validate"),
    "epicure":    (500, 140, "correct", "EpiCure editing",   "hand corrections"),
    "masks":      (308, 226, "seg",     "Cellpose → masks",  "instance masks"),
    "laptrack":   (500, 226, "track",   "LapTrack",          "link over time"),
    "core":       (501, 290, "core",    "Core",              "project folder + napari UI", 920, 38),
}
BOARD_EDGES = [
    ("maps", "ultrack"), ("ultrack", "cellseg"), ("cellseg", "contact"),
    ("contact", "aggregator"), ("masks", "laptrack"),
    ("ultrack", "ultrackcorr"), ("ultrack", "epicure"),
    ("cellseg", "epicure"), ("laptrack", "epicure"),
]
MEMBERS = {
    "parts-bin": list(BOARD_NODES),
    "all":       ["core", "maps", "ultrack", "cellseg", "ultrackcorr", "epicure", "contact", "aggregator"],
    "cellpose":  ["core", "masks", "laptrack", "epicure"],
    "tracking":  ["core", "ultrack", "ultrackcorr", "epicure"],
    "aggregate": ["core", "contact", "aggregator"],
    "core":      ["core"],
}


def render_board(members, P):
    S = set(members)
    nb = {}
    for nid, spec in BOARD_NODES.items():
        x, y, role, main, sub = spec[:5]
        w = spec[5] if len(spec) > 5 else 168
        h = spec[6] if len(spec) > 6 else 54
        nb[nid] = {"x": x, "y": y, "w": w, "h": h, "role": role, "main": main, "sub": sub}

    W, H = 1002, 320
    p = [
        svg_open(W, H, "Which ingredients this distribution ships, on the shared ITASC board."),
        markers(P), card(W, H, P),
    ]
    for a, b in BOARD_EDGES:
        na, nbb = nb[a], nb[b]
        active = a in S and b in S
        x1, y1 = border(na, nbb["x"], nbb["y"])
        x2, y2 = border(nbb, na["x"], na["y"])
        col, mk = (P["edge"], "ar") if active else (P["edge_off"], "arf")
        p.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{col}" stroke-width="1.6" marker-end="url(#{mk})"/>'
        )
    for nid, n in nb.items():
        if nid in S:
            ln, bg, ink = P["roles"][n["role"]]
            p.append(node_svg(n, ln, bg, ink, P["sub"]))
        else:
            p.append(node_svg(n, P["ghost_ln"], "none", P["ghost_ink"], P["ghost_ink"],
                              dash=' stroke-dasharray="3 4"'))
    p.append("</svg>")
    return "".join(p)


# ── Generate both themes ──────────────────────────────────────────────────────

os.makedirs(OUT, exist_ok=True)
light_cards, dark_cards = [], []
for suffix, P in ((".light", LIGHT), ("-dark", DARK)):
    tag = "" if suffix == ".light" else suffix
    bucket = light_cards if suffix == ".light" else dark_cards
    for name, spec in DIAGRAMS.items():
        svg = render(spec, P)
        with open(os.path.join(OUT, f"diagram-{name}{tag}.svg"), "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n' + svg + "\n")
        bucket.append(f'<figure>{svg}</figure>')
    for key, mem in MEMBERS.items():
        svg = render_board(mem, P)
        base = "diagram-parts" if key == "parts-bin" else f"diagram-distro-{key}"
        with open(os.path.join(OUT, f"{base}{tag}.svg"), "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n' + svg + "\n")
        bucket.append(f'<figure>{svg}</figure>')
    print(f"wrote {'light' if not tag else 'dark'} set")

with open(PREVIEW, "w") as f:
    f.write(
        "<title>ITASC diagrams — light &amp; dark</title>"
        f"<style>body{{margin:0;font-family:{SANS}}}"
        "section{padding:30px}h2{font-weight:600;font-size:18px;margin:0 0 18px}"
        "figure{margin:0 0 22px}svg{width:100%;height:auto;max-width:960px;display:block}"
        ".light{background:#eef1f1;color:#191d21}.dark{background:#131416;color:#e6e9ea}</style>"
        "<section class='light'><h2>Light</h2>" + "".join(light_cards) + "</section>"
        "<section class='dark'><h2>Dark</h2>" + "".join(dark_cards) + "</section>"
    )
print("wrote preview.html")
