"""Render the premade SuperPlots inside a CellFlow ``.iris`` bundle to static
figures (PNG + SVG) via the Iris engine.

A ``.iris`` already carries everything a figure needs — the tidy table, its
schema, and the premade analysis specs (:mod:`.analyses`). Rendering is therefore
a pure function of the bundle: hand each spec to the engine's FastAPI-free render
core and save the matplotlib figure. The engine is an *optional* dependency
(``cellflow[plots]``); importing it is deferred to call time so the rest of the
package — including the no-plots export path — never needs it.

Text is kept **editable** (real ``<text>`` elements / embedded TrueType, not
outlined paths) by saving under an rc-context that sets ``svg.fonttype='none'``
and ``pdf.fonttype=42``; matplotlib reads these at save time, and the engine's own
save helpers don't pin them, so the context here takes effect.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

__all__ = ["render_iris", "render_export_dir", "DEFAULT_FORMATS"]

#: Default output formats: raster preview + editable vector.
DEFAULT_FORMATS = ("png", "svg")

#: rcParams that keep saved text editable (selectable / re-typesettable in
#: Illustrator / Inkscape) rather than outlined to paths. Read by matplotlib at
#: savefig time.
_EDITABLE_TEXT_RC = {"svg.fonttype": "none", "pdf.fonttype": 42}


def _require_engine():
    """Import the Iris engine, or raise a pointed install hint."""
    try:
        from iris_engine import compiler, document, render
    except ImportError as exc:  # pragma: no cover - exercised via the import hint
        raise ImportError(
            "Rendering .iris figures requires the Iris engine, an optional "
            "dependency. Install it with:  pip install 'cellflow[plots]'  "
            "(for local dev against a source checkout:  "
            "pip install -e /path/to/Iris/engine)."
        ) from exc
    return compiler, document, render


def _slug(text: str) -> str:
    """A filesystem-safe slug for an analysis title/id (keeps it readable)."""
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    text = re.sub(r"[\s_]+", "-", text)
    return re.sub(r"-{2,}", "-", text).strip("-") or "analysis"


def render_iris(
    iris_path: Path | str,
    out_dir: Path | str,
    *,
    formats: Sequence[str] = DEFAULT_FORMATS,
) -> list[Path]:
    """Render every premade analysis in one ``.iris`` to *out_dir*.

    Figures land in ``<out_dir>/<iris_stem>/NN-<title-slug>.<fmt>`` (numbered in
    bundle order so the on-disk order matches the bundle's). Returns the written
    paths. Raises :class:`ImportError` with an install hint if the engine is
    absent; an engine ``RenderError`` for a malformed spec propagates.
    """
    compiler, document, render = _require_engine()
    # Force a non-interactive backend before pyplot is configured, so batch runs
    # never try to open a window.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iris_path = Path(iris_path)
    doc = document.load_document(iris_path.read_bytes())
    table = {"schema": doc["schema"], "rows": doc["rows"]}

    dest = Path(out_dir) / iris_path.stem
    dest.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for index, spec in enumerate(doc["analyses"], start=1):
        label = spec.get("title") or spec.get("id") or f"analysis-{index}"
        slug = f"{index:02d}-{_slug(str(label))}"
        fig, *_ = render.render(table, spec)
        try:
            with plt.rc_context(_EDITABLE_TEXT_RC):
                for fmt in formats:
                    out_path = dest / f"{slug}.{fmt}"
                    if fmt == "svg":
                        out_path.write_text(compiler.figure_to_svg(fig))
                    else:
                        out_path.write_bytes(compiler.figure_to_bytes(fig, fmt))
                    written.append(out_path)
        finally:
            compiler.close(fig)
    return written


def render_export_dir(
    iris_dir: Path | str,
    out_dir: Path | str,
    *,
    formats: Sequence[str] = DEFAULT_FORMATS,
) -> list[Path]:
    """Render every ``*.iris`` under *iris_dir* (what :func:`pipeline.export`
    wrote into ``<export>/iris``) to *out_dir*. Returns all written paths."""
    iris_dir = Path(iris_dir)
    written: list[Path] = []
    for iris_path in sorted(iris_dir.glob("*.iris")):
        written.extend(render_iris(iris_path, out_dir, formats=formats))
    return written
