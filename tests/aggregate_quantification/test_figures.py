"""Rendering CellFlow ``.iris`` bundles to static figures via the Iris engine.

These tests need the optional ``cellflow[plots]`` dependency (the engine), so the
whole module is skipped when it is absent — a no-engine checkout still runs green.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("iris_engine")

from cellflow.aggregate_quantification.iris_export import (  # noqa: E402
    render_export_dir,
    render_iris,
)
from cellflow.aggregate_quantification.iris_export.export import (  # noqa: E402
    export_table_frame,
)

_POSITIONS = {"VimKO": ("pos00", "pos01"), "WT": ("pos02", "pos03")}
_DATES = ("2026/04/01", "2026/04/02", "2026/04/03")


def _cells_table(n_cells: int = 6, n_frames: int = 2) -> pd.DataFrame:
    """A small ``cell_shape``-shaped table rich enough for a real SuperPlot:
    two conditions × three dates × two positions × cells × frames."""
    rng = np.random.default_rng(0)
    rows = []
    for condition in ("VimKO", "WT"):
        for date in _DATES:
            for position_id in _POSITIONS[condition]:
                for cell_id in range(1, n_cells + 1):
                    label = "epithelial" if cell_id % 2 else "mesenchymal"
                    for frame in range(n_frames):
                        rows.append({
                            "condition": condition,
                            "date": date,
                            "position_id": position_id,
                            "frame": frame,
                            "cell_id": cell_id,
                            "cell_shape.area_um2": float(rng.uniform(100, 500)),
                            "class_label": label,
                        })
    return pd.DataFrame(rows)


def _write_iris(tmp_path):
    iris_dir = tmp_path / "iris"
    return export_table_frame(_cells_table(), "cell_shape", iris_dir)


def test_render_iris_writes_png_and_svg(tmp_path):
    iris_path = _write_iris(tmp_path)

    written = render_iris(iris_path, tmp_path / "figures")

    assert written, "expected at least one rendered figure"
    pngs = [p for p in written if p.suffix == ".png"]
    svgs = [p for p in written if p.suffix == ".svg"]
    assert pngs and svgs
    # Every figure lands under figures/<iris_stem>/ and is non-empty.
    for path in written:
        assert path.parent.name == "cell_shape"
        assert path.is_file() and path.stat().st_size > 0


def test_rendered_svg_keeps_editable_text(tmp_path):
    """The export must keep real, selectable text (svg.fonttype='none'), not
    outline glyphs to paths — so labels stay editable in Illustrator/Inkscape."""
    iris_path = _write_iris(tmp_path)

    written = render_iris(iris_path, tmp_path / "figures", formats=("svg",))

    svg = written[0].read_text()
    assert "<text" in svg  # real text elements, not <path> outlines


def test_render_export_dir_covers_every_bundle(tmp_path):
    _write_iris(tmp_path)  # one .iris under tmp_path/iris

    written = render_export_dir(tmp_path / "iris", tmp_path / "figures")

    assert any(p.suffix == ".png" for p in written)
    assert any(p.suffix == ".svg" for p in written)


def test_format_selection_is_honored(tmp_path):
    iris_path = _write_iris(tmp_path)

    written = render_iris(iris_path, tmp_path / "figures", formats=("png",))

    assert written and all(p.suffix == ".png" for p in written)
