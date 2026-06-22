"""The Iris ``.iris`` exporter: schema typing, the SuperPlot template, and the
ZIP document format (the frozen contract with Iris)."""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cellflow.aggregate_quantification.iris_export import (
    build_analyses,
    export_dir,
    export_table,
    infer_schema,
    write_iris,
)
from cellflow.aggregate_quantification.iris_export.document import FORMAT_VERSION


# Each condition owns disjoint positions (a position belongs to one condition),
# so the spine date→position→cell_id is well-formed and condition nests at the
# position level — mirroring real cells_by_frame data.
_POSITIONS = {"VimKO": ("pos00", "pos01"), "WT": ("pos02", "pos03")}
_DATES = ("2026/04/01", "2026/04/02", "2026/04/03")  # ≥3 replicates for a test


def _cells_table(n_cells: int = 6, n_frames: int = 3,
                 conditions: tuple[str, ...] = ("VimKO", "WT")) -> pd.DataFrame:
    """A small ``cells_by_frame``-shaped table: conditions × 3 dates × positions
    × cells × frames, with a per-cell ``class_label`` and two descriptors."""
    rng = np.random.default_rng(0)
    rows = []
    for condition in conditions:
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
                            "cell_dynamics.speed_um_per_s": float(rng.uniform(0, 1)),
                            # prunable: a coordinate and a velocity component
                            "cell_shape.centroid_x_um": float(rng.uniform(0, 200)),
                            "cell_dynamics.vx_um_per_s": float(rng.uniform(-1, 1)),
                            "class_label": label,
                        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# schema typing
# --------------------------------------------------------------------------- #

def test_keys_typed_identifier_not_categorical():
    """date/position_id/cell_id must be identifiers so the date colour drives the
    per-point SuperPlot idiom instead of dodging the box."""
    schema = infer_schema(_cells_table())
    types = {c["name"]: c["type"] for c in schema["columns"]}
    assert types["date"] == "identifier"
    assert types["position_id"] == "identifier"
    assert types["cell_id"] == "identifier"
    assert types["frame"] == "identifier"


def test_axes_categorical_descriptors_numeric():
    schema = infer_schema(_cells_table())
    types = {c["name"]: c["type"] for c in schema["columns"]}
    assert types["condition"] == "categorical"
    assert types["class_label"] == "categorical"
    assert types["cell_shape.area_um2"] == "numeric"
    assert types["cell_dynamics.speed_um_per_s"] == "numeric"


def test_schema_label_unit_and_levels():
    schema = infer_schema(_cells_table())
    cols = {c["name"]: c for c in schema["columns"]}
    assert cols["cell_shape.area_um2"]["label"] == "area um2"
    assert cols["cell_shape.area_um2"]["unit"] == "µm²"
    assert cols["class_label"]["levels"] == ["epithelial", "mesenchymal"]


def test_schema_omits_id_but_keeps_other_columns():
    """Only the stable row ``id`` is bookkeeping. Iris removed its exclusion
    mechanism, so a boolean ``excluded`` column (if present) is an ordinary factor
    that must appear in the schema — not silently dropped."""
    df = _cells_table()
    df.insert(0, "id", [str(i) for i in range(len(df))])
    df["excluded"] = False
    names = {c["name"] for c in infer_schema(df)["columns"]}
    assert "id" not in names
    assert "excluded" in names


# --------------------------------------------------------------------------- #
# the SuperPlot template
# --------------------------------------------------------------------------- #

def test_one_superplot_per_axis_per_descriptor():
    df = _cells_table()
    analyses = build_analyses(df, infer_schema(df), object_key="cell_id")
    ids = {a["id"] for a in analyses}
    # 2 descriptors × {condition, class_label}
    assert ids == {
        "superplot__condition__cell_shape.area_um2",
        "superplot__condition__cell_dynamics.speed_um_per_s",
        "superplot__class_label__cell_shape.area_um2",
        "superplot__class_label__cell_dynamics.speed_um_per_s",
    }


def test_superplot_layers_levels_and_encodings():
    df = _cells_table()
    analyses = build_analyses(df, infer_schema(df), object_key="cell_id")
    spec = analyses[0]
    assert spec["encodings"] == {
        "x": {"column": "condition"}, "y": {"column": "cell_shape.area_um2"},
        "color": {"column": "date"}, "shape": {"column": "date"}, "size": None,
    }
    assert spec["id"] == "superplot__condition__cell_shape.area_um2"
    assert spec["hierarchy"]["spine"] == ["date", "position_id", "cell_id", "frame"]
    assert spec["stats"] == {}  # two conditions ==> unpinned test
    # violin for the per-cell distribution; swarm of per-date means on top. No
    # per-cell swarm, no notched box.
    geoms = [(layer["geom"], layer["level"]) for layer in spec["layers"]]
    assert geoms == [("violin", "cell_id"), ("dot", "date")]
    assert spec["layers"][1]["params"]["layout"] == "swarm"
    assert "notch" not in json.dumps(spec)


def test_single_level_axis_is_describe_only():
    # One condition ⇒ no test possible; the analysis must be describe-only so Iris
    # renders it instead of raising "needs at least 2 groups". Multi-level axes
    # stay unpinned.
    df = _cells_table(conditions=("VimKO",))
    analyses = build_analyses(df, infer_schema(df), object_key="cell_id")
    by_id = {a["id"]: a for a in analyses}
    assert by_id["superplot__condition__cell_shape.area_um2"]["stats"] == {
        "chosen_by": "describe_only"}
    assert by_id["superplot__class_label__cell_shape.area_um2"]["stats"] == {}


def test_axis_absent_from_table_is_skipped():
    df = _cells_table().drop(columns=["class_label"])
    analyses = build_analyses(df, infer_schema(df), object_key="cell_id")
    assert all("class_label" not in a["id"] for a in analyses)
    assert all(a["encodings"]["x"] == {"column": "condition"} for a in analyses)


def test_prune_drops_coordinate_and_velocity_columns():
    df = _cells_table()  # includes centroid_x_um and vx_um_per_s
    analyses = build_analyses(df, infer_schema(df), object_key="cell_id")
    plotted = {a["encodings"]["y"]["column"] for a in analyses}
    assert "cell_shape.centroid_x_um" not in plotted
    assert "cell_dynamics.vx_um_per_s" not in plotted
    assert "cell_shape.area_um2" in plotted


def test_title_carries_family_and_axis():
    df = _cells_table()
    titles = {a["id"]: a["title"]
              for a in build_analyses(df, infer_schema(df), object_key="cell_id")}
    assert titles["superplot__condition__cell_shape.area_um2"] == (
        "Cell shape · Area (µm²) — by condition")
    assert titles["superplot__class_label__cell_dynamics.speed_um_per_s"] == (
        "Cell motion · Speed — by class")


def test_analyses_grouped_by_family_order():
    df = _cells_table()
    families = [a["id"].split("__")[2].split(".")[0]
                for a in build_analyses(df, infer_schema(df), object_key="cell_id")]
    # the cell_shape block precedes the cell_dynamics block (family ordering)
    assert families == ["cell_shape", "cell_shape", "cell_dynamics", "cell_dynamics"]


# --------------------------------------------------------------------------- #
# the .iris document (frozen contract with Iris)
# --------------------------------------------------------------------------- #

def test_write_iris_zip_structure_and_roundtrip():
    df = _cells_table()
    df.insert(0, "id", [str(i) for i in range(len(df))])
    df["excluded"] = False
    schema = infer_schema(df)
    analyses = build_analyses(df, schema, object_key="cell_id")

    data = write_iris(df, schema, analyses, {"source_csv": "x.csv"})

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "data/table.parquet" in names
        assert "data/schema.json" in names
        assert "provenance.json" in names

        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["format_version"] == FORMAT_VERSION

        # analyses are numbered contiguously from 01
        analysis_names = sorted(n for n in names if n.startswith("analyses/"))
        assert len(analysis_names) == len(analyses)
        assert analysis_names[0].startswith("analyses/01-")
        assert analysis_names[-1].startswith(f"analyses/{len(analyses):02d}-")

        # table dtypes/values round-trip exactly through Parquet
        back = pd.read_parquet(io.BytesIO(zf.read("data/table.parquet")))
        pd.testing.assert_frame_equal(back, df)

        # the schema part matches what we passed
        assert json.loads(zf.read("data/schema.json")) == schema


def test_export_table_writes_reopenable_bundle(tmp_path):
    csv = tmp_path / "cell_shape.csv"
    _cells_table().to_csv(csv, index=False)

    out = export_table(csv, tmp_path / "iris")
    assert out.name == "cell_shape.iris"
    assert out.is_file()
    with zipfile.ZipFile(out) as zf:
        assert any(n.startswith("analyses/") for n in zf.namelist())


def test_export_table_unknown_stem_requires_object_key(tmp_path):
    csv = tmp_path / "mystery.csv"
    _cells_table().to_csv(csv, index=False)
    with pytest.raises(ValueError, match="object_key"):
        export_table(csv, tmp_path / "iris")


def test_export_dir_discovers_known_tables(tmp_path):
    _cells_table().to_csv(tmp_path / "cell_shape.csv", index=False)
    # a known-but-not-exported table (deferred) and an unknown CSV are both skipped
    _cells_table().to_csv(tmp_path / "cell_density.csv", index=False)
    _cells_table().to_csv(tmp_path / "unrelated.csv", index=False)

    written = export_dir(tmp_path)
    assert [p.name for p in written] == ["cell_shape.iris"]
    assert written[0].parent == tmp_path / "iris"


def test_export_dir_filters_curated_rows(tmp_path):
    # A small cell_shape table: one experiment, two positions, two frames.
    rows = []
    for position_id in ("p1", "p2"):
        for frame in (0, 1):
            for cell_id in (1, 2):
                rows.append({
                    "condition": "ctrl",
                    "experiment_id": "EXP1",
                    "date": "d1",
                    "position_id": position_id,
                    "frame": frame,
                    "cell_id": cell_id,
                    "cell_shape.area_um2": 100.0,
                })
    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "cell_shape.csv", index=False)

    curation = pd.DataFrame({
        "experiment_id": ["EXP1", "EXP1"],
        "position_id": ["p1", "p2"],
        "frame": [1, pd.NA],          # p1 frame 1, and all of p2
        "excluded": [True, True],
        "exclusion_reason": ["blur", "debris"],
    })

    written = export_dir(
        tmp_path, curation=curation, curation_path="qc/exclusions.csv"
    )

    assert [p.name for p in written] == ["cell_shape.iris"]
    # Read the parquet back out of the bundle and check the kept rows.
    with zipfile.ZipFile(written[0]) as zf:
        back = pd.read_parquet(io.BytesIO(zf.read("data/table.parquet")))
        provenance = json.loads(zf.read("provenance.json"))
    # Only p1 frame 0 survives (2 cells). p1 frame 1 dropped, all of p2 dropped.
    assert set(zip(back["position_id"], back["frame"])) == {("p1", 0)}
    assert len(back) == 2
    # Marker columns are not exported.
    assert "excluded" not in back.columns
    assert "exclusion_reason" not in back.columns
    # Provenance records the filter.
    assert provenance["curation"]["rows_dropped"] == 6
    assert provenance["curation"]["file"] == "qc/exclusions.csv"


def test_export_dir_without_curation_keeps_all_rows(tmp_path):
    _cells_table().to_csv(tmp_path / "cell_shape.csv", index=False)

    written = export_dir(tmp_path)  # no curation

    with zipfile.ZipFile(written[0]) as zf:
        back = pd.read_parquet(io.BytesIO(zf.read("data/table.parquet")))
        provenance = json.loads(zf.read("provenance.json"))
    assert len(back) == len(_cells_table())
    assert "curation" not in provenance


# --------------------------------------------------------------------------- #
# live integration: open a bundle in a real Iris engine (opt-in)
# --------------------------------------------------------------------------- #

def _load_iris_engine():
    """Import the Iris engine from ``$IRIS_ENGINE`` (a checkout's ``engine/`` dir),
    or skip. Kept opt-in so the suite stays self-contained without an Iris checkout
    and free of its heavy deps (fastapi, pingouin, matplotlib)."""
    root = os.environ.get("IRIS_ENGINE")
    if not root or not (Path(root) / "iris_engine").is_dir():
        pytest.skip("set IRIS_ENGINE to an Iris engine/ dir to run the live check")
    os.environ.setdefault("MPLBACKEND", "Agg")
    if root not in sys.path:
        sys.path.insert(0, root)
    document = pytest.importorskip("iris_engine.document")
    main = pytest.importorskip("iris_engine.main")
    return document, main


def test_bundle_loads_and_renders_in_iris(tmp_path):
    iris_doc, iris_main = _load_iris_engine()
    csv = tmp_path / "cell_shape.csv"
    _cells_table().to_csv(csv, index=False)
    bundle = export_table(csv, tmp_path / "iris")

    doc = iris_doc.load_document(bundle.read_bytes())
    table = {"schema": doc["schema"], "rows": doc["rows"]}

    by_id = {a["id"]: a for a in doc["analyses"]}
    # a real comparison (2-level class_label) and a describe-only axis both render
    for analysis_id in ("superplot__class_label__cell_shape.area_um2",
                        "superplot__condition__cell_shape.area_um2"):
        fig, _pg, res, _df, _schema, model, _issues = iris_main._run(
            table, by_id[analysis_id])
        assert "error" not in res
        assert model["inferential_level"] == "date"  # test reads the date unit
        svg = io.BytesIO()
        fig.savefig(svg, format="svg")
        assert b"<svg" in svg.getvalue()[:400]
