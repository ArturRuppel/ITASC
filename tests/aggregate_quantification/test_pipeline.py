"""The napari-free pipeline: discovery → build loop → aggregate (flat tables).

These cover the orchestration ``pipeline`` adds on top of the (separately tested)
stages: that ``build_quantities`` runs one ``.build()`` per buildable
(quantifier, position), threads shared params only into opt-in quantifiers, and
reports progress; and that ``aggregate`` / ``run`` pool the per-position products
into flat, label-agnostic tidy CSVs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from cellflow.aggregate_quantification import pipeline
from cellflow.aggregate_quantification.quantifier import PositionInputs, Quantifier
from cellflow.aggregate_quantification.quantifiers.cell_shape import CellShapeQuantifier


# --------------------------------------------------------------------- helpers


class _RecordingQuantifier(Quantifier):
    """A registry-free quantifier that records each build instead of computing.

    No ``quantity_id`` ⇒ it never auto-registers, so it cannot pollute
    ``available_quantifiers``; tests pass it explicitly via ``quantifiers=``.
    """

    display_name = "Recording (test)"
    requires = ("cell_labels_path",)
    default_output_name = "recording.txt"
    wants_build_params = False

    def __init__(self) -> None:
        self.calls: list[tuple[Path, dict | None]] = []

    def build(self, inputs, output_path, *, params=None, progress_cb=None):
        self.calls.append((Path(output_path), params))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("built")
        return Path(output_path)


class _ParamHungryQuantifier(_RecordingQuantifier):
    """Like the recorder but opts into the shared build params."""

    display_name = "Param-hungry (test)"
    wants_build_params = True


class _PerRecordParamQuantifier(_RecordingQuantifier):
    """Requires a build param it reads per-record (like cell_dynamics' pixel
    size), not from the shared params bar."""

    display_name = "Per-record param (test)"
    required_build_params = {"pixel_size_um": "pixel size (µm/px)"}


class _ProducerQuantifier(_RecordingQuantifier):
    """Writes the artifact a derived quantifier consumes (mirrors contacts)."""

    display_name = "Producer (test)"
    requires = ("cell_labels_path",)
    produces = "contact_analysis_path"
    default_output_name = "contact_analysis.h5"


class _ConsumerQuantifier(_RecordingQuantifier):
    """Builds only from the producer's artifact (mirrors a contacts-derived)."""

    display_name = "Consumer (test)"
    requires = ("contact_analysis_path",)
    default_output_name = "derived.txt"


def _record(tmp: Path, pid: str, *, with_cell: bool = True) -> dict:
    pdir = tmp / pid
    pdir.mkdir(parents=True, exist_ok=True)
    rec = {
        "id": pid,
        "condition": "ctrl",
        "date": "d1",
        "position_path": pdir,
        "cell_tracked_labels_path": (pdir / "cells.tif") if with_cell else None,
        # Pixel size is a global build param stamped on the record (the studio's
        # Parameters bar); quantifiers read it via PositionInputs, not ``params``.
        "pixel_size_um": 0.25,
    }
    return rec


# ----------------------------------------------------------- build_quantities


def test_build_quantities_runs_one_build_per_buildable_position(tmp_path):
    q = _RecordingQuantifier()
    rec_a = _record(tmp_path, "a")
    rec_b = _record(tmp_path, "b")
    rec_c = _record(tmp_path, "c", with_cell=False)  # no cell labels → skipped

    pipeline.build_quantities([rec_a, rec_b, rec_c], quantifiers=[q])

    built_dirs = {out.parent.parent.name for out, _ in q.calls}
    assert built_dirs == {"a", "b"}  # c skipped: can_build is False
    assert all(out.is_file() for out, _ in q.calls)


def test_build_quantities_builds_derived_after_producer_on_cold_run(tmp_path):
    """On a cold run a derived quantifier's input does not exist until its
    producer builds. The scheduler must run the producer first, then re-derive
    inputs so the consumer becomes buildable — even when listed producer-last."""
    from cellflow.aggregate_quantification.quantifier import OUTPUT_SUBDIR

    producer = _ProducerQuantifier()
    consumer = _ConsumerQuantifier()
    rec = _record(tmp_path, "a")
    # Cold: the producer's artifact (the consumer's only input) is not built yet.
    rec["contact_analysis_path"] = (
        Path(rec["position_path"]) / OUTPUT_SUBDIR / "contact_analysis.h5"
    )

    # Consumer first to prove ordering is by dependency, not list position.
    pipeline.build_quantities([rec], quantifiers=[consumer, producer])

    assert len(producer.calls) == 1
    assert len(consumer.calls) == 1  # 0 with the old plan-all-up-front loop


def test_build_quantities_reports_total_including_derived(tmp_path):
    """Progress total counts the derived job that only becomes buildable mid-run."""
    from cellflow.aggregate_quantification.quantifier import OUTPUT_SUBDIR

    producer = _ProducerQuantifier()
    consumer = _ConsumerQuantifier()
    rec = _record(tmp_path, "a")
    rec["contact_analysis_path"] = (
        Path(rec["position_path"]) / OUTPUT_SUBDIR / "contact_analysis.h5"
    )
    seen: list[tuple[int, int, str]] = []

    pipeline.build_quantities(
        [rec], quantifiers=[consumer, producer], progress_cb=lambda *a: seen.append(a)
    )

    assert [(d, t) for d, t, _ in seen] == [(1, 2), (2, 2)]


def test_build_quantities_threads_params_only_into_opt_in_quantifiers(tmp_path):
    plain = _RecordingQuantifier()
    hungry = _ParamHungryQuantifier()
    rec = _record(tmp_path, "a")

    pipeline.build_quantities(
        [rec], quantifiers=[plain, hungry], params={"fov_area_mm2": 2.0}
    )

    assert plain.calls[0][1] is None  # opted out → no shared params
    assert hungry.calls[0][1] == {"fov_area_mm2": 2.0}  # opted in


def test_build_quantities_builds_when_required_param_is_per_record(tmp_path):
    """A required build param supplied per-record (here pixel_size_um on the
    record) must let that position build even with no shared params bar — the
    gate is per-position, not shared-params-only."""
    q = _PerRecordParamQuantifier()
    rec = _record(tmp_path, "a")  # carries pixel_size_um=0.25
    assert rec["pixel_size_um"] == 0.25

    pipeline.build_quantities([rec], quantifiers=[q], params=None)

    assert len(q.calls) == 1  # built, not skipped


def test_build_quantities_skips_only_records_missing_required_param(tmp_path):
    """Per-position: a record with the param builds; one without is skipped."""
    q = _PerRecordParamQuantifier()
    has = _record(tmp_path, "has")            # pixel_size_um=0.25
    missing = _record(tmp_path, "missing")
    missing["pixel_size_um"] = None           # no per-record value, no shared bar

    pipeline.build_quantities([has, missing], quantifiers=[q], params=None)

    # output path is position_dir / OUTPUT_SUBDIR / name → position id is two up.
    built = {p.parent.parent.name for p, _ in q.calls}
    assert built == {"has"}


def test_build_quantities_reports_progress(tmp_path):
    q = _RecordingQuantifier()
    recs = [_record(tmp_path, "a"), _record(tmp_path, "b")]
    seen: list[tuple[int, int, str]] = []

    pipeline.build_quantities(recs, quantifiers=[q], progress_cb=lambda *a: seen.append(a))

    assert [(d, t) for d, t, _ in seen] == [(1, 2), (2, 2)]
    assert {name for _, _, name in seen} == {"a", "b"}


def test_build_quantities_defaults_to_registered_quantifiers(tmp_path):
    # cell_shape is registered and builds from cell labels; a real tif drives it.
    frame = np.zeros((6, 8), dtype=np.uint16)
    frame[:, :4] = 1
    frame[:, 4:] = 2
    rec = _record(tmp_path, "a")
    tifffile.imwrite(rec["cell_tracked_labels_path"], np.stack([frame, frame]))

    # pixel_size_um clears cell_shape's required-param gate; with no FOV area,
    # cell_density is gated out rather than raising — "build all" stays usable.
    pipeline.build_quantities([rec], params={"pixel_size_um": 0.25})

    out = CellShapeQuantifier().default_output(
        PositionInputs(position_dir=Path(rec["position_path"]))
    )
    assert out.is_file()  # the registered cell_shape quantifier ran


# ------------------------------------------------------------ build_catalog


def test_build_catalog_discovers_and_writes_skeleton(tmp_path):
    pos = tmp_path / "study" / "pos1"
    pos.mkdir(parents=True)
    tifffile.imwrite(pos / "cells.tif", np.zeros((2, 4, 4), dtype=np.uint16))
    out_csv = tmp_path / "catalog.csv"

    records = pipeline.build_catalog(
        tmp_path / "study", cell_name="cells.tif", out_csv=out_csv
    )

    assert [r["id"] for r in records] == ["pos1"]
    assert out_csv.is_file()
    from cellflow.aggregate_quantification.catalog import load_catalog

    assert [r["id"] for r in load_catalog(out_csv)] == ["pos1"]


# -------------------------------------------------------------- end to end


def test_pipeline_build_aggregate_round_trip(tmp_path):
    frame = np.zeros((6, 8), dtype=np.uint16)
    frame[:, :4] = 1
    frame[:, 4:] = 2
    recs = []
    for pid in ("a", "b"):
        rec = _record(tmp_path, pid)
        tifffile.imwrite(rec["cell_tracked_labels_path"], np.stack([frame, frame]))
        recs.append(rec)

    pipeline.build_quantities(
        recs, quantifiers=[CellShapeQuantifier()], params={"pixel_size_um": 0.25}
    )
    tables = pipeline.aggregate(recs, tmp_path / "catalogue")
    assert "cell_shape" in tables
    # Tables are written flat (no aggregate_quantification subdir) and are
    # label-agnostic (no class_label column).
    path = tables["cell_shape"]
    assert path == tmp_path / "catalogue" / "cell_shape.csv"
    df = pd.read_csv(path)
    assert "class_label" not in df.columns
    assert set(df["position_id"]) == {"a", "b"}


# --------------------------------------------------------- quantities selection


def test_select_quantifiers_empty_is_every_registered():
    from cellflow.aggregate_quantification.quantifier import available_quantifiers

    selected = {type(q).quantity_id for q in pipeline.select_quantifiers(())}
    assert selected == {cls.quantity_id for cls in available_quantifiers()}


def test_select_quantifiers_subset_pulls_in_producer():
    """Selecting a contacts-derived metric brings the contacts producer along, even
    though it was not named, so the derived metric is actually buildable."""
    selected = {type(q).quantity_id for q in pipeline.select_quantifiers(["neighbor_count"])}
    assert "neighbor_count" in selected
    assert "contacts" in selected  # producer of contact_analysis_path, pulled in


def test_select_quantifiers_unknown_raises():
    import pytest

    with pytest.raises(ValueError, match="bogus"):
        pipeline.select_quantifiers(["bogus"])


# ----------------------------------------------------------- run (config-driven)


def test_run_from_config_round_trip(tmp_path):
    from cellflow.aggregate_quantification.catalog import save_catalog

    frame = np.zeros((6, 8), dtype=np.uint16)
    frame[:, :4] = 1
    frame[:, 4:] = 2
    study = tmp_path / "study"
    recs = []
    for pid in ("a", "b"):
        pdir = study / pid
        pdir.mkdir(parents=True)
        cells = pdir / "cells.tif"
        tifffile.imwrite(cells, np.stack([frame, frame]))
        recs.append({
            "id": pid, "condition": "ctrl", "date": "d1",
            "experiment_id": f"EXP-{pid}", "position_path": pdir,
            "cell_tracked_labels_path": cells,
        })
    save_catalog(tmp_path / "catalog.csv", recs)

    config = tmp_path / "config.toml"
    config.write_text(
        'catalog = "catalog.csv"\n'
        'quantities = ["cell_shape"]\n'
        'out_dir = "tables"\n'
        "[params]\npixel_size_um = 0.25\n"
    )

    written = pipeline.run(config)

    # run() returns the table name → flat CSV path map; tables land in out_dir.
    out_dir = tmp_path / "tables"
    assert written["cell_shape"] == out_dir / "cell_shape.csv"
    measured = pd.read_csv(out_dir / "cell_shape.csv")
    assert set(measured["position_id"]) == {"a", "b"}
    assert "class_label" not in measured.columns


def test_run_defaults_out_dir_to_catalogue_root(tmp_path):
    from cellflow.aggregate_quantification.catalog import save_catalog

    frame = np.zeros((6, 8), dtype=np.uint16)
    frame[:, :4] = 1
    frame[:, 4:] = 2
    study = tmp_path / "study"
    pdir = study / "a"
    pdir.mkdir(parents=True)
    cells = pdir / "cells.tif"
    tifffile.imwrite(cells, np.stack([frame, frame]))
    save_catalog(tmp_path / "catalog.csv", [{
        "id": "a", "condition": "ctrl", "date": "d1", "experiment_id": "EXP-a",
        "position_path": pdir, "cell_tracked_labels_path": cells,
    }])
    config = tmp_path / "config.toml"
    config.write_text(
        'catalog = "catalog.csv"\nquantities = ["cell_shape"]\n'
        "[params]\npixel_size_um = 0.25\n"
    )

    written = pipeline.run(config)
    # No out_dir → catalogue root (the single position's parent = study/).
    assert written["cell_shape"] == study / "cell_shape.csv"


def test_run_forwards_progress_cb(tmp_path, monkeypatch):
    """run(progress_cb=...) threads the callback into build_quantities."""
    seen = {}

    def fake_build(catalog, *, quantifiers=None, params=None, progress_cb=None):
        seen["build"] = progress_cb

    monkeypatch.setattr(pipeline, "build_quantities", fake_build)
    monkeypatch.setattr(pipeline, "aggregate", lambda *a, **k: {})
    monkeypatch.setattr(pipeline, "load_catalog", lambda p: [])

    cfg = tmp_path / "config.toml"
    cfg.write_text('catalog = "catalog.csv"\n', encoding="utf-8")

    def cb(done, total, label):  # pragma: no cover - identity check only
        pass

    pipeline.run(cfg, progress_cb=cb)
    assert seen["build"] is cb
