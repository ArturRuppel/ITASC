"""The napari-free pipeline: discovery → build loop → aggregate → export.

These cover the orchestration ``pipeline`` adds on top of the (separately tested)
stages: that ``build_quantities`` runs one ``.build()`` per buildable
(quantifier, position), threads shared params only into opt-in quantifiers, and
reports progress; and that ``export`` emits the requested tidy artifacts plus the
``.iris`` bundles.
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


def test_build_quantities_threads_params_only_into_opt_in_quantifiers(tmp_path):
    plain = _RecordingQuantifier()
    hungry = _ParamHungryQuantifier()
    rec = _record(tmp_path, "a")

    pipeline.build_quantities(
        [rec], quantifiers=[plain, hungry], params={"fov_area_mm2": 2.0}
    )

    assert plain.calls[0][1] is None  # opted out → no shared params
    assert hungry.calls[0][1] == {"fov_area_mm2": 2.0}  # opted in


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


def test_pipeline_build_aggregate_export_round_trip(tmp_path):
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
    assert "cells_by_frame" in tables

    tables_dir = tables["cells_by_frame"].parent
    written = pipeline.export(tables_dir)

    suffixes = {p.suffix for p in written}
    assert ".parquet" in suffixes and ".iris" in suffixes
    parquet = next(p for p in written if p.suffix == ".parquet")
    assert "cell_shape.area_um2" in pd.read_parquet(parquet).columns


def test_export_rejects_unknown_format(tmp_path):
    tmp_path.joinpath("cells_by_frame.csv").write_text("frame,cell_id\n0,1\n")
    import pytest

    with pytest.raises(ValueError, match="unknown export format"):
        pipeline.export(tmp_path, formats=("xlsx",))
