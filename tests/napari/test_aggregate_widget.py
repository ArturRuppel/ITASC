"""Aggregate capstone: readiness partition + engine drive."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from itasc.napari.aggregate_widget import partition_ready


def _record(pos_dir: Path, *, ready: bool) -> dict:
    """A main_widget-shaped catalog record (see ``_catalog_record_for_position``).

    Identity lives in the ``columns`` bag under the seed level names
    (``condition`` / ``experiment_id`` / ``position_id``); ``save_catalog`` writes
    those columns verbatim, and their combination is the position's identity.
    ``ready`` controls whether the per-position ``contacts.h5`` exists on disk.
    Both label images are placed on disk: ``position_inputs_from_record`` gates a
    label input on the file existing, so a stamped-but-missing path is not an
    available input.
    """
    h5 = pos_dir / "4_contact_analysis" / "contact_analysis.h5"
    if ready:
        h5.parent.mkdir(parents=True, exist_ok=True)
        h5.write_bytes(b"")
    cell = pos_dir / "cell_labels.tif"
    nucleus = pos_dir / "nucleus_labels.tif"
    pos_dir.mkdir(parents=True, exist_ok=True)
    cell.touch()
    nucleus.touch()
    return {
        "position_path": pos_dir,
        "contact_analysis_path": h5,
        "cell_tracked_labels_path": cell,
        "nucleus_tracked_labels_path": nucleus,
        "columns": {
            "condition": "ctrl",
            "experiment_id": "exp1",
            "position_id": pos_dir.name,
        },
    }


def _nucleus_only_record(pos_dir: Path, *, pixel_size: float = 0.5) -> dict:
    """A position with only a nucleus label channel — no cell labels, no built
    ``contact_analysis.h5``. It can still pool the nucleus quantities (which need
    only nuclear labels + a pixel size), so it must count as ready to pool."""
    pos_dir.mkdir(parents=True, exist_ok=True)
    nucleus = pos_dir / "nucleus_labels.tif"
    nucleus.touch()  # the input is gated on the file existing
    return {
        "position_path": pos_dir,
        # The h5 path is stamped by the catalog but the file is never built here.
        "contact_analysis_path": pos_dir / "4_contact_analysis" / "contact_analysis.h5",
        "cell_tracked_labels_path": "",  # no cell channel
        "nucleus_tracked_labels_path": nucleus,
        "pixel_size_um": pixel_size,
        "columns": {
            "condition": "ctrl",
            "experiment_id": "exp1",
            "position_id": pos_dir.name,
        },
    }


def test_partition_ready_splits_by_poolable_inputs(tmp_path):
    # A position with a built h5 is ready (contact-derived quantities pool); one
    # with no h5, no pixel size, no other usable input pools nothing → not ready.
    a = _record(tmp_path / "posA", ready=True)
    b = _record(tmp_path / "posB", ready=False)
    ready, not_ready = partition_ready([a, b])
    assert ready == [a]
    assert not_ready == [b]


def test_partition_ready_accepts_nucleus_only_position(tmp_path):
    # The screenshot case: nucleus labels but no cell labels and no built h5. The
    # nucleus quantities pool from the nuclear labels alone, so it is ready.
    rec = _nucleus_only_record(tmp_path / "posN")
    ready, not_ready = partition_ready([rec])
    assert ready == [rec]
    assert not_ready == []


def test_partition_ready_empty():
    assert partition_ready([]) == ([], [])


from itasc.contact_analysis import load_catalog
from itasc.napari import aggregate_widget as aw


def test_pool_positions_authors_ready_subset_and_runs(tmp_path, monkeypatch):
    ready = [
        _record(tmp_path / "study" / "posA", ready=True),
        _record(tmp_path / "study" / "posB", ready=True),
    ]
    seen = {}

    def _fake_run(config_path, *, build=True):
        seen["config_path"] = Path(config_path)
        seen["build"] = build
        return {"object_table": Path(config_path).parent / "object_table.csv"}

    monkeypatch.setattr(aw, "run", _fake_run)

    result = aw.pool_positions(ready, skipped_names=["posC"])

    # The engine was driven with the authored config, pool-only (no rebuild).
    config_path = seen["config_path"]
    assert config_path.name == "config.toml"
    assert seen["build"] is False
    # The authored catalog contains exactly the two ready positions.
    catalog = load_catalog(config_path.parent / "catalog.csv")
    names = sorted(Path(rec["position_path"]).name for rec in catalog)
    assert names == ["posA", "posB"]
    # The result carries the skipped names and table map for the UI.
    assert result["skipped"] == ["posC"]
    assert "object_table" in result["tables"]
    # Tables land in the ready positions' common ancestor.
    assert result["project_dir"] == (tmp_path / "study")


from napari.qt import get_qapp


def test_widget_readout_reports_ready_split_and_names_not_ready(tmp_path):
    get_qapp()
    from itasc.napari.aggregate_widget import AggregateWidget

    w = AggregateWidget()
    w.set_records([
        _record(tmp_path / "posA", ready=True),
        _record(tmp_path / "posB", ready=False),
    ])
    assert "1 of 2" in w.readout.text()
    assert "posB" in w.readout.text()
    assert w.run_btn.isEnabled() is True
    assert w.section_status() == "in_progress"


def test_widget_run_button_disabled_when_nothing_ready(tmp_path):
    get_qapp()
    from itasc.napari.aggregate_widget import AggregateWidget

    w = AggregateWidget()
    w.set_records([_record(tmp_path / "posA", ready=False)])
    assert w.run_btn.isEnabled() is False
    assert w.section_status() == "not_started"


def test_aggregate_stage_accent_resolves():
    from itasc.napari.ui_style import stage_accent

    accent = stage_accent("aggregate")
    assert isinstance(accent, str) and accent.startswith("#")


# ------------------------------------------------------ quantity checkbox list


def test_widget_has_checkbox_per_pooled_quantifier():
    get_qapp()
    from itasc.napari.aggregate_widget import AggregateWidget, pooled_quantifiers

    w = AggregateWidget()
    expected = {cls.quantity_id for cls in pooled_quantifiers()}
    assert set(w._checks) == expected
    assert "contacts" not in w._checks  # a producer — never pooled


def test_widget_greys_unsupported_quantities(tmp_path):
    get_qapp()
    from itasc.napari.aggregate_widget import AggregateWidget

    # The bare _record has cell+nucleus labels and an existing contacts.h5, but no
    # pixel size / FOV area — so contacts-derived quantities are supported and the
    # pixel-size-gated shape quantities are not.
    w = AggregateWidget()
    w.set_records([_record(tmp_path / "posA", ready=True)])
    assert w._checks["neighbor_count"].isEnabled()
    assert w._checks["neighbor_count"].isChecked()
    assert not w._checks["cell_shape"].isEnabled()
    assert not w._checks["cell_shape"].isChecked()

    # Stamping a pixel size lifts cell_shape into support (enabled + re-checked).
    rec = _record(tmp_path / "posB", ready=True)
    rec["pixel_size_um"] = 0.5
    w.set_records([rec])
    assert w._checks["cell_shape"].isEnabled()
    assert w._checks["cell_shape"].isChecked()


def test_nucleus_only_position_pools_nucleus_quantities(tmp_path):
    get_qapp()
    from itasc.napari.aggregate_widget import AggregateWidget

    # Nucleus labels + pixel size, but no cell labels and no built h5. The nucleus
    # quantities light up; the cell- and contact-derived ones stay greyed, and the
    # position is still poolable (run button enabled).
    w = AggregateWidget()
    w.set_records([_nucleus_only_record(tmp_path / "posN")])
    assert w._checks["nucleus_shape"].isEnabled()
    assert w._checks["nucleus_shape"].isChecked()
    assert not w._checks["cell_shape"].isEnabled()
    assert not w._checks["neighbor_count"].isEnabled()  # contact-derived, no h5
    assert w.run_btn.isEnabled() is True
    assert "1 of 1" in w.readout.text()


def test_selected_quantities_collapses_and_respects_unchecks(tmp_path):
    get_qapp()
    from itasc.napari.aggregate_widget import AggregateWidget

    w = AggregateWidget()
    w.set_records([_record(tmp_path / "posA", ready=True)])
    # All supported boxes checked → collapse to () (= write every available table).
    assert w._selected_quantities() == ()
    # Unchecking a supported box yields the explicit checked subset.
    w._checks["neighbor_count"].setChecked(False)
    assert w._selected_quantities() == ("signed_contact_length",)


def test_pool_positions_authors_selected_quantities(tmp_path, monkeypatch):
    ready = [_record(tmp_path / "study" / "posA", ready=True)]
    seen = {}

    def _fake_run(config_path, *, build=True):
        seen["config_path"] = Path(config_path)
        return {}

    monkeypatch.setattr(aw, "run", _fake_run)
    aw.pool_positions(ready, skipped_names=[], quantities=("cell_shape",))

    config_text = (seen["config_path"]).read_text()
    assert 'quantities = ["cell_shape"]' in config_text


# ---------------------------------------------------------- FOV area + params

import numpy as np
import pytest
import tifffile


def _ready_record_with_labels(tmp_path, name, shape=(2, 6, 8), pixel_size=None):
    """A ready record whose cell-label TIFF actually exists on disk (so the FOV
    autofill can read its lateral pixel dims)."""
    rec = _record(tmp_path / name, ready=True)
    cells = Path(rec["cell_tracked_labels_path"])
    cells.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(cells, np.zeros(shape, dtype=np.uint16))
    if pixel_size is not None:
        rec["pixel_size_um"] = pixel_size
    return rec


def test_fov_autofills_from_image_size_and_pixel_size(tmp_path):
    get_qapp()
    from itasc.napari.aggregate_widget import AggregateWidget

    w = AggregateWidget()
    # 6x8 lateral pixels, 0.5 µm/px -> 48 px * 0.25 µm² = 12 µm² (field is µm²).
    w.set_records([_ready_record_with_labels(tmp_path, "posA", (2, 6, 8), pixel_size=0.5)])
    assert w.fov_field.value() == pytest.approx(12.0)
    # A positive FOV lifts Cell density into support (enabled + checked).
    assert w._checks["cell_density"].isEnabled()
    assert w._checks["cell_density"].isChecked()


def test_fov_autofill_backs_off_after_user_edit(tmp_path):
    get_qapp()
    from itasc.napari.aggregate_widget import AggregateWidget

    w = AggregateWidget()
    w.fov_field.setValue(3.0)  # user edit sets the sticky flag
    w.set_records([_ready_record_with_labels(tmp_path, "posA", (2, 6, 8), pixel_size=0.5)])
    assert w.fov_field.value() == 3.0  # autofill left the user's value alone


def test_manual_fov_lights_up_density_without_pixel_size(tmp_path):
    get_qapp()
    from itasc.napari.aggregate_widget import AggregateWidget

    w = AggregateWidget()
    w.set_records([_record(tmp_path / "posA", ready=True)])  # labels present, no pixel size, no FOV
    assert not w._checks["cell_density"].isEnabled()  # no FOV -> greyed
    w.fov_field.setValue(1.5)
    assert w._checks["cell_density"].isEnabled()  # typing an area supports density
    assert "fov_area_mm2" in w._current_params()


def test_current_params_carries_calibration_and_fov(tmp_path):
    get_qapp()
    from itasc.napari.aggregate_widget import AggregateWidget

    w = AggregateWidget()
    rec = _record(tmp_path / "posA", ready=True)
    rec["pixel_size_um"] = 0.25
    rec["time_interval_s"] = 2.0
    w.set_records([rec])
    w.fov_field.setValue(4.0)  # 4 µm² -> 4e-6 mm² backend param
    params = w._current_params()
    assert params["pixel_size_um"] == 0.25
    assert params["time_interval_s"] == 2.0
    assert params["fov_area_mm2"] == pytest.approx(4e-6)


def test_run_params_reach_the_authored_config(tmp_path, monkeypatch):
    ready = [_record(tmp_path / "study" / "posA", ready=True)]
    seen = {}

    def _fake_run(config_path, *, build=True):
        seen["config_path"] = Path(config_path)
        return {}

    monkeypatch.setattr(aw, "run", _fake_run)
    aw.pool_positions(ready, skipped_names=[], quantities=(), params={"fov_area_mm2": 2.5})
    assert "fov_area_mm2 = 2.5" in seen["config_path"].read_text()


def test_disabled_quantity_tooltip_has_no_em_dash(tmp_path):
    get_qapp()
    from itasc.napari.aggregate_widget import AggregateWidget

    w = AggregateWidget()
    w.set_records([_record(tmp_path / "posA", ready=True)])  # cell_shape unsupported
    tip = w._checks["cell_shape"].toolTip()
    assert not w._checks["cell_shape"].isEnabled()
    assert "—" not in tip and "–" not in tip
