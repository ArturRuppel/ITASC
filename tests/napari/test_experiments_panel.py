from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from cellflow.napari._experiments_panel import ExperimentsPanel, overall_status
from cellflow.napari._stage_status import (
    DONE,
    MISSING,
    STAGE_CELL,
    STAGE_CELLPOSE,
    STAGE_CONTACTS,
    STAGE_NUCLEUS,
    UNKNOWN,
    WORKING,
)


def _app():
    from napari.qt import get_qapp

    return get_qapp()


def _entry(key: str, cond: str = "WT", pos: str | None = None) -> dict:
    return {
        "key": key,
        "columns": {"condition": cond, "position_id": pos or key},
        "payload": {"position_path": key, "id": pos or key},
    }


def _panel(**kw) -> ExperimentsPanel:
    _app()
    defaults = dict(
        title="Positions",
        input_fields=[("cell", "Cell labels", "cell_labels.tif")],
        status_fn=lambda payload: {s: MISSING for s in (
            STAGE_CELLPOSE, STAGE_NUCLEUS, STAGE_CELL, STAGE_CONTACTS)},
    )
    defaults.update(kw)
    return ExperimentsPanel(**defaults)


# ----------------------------------------------------------------- overall_status
def test_overall_status_done_run_queued():
    stages = (STAGE_CELLPOSE, STAGE_NUCLEUS, STAGE_CELL, STAGE_CONTACTS)
    assert overall_status({s: DONE for s in stages}) == "done"
    assert overall_status({s: MISSING for s in stages}) == "queued"
    mixed = {STAGE_CELLPOSE: DONE, STAGE_NUCLEUS: WORKING,
             STAGE_CELL: MISSING, STAGE_CONTACTS: MISSING}
    assert overall_status(mixed) == "run"
    # An all-unknown row (no canonical root) reads as queued, not done.
    assert overall_status({s: UNKNOWN for s in stages}) == "queued"


# ----------------------------------------------------------------- discover (additive Find)
def test_discover_adds_rows_directly():
    seen: list[tuple[str, dict]] = []

    def discover_fn(root, names):
        seen.append((root, names))
        return [_entry("/data/WT/p1"), _entry("/data/WT/p2")]

    panel = _panel(discover_fn=discover_fn)
    found = panel.discover("/data")
    assert len(found) == 2
    assert seen[0][0] == "/data"
    # One additive Find commits straight to the list — no staging step.
    assert panel.keys() == ["/data/WT/p1", "/data/WT/p2"]
    # Columns derived from the entries' column dicts.
    assert panel.column_names() == ["condition", "position_id"]


def test_discover_is_additive_and_dedupes():
    calls = {"n": 0}

    def discover_fn(root, names):
        calls["n"] += 1
        if calls["n"] == 1:
            return [_entry("/data/WT/p1"), _entry("/data/WT/p2")]
        # Second scan: one overlap, one fresh.
        return [_entry("/data/WT/p2"), _entry("/data/KO/p1")]

    panel = _panel(discover_fn=discover_fn)
    panel.discover("/data/WT")
    panel.discover("/data/KO")
    # Accumulated, overlap deduped by key, order preserved.
    assert panel.keys() == ["/data/WT/p1", "/data/WT/p2", "/data/KO/p1"]


def test_discover_dry_scan_reports_missing_inputs():
    panel = _panel(discover_fn=lambda r, n: [])
    panel.discover("/somewhere")
    assert "No new data folders" in panel._hint.text()
    assert "Setup" in panel._hint.text()


def test_discover_nudges_when_pick_is_too_deep(tmp_path):
    # A too-deep pick: root is the position's own subfolder, so the found
    # position resolves to root's parent (above root).
    pos = tmp_path / "pos11"
    (pos / "0_input").mkdir(parents=True)
    root = str(pos / "0_input")
    panel = _panel(discover_fn=lambda r, n: [_entry(str(pos))])
    panel.discover(root)
    assert "above the folder you picked" in panel._hint.text()
    assert "parent directory" in panel._hint.text()


def test_discover_no_nudge_for_normal_parent_pick(tmp_path):
    # Positions under the picked root: no nudge.
    root = tmp_path / "study"
    (root / "p1").mkdir(parents=True)
    panel = _panel(discover_fn=lambda r, n: [_entry(str(root / "p1"))])
    panel.discover(str(root))
    assert "above the folder" not in panel._hint.text()
    assert "Added 1 data folder." in panel._hint.text()


def test_discover_passes_input_names():
    captured = {}

    def discover_fn(root, names):
        captured.update(names)
        return []

    panel = _panel(
        input_fields=[("cell", "Cell", "cell_labels.tif"),
                      ("nucleus", "Nucleus", "")],
        discover_fn=discover_fn,
    )
    panel.discover("/data")
    # Blank field dropped; filled field passed through.
    assert captured == {"cell": "cell_labels.tif"}


# ----------------------------------------------------------------- selection
def test_plain_click_activates_and_emits_payload():
    panel = _panel()
    panel.set_records([_entry("a"), _entry("b")])
    got: list = []
    panel.active_changed.connect(lambda payload: got.append(payload))
    panel._on_row_clicked("b", 0)
    assert panel.active_payload()["id"] == "b"
    assert got and got[-1]["id"] == "b"
    assert panel.selected_payloads() == [{"position_path": "b", "id": "b"}]


def test_ctrl_click_toggles_multi_selection():
    panel = _panel()
    panel.set_records([_entry("a"), _entry("b"), _entry("c")])
    panel._on_row_clicked("a", 0)
    panel._on_row_clicked("c", 1)  # ctrl-add
    keys = {p["id"] for p in panel.selected_payloads()}
    assert keys == {"a", "c"}
    panel._on_row_clicked("c", 1)  # ctrl-remove
    assert {p["id"] for p in panel.selected_payloads()} == {"a"}


def test_shift_click_selects_range():
    panel = _panel()
    panel.set_records([_entry(k) for k in ("a", "b", "c", "d")])
    panel._on_row_clicked("a", 0)
    panel._on_row_clicked("c", 2)  # shift-range a..c
    assert {p["id"] for p in panel.selected_payloads()} == {"a", "b", "c"}


def test_select_all_and_clear():
    panel = _panel()
    panel.set_records([_entry("a"), _entry("b")])
    panel.select_all()
    assert len(panel.selected_payloads()) == 2
    panel.clear_selection()
    assert panel.selected_payloads() == []


# ----------------------------------------------------------------- delete
def test_delete_selected_removes_committed_rows():
    panel = _panel()
    panel.set_records([_entry("a"), _entry("b"), _entry("c")])
    panel.set_active("b")
    panel.delete_selected()
    assert panel.keys() == ["a", "c"]


# ----------------------------------------------------------------- columns
def test_rename_column_carries_values_table_wide():
    panel = _panel()
    panel.set_records([_entry("a", cond="WT"), _entry("b", cond="KO")])
    assert panel.column_names()[0] == "condition"
    panel.rename_column(0, "genotype")
    assert panel.column_names()[0] == "genotype"
    # Values carried across under the new name.
    payload_cols = panel._records["a"]["columns"]
    assert payload_cols["genotype"] == "WT" and "condition" not in payload_cols


def test_empty_state_shows_call_to_action():
    panel = _panel()
    assert "add folders" in panel._hint.text().lower()
    panel.set_records([_entry("a")])
    assert panel._hint.text() == ""


def test_run_requested_emits_selection_and_workers():
    panel = _panel()
    panel.set_records([_entry("a"), _entry("b")])
    panel.select_all()
    got: list = []
    panel.run_requested.connect(lambda payloads, workers: got.append((payloads, workers)))
    panel._on_run_clicked()
    assert got and len(got[-1][0]) == 2 and got[-1][1] == 1


def test_run_button_enabled_only_with_selection():
    panel = _panel()
    panel.set_records([_entry("a")])
    assert not panel.run_btn.isEnabled()
    panel.set_active("a")
    assert panel.run_btn.isEnabled()


# ----------------------------------------------------------------- status/chip
def test_rail_and_chip_reflect_status_fn():
    done = {STAGE_CELLPOSE: DONE, STAGE_NUCLEUS: DONE, STAGE_CELL: DONE,
            STAGE_CONTACTS: DONE}
    panel = _panel(status_fn=lambda payload: done)
    panel.set_records([_entry("a")])
    row = panel._rows[0]
    assert row._chip.text() == "done"
    assert all(dot.state == DONE for dot in row.rail.dots)


def test_setup_collapses_after_first_find():
    panel = _panel(discover_fn=lambda r, n: [_entry("a")])
    assert panel.setup_section.is_expanded
    panel.discover("/data")
    assert not panel.setup_section.is_expanded


def test_calibration_round_trips():
    panel = _panel(show_calibration=True)
    panel.set_calibration_values({"pixel_size_um": 0.1, "time_interval_s": 30})
    vals = panel.calibration_values()
    assert vals["pixel_size_um"] == "0.1"
    assert vals["time_interval_s"] == "30"


def test_quickstart_mentions_project_file():
    from cellflow.napari._experiments_panel import _QUICKSTART_HTML

    text = _QUICKSTART_HTML.lower()
    assert "no project file" not in text
    assert "project" in text and "catalog" in text
