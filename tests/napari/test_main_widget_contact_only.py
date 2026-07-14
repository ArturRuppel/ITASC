"""The standalone ITASC Aggregate app: the main widget in contact-only mode.

``make_aggregate_app_widget`` builds ``ITASCMainWidget(upstream_stages=False)``
— the full catalog UI (Data folders → Contact Analysis → Aggregate capstone)
with the three upstream segmentation/tracking stages omitted and a three-dot
per-position rail (cell labels → nucleus labels → contact analysis).
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from napari.qt import get_qapp
from itasc.napari._stage_status import CONTACT_STAGES
from itasc.napari.main_widget import ITASCMainWidget


def _fake_viewer():
    class _Sel:
        active = None

    class _Layers(dict):
        selection = _Sel()
        events = SimpleNamespace(removed=SimpleNamespace(connect=lambda cb: None))

        def remove(self, layer):
            self.pop(layer.name, None)

    viewer = SimpleNamespace()
    viewer.layers = _Layers()
    viewer.dims = SimpleNamespace(
        current_step=(0, 0),
        events=SimpleNamespace(current_step=SimpleNamespace(connect=lambda cb: None)),
    )
    viewer.add_image = MagicMock()
    viewer.add_labels = MagicMock()
    viewer.add_shapes = MagicMock()
    viewer.bind_key = MagicMock()
    return viewer


def _contact_only_app():
    get_qapp()
    return ITASCMainWidget(_fake_viewer(), upstream_stages=False)


def test_contact_only_omits_the_upstream_stage_widgets():
    w = _contact_only_app()
    assert w._has_upstream is False
    assert not hasattr(w, "_cellpose_widget")
    assert not hasattr(w, "nucleus_workflow_widget")
    assert not hasattr(w, "cell_workflow_widget")
    assert not hasattr(w, "cellpose_section")


def test_contact_only_keeps_contact_and_aggregate_sections():
    w = _contact_only_app()
    assert hasattr(w, "contact_analysis_section")
    assert hasattr(w, "aggregate_section")
    # The one stage section is the Contact Analysis detail pane.
    assert w._stage_sections == (w.contact_analysis_section,)
    # Aggregate is below Contact Analysis in the scroll stack.
    layout = w.scroll_layout
    order = [layout.itemAt(i).widget() for i in range(layout.count())]
    assert order.index(w.aggregate_section) > order.index(w.contact_analysis_section)


def test_contact_only_rail_uses_the_three_contact_stages():
    w = _contact_only_app()
    assert w._positions_panel._stages == tuple(CONTACT_STAGES)


def test_contact_only_state_carries_only_calibration_metadata():
    w = _contact_only_app()
    assert sorted(w.get_state()) == ["metadata"]
    # A full-app config file (with upstream stage params) loads without error;
    # the upstream keys are simply ignored.
    w.set_state({"metadata": {"pixel_size_um": "0.5"}, "cellpose": {}, "nucleus": {}})


def test_full_mode_still_builds_all_four_stages():
    get_qapp()
    w = ITASCMainWidget(_fake_viewer())
    assert w._has_upstream is True
    assert len(w._stage_sections) == 4


def _add_positions(panel, positions):
    panel.set_records(
        [
            {"key": str(p), "columns": {}, "payload": {"position_path": p}}
            for p in positions
        ]
    )


def test_run_all_row_hidden_until_positions_are_listed(tmp_path):
    # The widget tree is never shown in the test, so isVisible() is always
    # False; assert the explicit hidden flag the row toggles instead.
    w = _contact_only_app()
    assert w._run_all_row.isHidden() is True
    _add_positions(w._positions_panel, [tmp_path / "pos01"])
    assert w._run_all_row.isHidden() is False


def test_missing_contact_jobs_covers_only_runnable_unbuilt_positions(tmp_path):
    import numpy as np
    import tifffile

    def _write(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(path), np.zeros((2, 4, 4), dtype=np.uint16))

    ready = tmp_path / "ready"          # cell labels, no result → a job
    with_nucleus = tmp_path / "nuc"     # cell + nucleus, no result → a job w/ nucleus
    done = tmp_path / "done"            # already has a result → skipped
    no_cell = tmp_path / "empty"        # no cell labels → skipped
    _write(ready / "cell_labels.tif")
    _write(with_nucleus / "cell_labels.tif")
    _write(with_nucleus / "nucleus_labels.tif")
    _write(done / "cell_labels.tif")
    (done / "contact_analysis.h5").write_bytes(b"")
    no_cell.mkdir()

    w = _contact_only_app()
    _add_positions(w._positions_panel, [ready, with_nucleus, done, no_cell])

    jobs = {j.group_dir: j for j in w._missing_contact_jobs()}
    assert set(jobs) == {ready, with_nucleus}
    assert jobs[ready].nucleus_labels is None
    assert jobs[with_nucleus].nucleus_labels == with_nucleus / "nucleus_labels.tif"
    assert jobs[ready].output == ready / "contact_analysis.h5"


def test_run_all_with_nothing_to_do_reports_and_does_not_launch(tmp_path):
    # A listed position with no cell labels: the batch has no runnable jobs, so
    # clicking must leave a message and start no worker.
    w = _contact_only_app()
    _add_positions(w._positions_panel, [tmp_path / "empty"])
    w._on_run_all_contacts()
    assert w._run_all_worker is None
    assert "Nothing to run" in w._run_all_status.text()
