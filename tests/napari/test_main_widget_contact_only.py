"""The standalone CellFlow Aggregate app: the main widget in contact-only mode.

``make_aggregate_app_widget`` builds ``CellFlowMainWidget(upstream_stages=False)``
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
from cellflow.napari._stage_status import CONTACT_STAGES
from cellflow.napari.main_widget import CellFlowMainWidget


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
    return CellFlowMainWidget(_fake_viewer(), upstream_stages=False)


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
    w = CellFlowMainWidget(_fake_viewer())
    assert w._has_upstream is True
    assert len(w._stage_sections) == 4
