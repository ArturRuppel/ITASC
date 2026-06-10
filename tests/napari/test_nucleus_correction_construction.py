"""Smoke test: the real ``NucleusCorrectionWidget`` constructs and its event hub
delivers.

The rest of the suite exercises the widget's logic through lightweight stubs, so
nothing else actually *builds* the widget — which is how a broken
``_wire_events`` (a bad signal/slot connection in ``__init__``) could slip
through CI. This constructs the genuine widget against a real, headless napari
viewer and checks that emitting a domain event reaches its listeners. It skips
cleanly where a viewer can't start headless (no GL / no display).
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

# Prefer the offscreen Qt platform when nothing else picked one, so the viewer
# can start on a headless box. Ignored once a QApplication already exists.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

napari = pytest.importorskip("napari")

from cellflow.napari._correction_events import CorrectionEvents
from cellflow.napari.nucleus_correction_widget import NucleusCorrectionWidget


@pytest.fixture
def viewer():
    try:
        v = napari.Viewer(show=False)
    except Exception as exc:  # pragma: no cover - headless env without GL/display
        pytest.skip(f"napari viewer unavailable headless: {exc}")
    try:
        yield v
    finally:
        v.close()


def test_widget_constructs_and_wires_its_event_hub(viewer):
    # Construction alone exercises _wire_events — the signal/slot connections in
    # __init__ that no other test reaches. A bad connection raises here.
    widget = NucleusCorrectionWidget(viewer)
    assert isinstance(widget.events, CorrectionEvents)

    # The event mechanism must actually deliver on the real widget instance.
    # (The wired listeners also run; they no-op safely with no project loaded.)
    spy = MagicMock()
    widget.events.labels_edited.connect(spy)
    widget._on_cells_edited(2, {5})
    spy.assert_called_once_with(2, {5})
