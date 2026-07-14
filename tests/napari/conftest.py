from __future__ import annotations

import sys
from types import MethodType
from unittest.mock import MagicMock

import pytest


# Every slot ``NucleusCorrectionWidget._wire_events`` connects. Kept in one place
# so adding an event only updates this list, not every event test's stub.
_WIRE_EVENT_SLOTS = (
    "_refresh_correction_label_visuals_for_edit",
    "_refresh_correction_label_visuals",
    "_apply_overlay_edit",
    "_apply_track_path_edit",
    "_apply_track_path_rebuilt",
    "_refresh_validated_overlay",
    "_refresh_validation_counter",
    "_refresh_lineage_canvas_if_shown",
    "_refresh_lineage_canvas_status_if_shown",
    "_refresh_lineage_detail_if_shown",
    "_refresh_candidate_gallery_if_shown",
)


class _WidgetStub:
    """Weak-referenceable host (Qt holds a weakref to a bound-method slot)."""


@pytest.fixture
def wired_stub():
    """Factory for a correction-widget stub with a real event hub wired up.

    Every ``_wire_events`` slot is a ``MagicMock`` by default; pass ``bind=[...]``
    to install the *real* handler methods (to exercise their logic) and keyword
    args to override any slot / supply extra attributes. Emitting an event on
    ``stub.events`` then runs the genuine subscriber set.
    """
    from itasc.napari.correction.nucleus_correction_widget import NucleusCorrectionWidget
    from itasc.napari.correction._correction_events import CorrectionEvents

    def _make(*, bind=(), **attrs):
        stub = _WidgetStub()
        stub.events = CorrectionEvents()
        for name in _WIRE_EVENT_SLOTS:
            setattr(stub, name, MagicMock())
        for name, value in attrs.items():
            setattr(stub, name, value)
        for name in bind:
            setattr(stub, name, MethodType(getattr(NucleusCorrectionWidget, name), stub))
        NucleusCorrectionWidget._wire_events(stub)
        return stub

    return _make


@pytest.fixture(autouse=True)
def _restore_napari_import_stubs_and_close_viewers():
    tracked_roots = {
        "itasc.napari",
        "itasc.tracking_ultrack",
        "itasc.segmentation",
    }
    tracked_prefixes = tuple(f"{name}." for name in tracked_roots)
    originals = {
        name: module
        for name, module in sys.modules.items()
        if name in tracked_roots or name.startswith(tracked_prefixes)
    }
    yield
    try:
        import napari

        napari.Viewer.close_all()
    except Exception:
        pass
    for name in list(sys.modules):
        if (name in tracked_roots or name.startswith(tracked_prefixes)) and name not in originals:
            sys.modules.pop(name, None)
    for name, module in originals.items():
        sys.modules[name] = module
