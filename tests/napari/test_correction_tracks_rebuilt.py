"""The ``tracks_rebuilt`` event (extend / retrack) repaints the live track views.

Extend and retrack reshape the selected track's geometry, so they emit
``tracks_rebuilt``; the comet (only while the track-path overlay is on), the
lineage canvas and the candidate gallery repaint themselves. This drives the
real hub (wire -> emit) and pins that granularity.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

from cellflow.napari.nucleus_correction_widget import NucleusCorrectionWidget
from cellflow.napari._correction_events import CorrectionEvents


class _Stub:
    """Weak-referenceable stand-in (Qt holds a weakref to a slot's receiver)."""


def _stub_widget(*, track_path_on: bool) -> _Stub:
    stub = _Stub()
    stub.events = CorrectionEvents()
    stub.track_path_btn = SimpleNamespace(isChecked=lambda: track_path_on)
    stub._refresh_track_path_overlay = MagicMock()
    stub._refresh_track_path_spotlight = MagicMock()
    stub._refresh_lineage_canvas_if_shown = MagicMock()
    stub._refresh_candidate_gallery_if_shown = MagicMock()
    # Referenced while wiring labels_edited (not exercised here); present so
    # _wire_events can connect without AttributeError.
    stub._refresh_correction_label_visuals_for_edit = MagicMock()
    stub._validated_overlay = SimpleNamespace(on_cells_edited=MagicMock())
    stub.validation_counter_lbl = object()
    stub.correction_widget = SimpleNamespace(_selected_label=0)
    for name in ("_wire_events", "_apply_track_path_rebuilt",
                 "_apply_overlay_edit", "_apply_track_path_edit"):
        setattr(stub, name, MethodType(getattr(NucleusCorrectionWidget, name), stub))
    stub._wire_events()
    return stub


def test_tracks_rebuilt_repaints_comet_canvas_and_gallery_when_overlay_on():
    stub = _stub_widget(track_path_on=True)

    stub.events.tracks_rebuilt.emit()

    stub._refresh_track_path_overlay.assert_called_once_with()
    stub._refresh_track_path_spotlight.assert_called_once_with()
    stub._refresh_lineage_canvas_if_shown.assert_called_once_with()
    stub._refresh_candidate_gallery_if_shown.assert_called_once_with()


def test_tracks_rebuilt_skips_comet_when_overlay_off():
    stub = _stub_widget(track_path_on=False)

    stub.events.tracks_rebuilt.emit()

    # Comet stays untouched while the track-path overlay is off …
    stub._refresh_track_path_overlay.assert_not_called()
    stub._refresh_track_path_spotlight.assert_not_called()
    # … but the canvas + gallery still refresh (each self-gates internally).
    stub._refresh_lineage_canvas_if_shown.assert_called_once_with()
    stub._refresh_candidate_gallery_if_shown.assert_called_once_with()
