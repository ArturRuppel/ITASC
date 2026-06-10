"""The ``tracks_rebuilt`` event (extend / retrack) repaints the live track views.

Extend and retrack reshape the selected track's geometry, so they emit
``tracks_rebuilt``; the comet (only while the track-path overlay is on), the
lineage canvas and the candidate gallery repaint themselves. This drives the
real hub (wire -> emit) and pins that granularity.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock


def test_tracks_rebuilt_repaints_comet_canvas_and_gallery_when_overlay_on(wired_stub):
    stub = wired_stub(
        bind=["_apply_track_path_rebuilt"],
        track_path_btn=SimpleNamespace(isChecked=lambda: True),
        _refresh_track_path_overlay=MagicMock(),
        _refresh_track_path_spotlight=MagicMock(),
    )

    stub.events.tracks_rebuilt.emit()

    stub._refresh_track_path_overlay.assert_called_once_with()
    stub._refresh_track_path_spotlight.assert_called_once_with()
    stub._refresh_lineage_canvas_if_shown.assert_called_once_with()
    stub._refresh_candidate_gallery_if_shown.assert_called_once_with()


def test_tracks_rebuilt_skips_comet_when_overlay_off(wired_stub):
    stub = wired_stub(
        bind=["_apply_track_path_rebuilt"],
        track_path_btn=SimpleNamespace(isChecked=lambda: False),
        _refresh_track_path_overlay=MagicMock(),
        _refresh_track_path_spotlight=MagicMock(),
    )

    stub.events.tracks_rebuilt.emit()

    # Comet stays untouched while the track-path overlay is off …
    stub._refresh_track_path_overlay.assert_not_called()
    stub._refresh_track_path_spotlight.assert_not_called()
    # … but the canvas + gallery still refresh (each self-gates internally).
    stub._refresh_lineage_canvas_if_shown.assert_called_once_with()
    stub._refresh_candidate_gallery_if_shown.assert_called_once_with()
