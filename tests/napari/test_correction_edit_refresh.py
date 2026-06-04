"""Regression: a hand mask edit must rebuild the lineage canvas, not just overlays.

Retrack / extend rebuild the canvas via ``_refresh_track_visuals_live``, but
ordinary pixel edits (draw / merge / relabel / redraw / fill) all funnel through
the shared ``_on_cells_edited`` callback.  That callback used to refresh only the
label visuals and the validated overlay, so the canvas (overview presence + the
selected track's detail strip) went stale whenever a mask was edited by hand.
This pins the callback to also rebuild the canvas.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from cellflow.napari.nucleus_correction_widget import NucleusCorrectionWidget


def _stub_widget() -> SimpleNamespace:
    """A minimal object exposing only what ``_on_cells_edited`` touches."""
    return SimpleNamespace(
        _refresh_correction_label_visuals_for_edit=MagicMock(),
        _validated_overlay=SimpleNamespace(on_cells_edited=MagicMock()),
        validation_counter_lbl=object(),
        # No cell is selected, so the track-path rebuild branch is skipped and
        # this test stays focused on the canvas/gallery refresh it pins.
        correction_widget=SimpleNamespace(_selected_label=0),
        _refresh_lineage_canvas_if_shown=MagicMock(),
        _refresh_candidate_gallery_if_shown=MagicMock(),
    )


def test_cells_edited_rebuilds_canvas() -> None:
    stub = _stub_widget()

    NucleusCorrectionWidget._on_cells_edited(stub, t=3, changed_ids={7})

    # The pre-existing refreshes still fire …
    stub._refresh_correction_label_visuals_for_edit.assert_called_once_with(3, {7})
    stub._validated_overlay.on_cells_edited.assert_called_once()
    # … and the canvas + candidate gallery get a live rebuild on every edit
    # (each gated inside its helper).
    stub._refresh_lineage_canvas_if_shown.assert_called_once_with()
    stub._refresh_candidate_gallery_if_shown.assert_called_once_with()
