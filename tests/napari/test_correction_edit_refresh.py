"""Regression: a hand mask edit must rebuild the lineage canvas, not just overlays.

Retrack / extend rebuild the canvas via the ``tracks_rebuilt`` event, but
ordinary pixel edits (draw / merge / relabel / redraw / fill) funnel through the
``_on_cells_edited`` callback, which now *emits* ``labels_edited``. The display
collaborators wired in ``_wire_events`` repaint themselves. This drives the real
hub (wire -> emit) and pins that the full fan-out — label visuals, validated
overlay, canvas and candidate gallery — still fires on every edit.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from itasc.napari.correction.nucleus_correction_widget import NucleusCorrectionWidget


def test_cells_edited_fans_out_via_events(wired_stub) -> None:
    overlay = SimpleNamespace(on_cells_edited=MagicMock())
    stub = wired_stub(
        # Exercise the real overlay handler so we see it reach on_cells_edited.
        bind=["_apply_overlay_edit"],
        _validated_overlay=overlay,
        validation_counter_lbl=object(),
        # No cell selected -> the track-path branch is skipped.
        correction_widget=SimpleNamespace(_selected_label=0),
    )

    NucleusCorrectionWidget._on_cells_edited(stub, t=3, changed_ids={7})

    stub._refresh_correction_label_visuals_for_edit.assert_called_once_with(3, {7})
    overlay.on_cells_edited.assert_called_once()
    stub._refresh_lineage_canvas_if_shown.assert_called_once_with()
    stub._refresh_candidate_gallery_if_shown.assert_called_once_with()
