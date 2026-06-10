"""Regression: a hand mask edit must rebuild the lineage canvas, not just overlays.

Retrack / extend rebuild the canvas via ``_refresh_track_visuals_live``, but
ordinary pixel edits (draw / merge / relabel / redraw / fill) all funnel through
the shared ``_on_cells_edited`` callback.  That callback used to refresh only the
label visuals and the validated overlay, so the canvas (overview presence + the
selected track's detail strip) went stale whenever a mask was edited by hand.

``_on_cells_edited`` now *emits* the ``labels_edited`` domain event and the
display collaborators, wired in ``_wire_events``, repaint themselves. This drives
the real event hub end-to-end (wire → emit) and pins that the full fan-out — label
visuals, validated overlay, canvas and candidate gallery — still fires.
"""

from __future__ import annotations

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

from cellflow.napari.nucleus_correction_widget import NucleusCorrectionWidget
from cellflow.napari._correction_events import CorrectionEvents


class _Stub:
    """A weak-referenceable stand-in (Qt holds a weakref to a slot's receiver,
    which a ``SimpleNamespace`` can't provide) with the wiring methods bound, so
    ``_wire_events`` + an emit run the genuine pub/sub path."""


def _stub_widget() -> _Stub:
    stub = _Stub()
    stub.events = CorrectionEvents()
    stub._refresh_correction_label_visuals_for_edit = MagicMock()
    stub._validated_overlay = SimpleNamespace(on_cells_edited=MagicMock())
    stub.validation_counter_lbl = object()
    # No cell is selected, so the track-path rebuild branch is skipped and this
    # test stays focused on the canvas/gallery refresh it pins.
    stub.correction_widget = SimpleNamespace(_selected_label=0)
    stub._refresh_lineage_canvas_if_shown = MagicMock()
    stub._refresh_candidate_gallery_if_shown = MagicMock()
    for name in ("_wire_events", "_on_cells_edited", "_apply_overlay_edit",
                 "_apply_track_path_edit"):
        setattr(stub, name, MethodType(getattr(NucleusCorrectionWidget, name), stub))
    return stub


def test_cells_edited_fans_out_via_events() -> None:
    stub = _stub_widget()
    stub._wire_events()  # connect the labels_edited subscribers

    stub._on_cells_edited(t=3, changed_ids={7})  # emits labels_edited

    # The label-visuals + validated-overlay refreshes still fire …
    stub._refresh_correction_label_visuals_for_edit.assert_called_once_with(3, {7})
    stub._validated_overlay.on_cells_edited.assert_called_once()
    # … and the canvas + candidate gallery get a live rebuild on every edit
    # (each gated inside its own listener / helper).
    stub._refresh_lineage_canvas_if_shown.assert_called_once_with()
    stub._refresh_candidate_gallery_if_shown.assert_called_once_with()
