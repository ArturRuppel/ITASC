"""Regression: reassign-IDs and remove-unvalidated must rebuild the track overview.

Both operations rewrite the tracked label stack (reassigning cell IDs to a
contiguous range, or dropping every pixel of an unvalidated track), so the
lineage swimlane overview goes stale unless it is rebuilt.  Like every other
*topology*-changing handler (e.g. ``_on_cells_edited``), these two must call
``_refresh_lineage_canvas_if_shown``.  This pins that behaviour.  (Flag-only
handlers like ``_on_validate_track`` instead use the lighter
``_refresh_lineage_canvas_status_if_shown``, which recolours without a rescan.)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

import cellflow.napari.nucleus_correction_widget as mod
from cellflow.napari.nucleus_correction_widget import NucleusCorrectionWidget


def test_reassign_ids_done_rebuilds_canvas() -> None:
    layer = SimpleNamespace(data=np.zeros((1, 1, 1), dtype=int))
    stub = SimpleNamespace(
        _correction_tracked_layer=MagicMock(return_value=layer),
        _refresh_correction_label_visuals=MagicMock(),
        _refresh_lineage_canvas_if_shown=MagicMock(),
        _correction_status=MagicMock(),
        # No project / no remap, so remap_validated_tracks is never reached.
        _pos_dir=None,
    )

    remapped = np.ones((1, 1, 1), dtype=int)
    NucleusCorrectionWidget._on_reassign_ids_done(stub, (remapped, 1, {}))

    assert layer.data is remapped
    stub._refresh_lineage_canvas_if_shown.assert_called_once_with()


def test_remove_unvalidated_rebuilds_canvas(monkeypatch) -> None:
    layer = SimpleNamespace(
        data=np.ones((1, 2, 2), dtype=int),
        refresh=MagicMock(),
    )
    stub = SimpleNamespace(
        _pos_dir=object(),
        _correction_tracked_layer=MagicMock(return_value=layer),
        _refresh_correction_label_visuals=MagicMock(),
        _refresh_validated_overlay=MagicMock(),
        _refresh_validation_counter=MagicMock(),
        _refresh_lineage_canvas_if_shown=MagicMock(),
        _correction_status=MagicMock(),
        # No cell selected, so the selection-reset branch is skipped.
        correction_widget=SimpleNamespace(_selected_label=0),
    )

    monkeypatch.setattr(mod, "read_validated_tracks", lambda _pos: set())
    monkeypatch.setattr(
        mod,
        "remove_unvalidated_from_data",
        lambda _data, _tracks: SimpleNamespace(changed_pixels=4, changed_frames=1),
    )

    NucleusCorrectionWidget._on_remove_unvalidated_labels(stub)

    stub._refresh_lineage_canvas_if_shown.assert_called_once_with()


def test_validate_track_uses_lightweight_status_refresh(monkeypatch) -> None:
    """Validation is flag-only, so it must recolour, not rescan the whole stack.

    Rebuilding the lineage over the whole stack on every validation froze the
    GUI for seconds on long tracks; the status refresh skips ``build_lineage``.
    """
    layer = SimpleNamespace(data=np.ones((2, 2, 2), dtype=int))
    stub = SimpleNamespace(
        _pos_dir=object(),
        _correction_tracked_layer=MagicMock(return_value=layer),
        _frames_with_cell=MagicMock(return_value=[0, 1]),
        _refresh_validated_overlay=MagicMock(),
        _refresh_validation_counter=MagicMock(),
        _refresh_lineage_canvas_if_shown=MagicMock(),
        _refresh_lineage_canvas_status_if_shown=MagicMock(),
        _correction_status=MagicMock(),
        correction_widget=SimpleNamespace(_selected_label=7),
    )

    monkeypatch.setattr(mod, "corrections_for_label_frames", lambda *a, **k: [])
    monkeypatch.setattr(mod, "add_corrections", MagicMock())

    NucleusCorrectionWidget._on_validate_track(stub)

    stub._refresh_lineage_canvas_status_if_shown.assert_called_once_with()
    stub._refresh_lineage_canvas_if_shown.assert_not_called()


def test_refresh_overview_when_film_strip_toggled_off() -> None:
    """The always-visible overview must rebuild even with the detail strip off.

    ``lineage_canvas_check`` only toggles the film-strip *detail*; the swimlane
    overview stays docked the whole time focus mode is active. So the rebuild is
    gated on the workspace splitter, not on that checkbox.
    """
    canvas = SimpleNamespace(refresh=MagicMock())
    stub = SimpleNamespace(
        _workspace_splitter=object(),  # focus mode active → overview visible
        lineage_canvas_check=SimpleNamespace(isChecked=lambda: False),
        _lineage_canvas=canvas,
    )

    NucleusCorrectionWidget._refresh_lineage_canvas_if_shown(stub)

    canvas.refresh.assert_called_once_with()


def test_refresh_overview_skipped_when_not_in_focus_mode() -> None:
    canvas = SimpleNamespace(refresh=MagicMock())
    stub = SimpleNamespace(
        _workspace_splitter=None,  # focus mode off → nothing docked
        lineage_canvas_check=SimpleNamespace(isChecked=lambda: True),
        _lineage_canvas=canvas,
    )

    NucleusCorrectionWidget._refresh_lineage_canvas_if_shown(stub)

    canvas.refresh.assert_not_called()


def test_remove_unvalidated_noop_skips_canvas(monkeypatch) -> None:
    """When nothing changes the overview is already current; don't rebuild it."""
    layer = SimpleNamespace(data=np.ones((1, 2, 2), dtype=int), refresh=MagicMock())
    stub = SimpleNamespace(
        _pos_dir=object(),
        _correction_tracked_layer=MagicMock(return_value=layer),
        _refresh_lineage_canvas_if_shown=MagicMock(),
        _correction_status=MagicMock(),
    )

    monkeypatch.setattr(mod, "read_validated_tracks", lambda _pos: set())
    monkeypatch.setattr(
        mod,
        "remove_unvalidated_from_data",
        lambda _data, _tracks: SimpleNamespace(changed_pixels=0, changed_frames=0),
    )

    NucleusCorrectionWidget._on_remove_unvalidated_labels(stub)

    stub._refresh_lineage_canvas_if_shown.assert_not_called()
