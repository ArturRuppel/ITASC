"""The ``swap_stepped`` event is the *cheap* post-swap refresh.

Stepping swap candidates (Z / C, or a gallery pick) must stay responsive, so it
refreshes the comet + the selected track's detail strip + the gallery — never
the whole-stack lineage rebuild (``_refresh_lineage_canvas_if_shown``), which
froze the GUI on every keystroke.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock


def test_swap_stepped_does_cheap_detail_refresh_not_full_rebuild(wired_stub):
    stub = wired_stub(
        bind=["_apply_track_path_rebuilt", "_refresh_lineage_detail_if_shown"],
        track_path_btn=SimpleNamespace(isChecked=lambda: True),
        _workspace_splitter=object(),  # focus mode docked
        _lineage_canvas=SimpleNamespace(refresh_detail=MagicMock()),
        _refresh_track_path_overlay=MagicMock(),
        _refresh_track_path_spotlight=MagicMock(),
    )

    stub.events.swap_stepped.emit()

    # Comet + detail strip + gallery refresh …
    stub._refresh_track_path_overlay.assert_called_once_with()
    stub._refresh_track_path_spotlight.assert_called_once_with()
    stub._lineage_canvas.refresh_detail.assert_called_once_with()
    stub._refresh_candidate_gallery_if_shown.assert_called_once_with()
    # … but never the expensive whole-stack lineage rebuild.
    stub._refresh_lineage_canvas_if_shown.assert_not_called()
