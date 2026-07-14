"""Reassign-IDs must hand the lowest IDs to validated tracks first.

When the user reassigns cell IDs to a contiguous 1-N range, validated tracks
should claim ``1, 2, …`` ahead of any unvalidated tracks, so the well-curated
tracks get the stable low numbers. ``_validated_track_ids`` lists the priority
group and both the interactive and commit reassign paths feed it (with the
stack) to ``track_order_by_frame_and_size`` and ``reassign_ids_ordered``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from itasc.napari.correction.nucleus_correction_widget import NucleusCorrectionWidget


def _stub_with_validated(validated: dict[int, set[int]]) -> SimpleNamespace:
    return SimpleNamespace(
        _pos_dir=object(),
        _dependency=lambda _name: (lambda _pos: validated),
    )


def test_validated_track_ids_is_sorted_ids() -> None:
    stub = _stub_with_validated({7: {0}, 3: {0, 1}, 5: {2}})

    order = NucleusCorrectionWidget._validated_track_ids(stub)

    assert order == [3, 5, 7]


def test_validated_track_ids_empty_without_project() -> None:
    stub = SimpleNamespace(_pos_dir=None)

    assert NucleusCorrectionWidget._validated_track_ids(stub) == []


def test_commit_reassign_gives_validated_tracks_low_ids() -> None:
    # Track 9 is validated; 3 and 5 are not. After reassign, 9 -> 1.
    layer = SimpleNamespace(
        data=np.array([[0, 5, 9], [3, 9, 5]], dtype=np.uint32),
    )
    stub = SimpleNamespace(
        _pos_dir=object(),
        _refresh_correction_label_visuals=lambda: None,
        _validated_track_ids=lambda: [9],
    )

    captured: dict = {}

    def _remap(_pos, old_to_new):
        captured["old_to_new"] = old_to_new

    import itasc.napari.correction.nucleus_correction_widget as widget_mod

    original = widget_mod.remap_validated_tracks
    widget_mod.remap_validated_tracks = _remap
    try:
        n_cells = NucleusCorrectionWidget._commit_reassign_ids(stub, layer)
    finally:
        widget_mod.remap_validated_tracks = original

    assert n_cells == 3
    assert captured["old_to_new"][9] == 1
    # Validated track 9 now carries id 1 everywhere it appeared.
    assert layer.data[0, 2] == 1
    assert layer.data[1, 1] == 1
