"""Unit tests for the single-cell *focus* presentation.

Selecting a cell in correction mode focuses it: the nucleus-side overlay is the
single all-tracks layer, and ``_apply_focus_presentation`` just hands the
selected id to that layer's controller, which recolours in place (focused track
viridis-by-time, the rest faint grey) and drops a current-frame tip cross.
Deselecting (``lab == 0``) restores the overview.

We exercise the delegation logic by binding the unbound method to a minimal
stand-in, without building the full Qt widgets.
"""

from __future__ import annotations

import types

from cellflow.napari.nucleus_correction_widget import NucleusCorrectionWidget


def _stub():
    calls = []
    obj = types.SimpleNamespace(
        _all_tracks=types.SimpleNamespace(set_focus=lambda lab: calls.append(lab)),
    )
    return obj, calls


def test_focus_delegates_selected_label_to_all_tracks():
    obj, calls = _stub()
    NucleusCorrectionWidget._apply_focus_presentation(obj, 7)
    assert calls == [7]


def test_deselect_passes_zero_to_all_tracks():
    obj, calls = _stub()
    NucleusCorrectionWidget._apply_focus_presentation(obj, 0)
    assert calls == [0]


def test_falsy_label_is_normalised_to_zero():
    obj, calls = _stub()
    NucleusCorrectionWidget._apply_focus_presentation(obj, None)
    assert calls == [0]
