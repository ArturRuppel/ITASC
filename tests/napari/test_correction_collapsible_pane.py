"""Coverage for the collapsible workspace pane + the host's keep-one-open rule.

The candidate gallery and the tracking-overview accordion each live in a
``CollapsiblePane``: a header with a ✕ that collapses the panel to a slim
full-height show-tab (the ▸), which expands it again. The host wires both panes
so that hiding one while the other is already collapsed re-opens that other one —
the workspace never goes fully blank.

A QApplication is required, so the module skips cleanly if Qt cannot start
headless.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication, QLabel, QToolButton  # noqa: E402

from cellflow.napari.correction._correction_ui import CollapsiblePane, _PANE_STRIP_W  # noqa: E402
from cellflow.napari.correction.nucleus_correction_widget import NucleusCorrectionWidget  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


def _hide_button(pane: CollapsiblePane) -> QToolButton:
    return next(b for b in pane.findChildren(QToolButton) if b.text() == "✕")


def _show_tab(pane: CollapsiblePane) -> QToolButton:
    return next(b for b in pane.findChildren(QToolButton) if b.text() == "▸")


def test_pane_starts_expanded_with_content_and_title(_app):
    content = QLabel("body")
    pane = CollapsiblePane(content, title="Candidate gallery")

    assert not pane.is_collapsed()
    assert pane._stack.currentIndex() == 0  # the content page
    assert content.isVisibleTo(pane)


def test_hide_button_collapses_to_a_pinned_show_tab(_app):
    content = QLabel("body")
    pane = CollapsiblePane(content, title="Candidate gallery")
    flips: list[bool] = []
    pane.collapsed_changed.connect(flips.append)

    _hide_button(pane).click()

    assert pane.is_collapsed()
    assert pane._stack.currentIndex() == 1  # the slim show-tab
    # Collapsed → pinned narrow to the strip width.
    assert pane.maximumWidth() == _PANE_STRIP_W
    assert pane.minimumWidth() == _PANE_STRIP_W
    assert flips == [True]


def test_show_tab_expands_again_and_releases_width(_app):
    pane = CollapsiblePane(QLabel("body"), title="Candidate gallery")
    pane.set_collapsed(True)
    flips: list[bool] = []
    pane.collapsed_changed.connect(flips.append)

    _show_tab(pane).click()

    assert not pane.is_collapsed()
    assert pane._stack.currentIndex() == 0
    assert pane.maximumWidth() > _PANE_STRIP_W  # width released
    assert pane.minimumWidth() == 0
    assert flips == [False]


def test_set_collapsed_is_idempotent(_app):
    pane = CollapsiblePane(QLabel("body"), title="x")
    flips: list[bool] = []
    pane.collapsed_changed.connect(flips.append)

    pane.set_collapsed(False)  # already expanded → no-op, no signal
    assert flips == []

    pane.set_collapsed(True)
    pane.set_collapsed(True)  # already collapsed → no second signal
    assert flips == [True]


def test_host_keeps_at_least_one_panel_open(_app):
    # Hiding a pane while the other is already collapsed re-opens that other one,
    # so the workspace never goes blank. Real CollapsiblePanes (their is_collapsed
    # / set_collapsed are exercised); the size + refresh hooks are stubbed.
    gallery = CollapsiblePane(QLabel("gallery"), title="Candidate gallery")
    accordion = CollapsiblePane(QLabel("overview"), title="Tracking overview")
    stub = SimpleNamespace(
        _gallery_pane=gallery,
        _accordion_pane=accordion,
        _apply_workspace_panel_sizes=MagicMock(),
        _gallery_is_shown=lambda: not gallery.is_collapsed(),
        _refresh_candidate_gallery_if_shown=MagicMock(),
    )

    # Collapse the accordion first (gallery still open) — allowed, sizes applied.
    accordion.set_collapsed(True)
    NucleusCorrectionWidget._on_workspace_panel_toggled(stub, accordion, True)
    assert accordion.is_collapsed() and not gallery.is_collapsed()
    assert stub._apply_workspace_panel_sizes.call_count == 1

    # Now collapse the gallery too: the handler must re-open the accordion.
    gallery.set_collapsed(True)
    NucleusCorrectionWidget._on_workspace_panel_toggled(stub, gallery, True)
    assert gallery.is_collapsed()
    assert not accordion.is_collapsed()  # re-opened to keep one visible
