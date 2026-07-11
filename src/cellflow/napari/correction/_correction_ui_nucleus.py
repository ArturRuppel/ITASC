"""Qt construction for :class:`NucleusCorrectionWidget`.

The widget is large enough that its one-time control assembly — buttons,
parameter sliders, the embedded :class:`CorrectionWidget`, the reveal sections,
the toolbar and the signal wiring — is kept here so the widget module is about
behaviour rather than scaffolding. :func:`build_nucleus_correction_ui` populates
the passed widget in place (setting the same ``w.<name>`` attributes the rest of
the class reads), mirroring how the shared header / toolbar builders in
:mod:`cellflow.napari.correction._correction_ui` already work.
"""

from __future__ import annotations

from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from cellflow.napari._widget_helpers import (
    btn as _btn,
    dslider as _dslider,
    heading as _heading,
    make_status as _make_status,
    tool_btn as _tool_btn,
)
from cellflow.napari.correction._correction_ui import (
    build_correction_header,
    build_correction_toolbar,
    build_shortcuts_widget,
    flatten_embedded_section,
)
from cellflow.napari.ui_style import danger_button
from cellflow.napari.widgets import CollapsibleSection
from cellflow.napari.correction.correction_widget import CorrectionWidget


def build_nucleus_correction_ui(w) -> None:
    """Build and wire all of ``w``'s correction controls (run once from init)."""
    root = QVBoxLayout(w)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)

    _init_candidate_refresh_timer(w)

    inner = QWidget(w)
    group_lay = QVBoxLayout(inner)
    group_lay.setContentsMargins(0, 0, 0, 0)
    group_lay.setSpacing(6)

    _build_action_buttons(w)
    _build_status_labels(w)
    _build_correction_subwidget(w)
    _build_extend_retrack_section(w)
    _build_shortcuts_section(w)
    _build_toolbar(w)
    _build_view_toggle_buttons(w)
    _assemble_reveal_area(w, group_lay)

    # The full-width top bar carries the title, the activate / shortcuts /
    # params toggles, the checkable view toggles, and a single one-line
    # status (the save/action status only) right-aligned. The track /
    # validated summary lives in the tracking-overview panel instead.
    w.header, w.header_lbl = build_correction_header(
        w,
        shortcuts_btn=w.shortcuts_btn,
        params_btn=w.params_btn,
        active_btn=w.active_btn,
        finalize_btn=getattr(w, "_finalize_header_btn", None),
        view_toggle_btns=(
            w.track_path_btn,
            w.spotlight_btn,
            w.filled_view_btn,
        ),
        status_lbl=w.status_lbl,
    )
    # ``section`` is the full-width reveal area (params + shortcuts) shown
    # below the top bar; both it and the header are reparented between the
    # plugin dock (inactive) and the workspace dock (active).
    w.section = CollapsibleSection("Correction", inner, expanded=False)
    w.section._toggle.setVisible(False)
    w.section._toggle.setEnabled(False)

    w.correction_active_btn = w.active_btn
    w.correction_shortcuts_btn = w.shortcuts_btn
    w.correction_status_lbl = w.status_lbl
    w.correction_mode_section = w.section
    w._correction_active_content_visible = False
    _connect_signals(w)
    # Start collapsed: the inactive plugin-dock entry point shows only the
    # on/off button (title + toggles appear once correction is active).
    w._sync_correction_panel_visibility()


def _init_candidate_refresh_timer(w) -> None:
    # Searching + rendering the candidate gallery is expensive, so debounce
    # it: every refresh request (re)starts this timer and the actual rebuild
    # only fires once the frame has been still for the interval — fast
    # scrubbing no longer recomputes candidates for every frame it sweeps.
    w._candidate_refresh_timer = QTimer(w)
    w._candidate_refresh_timer.setSingleShot(True)
    w._candidate_refresh_timer.setInterval(200)
    w._candidate_refresh_timer.timeout.connect(w._refresh_candidate_gallery_now)


def _build_action_buttons(w) -> None:
    w.active_btn = _tool_btn(
        "⏻",
        "Activate correction mode and show correction layers and controls.",
        checkable=True,
    )
    w.active_btn.setToolTip(
        "Activate correction mode and show correction layers and controls."
    )
    w.params_btn = _tool_btn(
        "⚙", "Show correction parameters.", checkable=True
    )
    w.shortcuts_btn = _tool_btn(
        "📖", "Show correction shortcuts.", checkable=True
    )

    w.save_tracked_btn = _tool_btn(
        "💾", "Save corrected tracked nucleus labels to disk (S)."
    )
    w.extend_back_btn = _tool_btn(
        "◀", "Extend selected track one frame backward (A)."
    )
    w.extend_fwd_btn = _tool_btn(
        "▶", "Extend selected track one frame forward (D)."
    )
    w.retrack_back_btn = _tool_btn(
        "↶", "Retrack all labels backward from current frame (Q)."
    )
    w.retrack_fwd_btn = _tool_btn(
        "↷", "Retrack all labels forward from current frame (E)."
    )
    w.swap_smaller_btn = _tool_btn(
        "⮂", "Swap selected cell with the next smaller candidate fragment (Z)."
    )
    w.swap_larger_btn = _tool_btn(
        "⮀", "Swap selected cell with the next larger candidate fragment (C)."
    )
    w.reassign_ids_btn = _tool_btn(
        "#",
        "Reassign cell IDs to contiguous range 1-N (validated tracks first, "
        "then by earliest start frame, then by longest track).",
    )
    w.validate_track_btn = _tool_btn(
        "✓", "Lock selected cell geometry in every frame where it appears (V)."
    )
    w.anchor_here_btn = _tool_btn(
        "⚓", "Anchor selected cell identity at the current frame (B)."
    )
    w.annotate_db_btn = _tool_btn(
        "✎", "Apply saved validations and anchors to the Ultrack database."
    )
    w.remove_unvalidated_btn = _tool_btn(
        "🗑",
        "Remove nucleus label pixels not marked validated for their frame.",
    )
    danger_button(w.remove_unvalidated_btn)

    w.commit_btn = _btn(
        "Commit",
        "Reassign cell IDs, remove unvalidated labels, and save tracked labels.",
    )


def _build_status_labels(w) -> None:
    w.status_lbl = _make_status()
    # Drop the smaller status font so the status line reads at the same
    # size as the rest of the controls column. Keep it to a single line so
    # it never wraps and pushes the top-bar title pill onto two rows.
    w.status_lbl.setStyleSheet("")
    w.status_lbl.setWordWrap(False)

    # The track / validated summary now lives in the tracking-overview title
    # (see LineageCanvasController). This label is kept as a hidden sink so
    # the existing counter-refresh plumbing keeps working without surfacing
    # a second status line in the top bar; it is parented (never shown) so
    # toggling its visibility can't spawn a stray top-level window.
    w.validation_counter_lbl = QLabel("", w)
    w.validation_counter_lbl.setVisible(False)


def _build_correction_subwidget(w) -> None:
    w.correction_widget = CorrectionWidget(
        w.viewer,
        show_activate_btn=False,
        show_shortcuts=False,
        inspector_first=True,
        show_cleanup=False,
        # The lineage canvas is the navigation surface now — drop the
        # redundant "Inspect cell" group from the correction column.
        show_inspector=False,
    )
    w.correction_widget.set_edit_callback(w._edit_callback)
    w.correction_widget.set_protected_mask_callback(
        w._manual_correction_protected_mask
    )
    w.correction_widget.set_intensity_frame_callback(
        w._correction_intensity_frame
    )
    # Use an additive listener, not set_selection_callback: the workflow
    # widget owns that single slot and would otherwise clobber the comet's
    # rebuild-on-selection (so it would only build on first checkbox tick).
    w.correction_widget.add_selection_listener(
        w._on_track_selection_changed
    )
    w.correction_widget.set_spotlight_mask_provider(
        w._track_path_spotlight_mask
    )
    w.correction_widget._status.setVisible(False)


def _build_extend_retrack_section(w) -> None:
    # Wide-and-short reveal area: each remaining parameter gets its own
    # column so the sliders spread horizontally across the full-width panel
    # rather than stacking. Greedy overwrite and the outline view are now
    # always-on defaults, so neither is surfaced as a control here.
    extend_retrack_inner = QWidget(w)
    extend_retrack_lay = QVBoxLayout(extend_retrack_inner)
    extend_retrack_lay.setContentsMargins(0, 0, 0, 0)
    extend_retrack_lay.setSpacing(6)

    columns = QHBoxLayout()
    columns.setContentsMargins(0, 0, 0, 0)
    columns.setSpacing(16)

    def _column(*items) -> QWidget:
        col = QWidget()
        lay = QVBoxLayout(col)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        for item in items:
            if isinstance(item, QWidget):
                lay.addWidget(item)
            else:
                lay.addLayout(item)
        lay.addStretch(1)
        return col

    # Greedy overwrite is now the always-on default: keep the (hidden)
    # checkbox checked so the extend/retrack paint paths that read its state
    # still behave greedily, but drop it from the UI.
    w.extend_greedy_overwrite_check = QCheckBox("Greedy overwrite")
    w.extend_greedy_overwrite_check.setChecked(True)

    w.retrack_max_dist_spin = _dslider(0, 500, 20.0, 1.0, 1)
    # Scoring weights for the retrack frame matcher. Kept under the
    # ``extend_*`` attribute names for settings back-compat; extend no longer
    # uses them.
    w.extend_area_weight_spin = _dslider(0, 10, 1.0, 0.1, 2)
    w.extend_iou_weight_spin = _dslider(0, 10, 1.0, 0.1, 2)
    w.extend_distance_weight_spin = _dslider(0, 10, 0.05, 0.01, 3)

    # Each retrack weight + the cell-radius control gets its own column
    # (stretch=1) so they fan out across the wide reveal area. The cell-radius
    # slider (middle-click cell creation) is reused straight from the embedded
    # correction widget, dropped under a plain "Cell Radius" heading so it
    # matches the other sliders rather than carrying its own inline label.
    for heading_text, widget in (
        ("Max distance", w.retrack_max_dist_spin),
        ("Area weight", w.extend_area_weight_spin),
        ("IoU weight", w.extend_iou_weight_spin),
        ("Distance weight", w.extend_distance_weight_spin),
        ("Cell Radius", w.correction_widget._cell_radius_spin),
    ):
        columns.addWidget(_column(_heading(heading_text), widget), stretch=1)
    extend_retrack_lay.addLayout(columns)

    w.extend_retrack_params_section = CollapsibleSection(
        "Extend / Retrack Parameters",
        extend_retrack_inner,
        expanded=False,
    )
    flatten_embedded_section(w.extend_retrack_params_section)
    w.extend_retrack_params_section.setVisible(False)
    w.extend_params_section = w.extend_retrack_params_section
    w.retrack_params_section = w.extend_retrack_params_section


def _build_shortcuts_section(w) -> None:
    # The disclaimer / attribution label rides at the bottom of the wide
    # shortcuts panel now (reparented out of the embedded correction widget).
    w.shortcuts_section = CollapsibleSection(
        "Correction Shortcuts",
        build_shortcuts_widget(w.correction_widget._attrib_lbl),
        expanded=False,
    )
    flatten_embedded_section(w.shortcuts_section)
    w.shortcuts_section.setVisible(False)
    w.correction_widget.setVisible(False)


def _build_toolbar(w) -> None:
    # Extend / swap are driven by clicking into the candidate gallery now,
    # so they're dropped from the toolbar (the A/D/Z/C shortcuts still work).
    groups = [
        (w.save_tracked_btn,),
        (w.retrack_back_btn, w.retrack_fwd_btn),
        (w.validate_track_btn, w.anchor_here_btn),
        (w.annotate_db_btn,),
        (w.reassign_ids_btn, w.remove_unvalidated_btn),
    ]
    # The host workflow widget's Finalize button rides at the tail of the
    # toolbar (its own ruled group) — finalizing is the natural last step
    # once the tracks are corrected.
    if getattr(w, "_finalize_btn", None) is not None:
        groups.append((w._finalize_btn,))
    w.toolbar = build_correction_toolbar(w, groups)
    w.toolbar.setVisible(False)


def _build_view_toggle_buttons(w) -> None:
    # View toggles ride in the top bar as checkable icon tool-buttons; the
    # old "Lineage canvas" toggle is gone — the accordion is the always-on
    # main surface. Tooltips carry the longer descriptions the old check
    # captions spelled out.
    w.track_path_btn = _tool_btn(
        "👁",
        "Track path: paint the selected track's whole trajectory as a fading "
        "comet (viridis, oldest→newest) with a frame number in each mask.",
        checkable=True,
    )
    w.spotlight_btn = _tool_btn(
        "🔦",
        "Spotlight: dim everything outside the selected cell. Off = mark the "
        "selection with a plain yellow border instead (the rest of the frame "
        "stays at full brightness).",
        checkable=True,
    )
    # Spotlight is the default selection indicator, so the toggle starts on.
    w.spotlight_btn.setChecked(True)
    w.filled_view_btn = _tool_btn(
        "🎨",
        "Filled labels (by ID): hide the cell + nucleus images and draw the "
        "labels and tracks opaque and filled (not outlines), coloured by ID. "
        "Off = the default outline view in one neutral colour.",
        checkable=True,
    )
    # The candidate gallery is shown/hidden by its own ✕ button + slim
    # show-tab (see CollapsiblePane), not a top-bar toggle.
    for button in (w.track_path_btn, w.spotlight_btn, w.filled_view_btn):
        button.setVisible(False)
    w.validation_counter_lbl.setVisible(False)


def _assemble_reveal_area(w, group_lay: QVBoxLayout) -> None:
    # The section body is the full-width reveal area below the top bar: the
    # params and shortcuts panels, openable independently. The toolbar, the
    # view toggles and the status now live elsewhere (the toolbar in the body
    # splitter's thin left column, the toggles + status in the top bar). The
    # embedded correction widget is a logic holder only — its visible bits
    # (outline + spawn controls, attribution label) were reparented into the
    # params / shortcuts panels — so it is kept hidden here to stay owned.
    group_lay.addWidget(w.extend_retrack_params_section)
    group_lay.addWidget(w.shortcuts_section)
    w.correction_widget.setVisible(False)
    group_lay.addWidget(w.correction_widget)


def _connect_signals(w) -> None:
    w.save_tracked_btn.clicked.connect(w._on_save_tracked)
    w.reassign_ids_btn.clicked.connect(w._on_reassign_ids)
    w.validate_track_btn.clicked.connect(w._on_validate_track)
    w.anchor_here_btn.clicked.connect(w._on_anchor_here)
    w.annotate_db_btn.clicked.connect(w._on_annotate_database)
    w.extend_back_btn.clicked.connect(w._on_extend_backward)
    w.extend_fwd_btn.clicked.connect(w._on_extend_forward)
    w.retrack_back_btn.clicked.connect(w._on_retrack_backward)
    w.retrack_fwd_btn.clicked.connect(w._on_retrack_forward)
    w.swap_smaller_btn.clicked.connect(
        lambda: w._on_swap_step(direction="smaller")
    )
    w.swap_larger_btn.clicked.connect(
        lambda: w._on_swap_step(direction="larger")
    )
    w.remove_unvalidated_btn.clicked.connect(w._on_remove_unvalidated_labels)
    w.commit_btn.clicked.connect(w._on_commit)
    w.track_path_btn.toggled.connect(w._on_toggle_track_path)
    w.spotlight_btn.toggled.connect(w._on_toggle_spotlight)
    w.filled_view_btn.toggled.connect(w._on_toggle_filled_view)
    w.params_btn.toggled.connect(w._on_correction_params_button_toggled)
    w.shortcuts_btn.toggled.connect(w._on_correction_shortcuts_button_toggled)
    w.active_btn.toggled.connect(w._on_correction_active_button_toggled)
    w.correction_widget._activate_btn.toggled.connect(
        w._on_correction_mode_toggled
    )
    w._install_correction_shortcuts()
