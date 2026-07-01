# Cellpose Segment + Track: arrival & progressive-enable UX — Implementation Plan

Spec: `docs/superpowers/specs/2026-07-01-cellpose-segment-track-arrival-ux-design.md`

**Goal:** buttons disable themselves (with a tooltip reason) until their precondition
is met, and the status label always names the next available action instead of
sitting blank at idle.

**Files:**
- Modify `src/cellflow/napari/cellpose_segment_track_widget.py` — per-button gate
  `when`/`reason`, status text at idle/bind/segment-done/track-done.
- Modify `src/cellflow/napari/cell_correction_widget.py` — local activate-button
  gating in active-layer mode + a new `selection.events.active` listener.
- Modify `tests/napari/test_cellpose_segment_track_widget.py` — new assertions
  (this file already covers both widgets; there is no separate correction-widget
  test file, confirmed via `git log --follow`).

---

## Task 1 — Per-button gate predicates + reasons (`cellpose_segment_track_widget.py`)

Replace the uniform loop in `_register_gate_controls` (currently all three
Channel-1 action buttons share one `when=own` predicate) with explicit
per-button registrations, plus reason callables for the joint buttons:

```python
def _register_gate_controls(self) -> None:
    g = self.gate

    def _own(btn) -> bool:
        return self._running is None or self._active_btn() is btn

    g.register(self.ch1_params_btn, ControlClass.HARMLESS)
    g.register(self.ch2_params_btn, ControlClass.HARMLESS)
    g.register(
        self.ch1_preview_btn, ControlClass.RUN_VIEWER,
        when=lambda: _own(self.ch1_preview_btn) and self._channel_present(1),
        reason="Bind Channel 1 first — click ⧉.",
    )
    g.register(
        self.ch1_seg_btn, ControlClass.RUN_VIEWER,
        when=lambda: _own(self.ch1_seg_btn) and self._channel_present(1),
        reason="Bind Channel 1 first — click ⧉.",
    )
    g.register(
        self.ch1_track_btn, ControlClass.RUN_VIEWER,
        when=lambda: _own(self.ch1_track_btn) and self._ch1_masks_available(),
        reason=self._ch1_track_reason,
    )
    # Channel 2's actions (preview + run) are joint-only: both require both
    # inputs to be present.
    for btn in self._joint_buttons():
        g.register(
            btn, ControlClass.RUN_VIEWER,
            when=lambda b=btn: _own(b) and self._both_inputs(),
            reason=self._ch2_joint_reason,
        )
    g.recompute()

def _ch1_masks_available(self) -> bool:
    return _layer_name(_CH1_LABEL, "masks") in self.viewer.layers

def _ch1_track_reason(self) -> str:
    if not self._channel_present(1):
        return "Bind Channel 1 first — click ⧉."
    return "Segment Channel 1 first."

def _ch2_joint_reason(self) -> str:
    if not self._channel_present(1):
        return "Bind Channel 1 first."
    return "Bind Channel 2 for joint segmentation."
```

`_action_buttons()` becomes dead (only the old loop used it) — delete it.
`_joint_buttons()` stays (still used).

**Reachability of `gate.recompute()`:** already wired —
`_set_channel_layer`/`_on_layers_changed` call it on bind/layer-list changes
(masks layer appearing after Segment fires `_on_layers_changed` via the
viewer's `inserted` event), and `_set_running` calls it on every run
start/stop. No new event plumbing needed.

## Task 2 — Status label text (`cellpose_segment_track_widget.py`)

1. **Idle hint at construction.** At the end of `__init__` (after
   `self._connect_layer_events()`):
   ```python
   self._status(
       "Bind Channel 1 to begin. "
       "(Optional: also bind Channel 2 for joint segmentation.)"
   )
   ```

2. **After binding Channel 1** — in `_set_channel_layer`, append the next-step
   sentence only for `which == 1`:
   ```python
   if layer is not None:
       label = _CH1_LABEL if which == 1 else _CH2_LABEL
       msg = f"{label} ← layer '{layer.name}'."
       if which == 1:
           msg += " Preview a frame or Segment the full stack."
       self._status(msg)
   ```

3. **After Segment / Track complete.** The existing test suite never drives
   `_run_segment`/`_run_track` end-to-end (both are `thread_worker`-wrapped;
   the module's own docstring convention is "Qt-free compute steps, callable
   directly in tests; the worker just wraps them"). To keep the new message
   text testable without threading, extract it into two module-level pure
   helpers next to `_layer_name`:
   ```python
   def _segment_done_status(masks_name: str, prob_name: str, flow_name: str) -> str:
       return (
           f"Channel 1 → '{masks_name}', '{prob_name}', '{flow_name}'. "
           "Save from the layers. Track to link masks across time."
       )

   def _track_done_status(tracked_name: str) -> str:
       return (
           f"Channel 1 tracked → '{tracked_name}'. Save from the layer. "
           "Select it below to correct."
       )
   ```
   `_run_segment`'s `_done` calls `self._status(_segment_done_status(*names))`;
   `_run_track`'s `_done` calls `self._status(_track_done_status(tracked_name))`.

## Task 3 — Local activate-button gating (`cell_correction_widget.py`)

Add a method (near `_correction_data_available`):

```python
def _sync_active_btn_enabled(self) -> None:
    """Locally gate the activate button in active-layer mode.

    Not routed through the parent widget's ``UiGate`` — this widget is reused
    in disk-mode contexts (the app) where this precondition doesn't apply.
    Kept enabled while already checked so an active session can always be
    turned back off even if the viewer's active-layer selection has since
    moved off the bound Labels layer.
    """
    if not self._active_layer_mode():
        return
    available = self._correction_data_available() or self.active_btn.isChecked()
    self.active_btn.setEnabled(available)
    self.active_btn.setToolTip(
        "Activate correction mode and show correction controls."
        if available
        else "Select a Labels layer to correct — the active layer is not one."
    )
```

Wire it in `_connect_signals` (active-layer mode only — the getattr chain
mirrors `_connect_layer_events`'s guard style in the parent widget for a fake
viewer without `.selection.events`):

```python
if self._active_layer_mode():
    active_events = getattr(
        getattr(getattr(self.viewer.layers, "selection", None), "events", None),
        "active", None,
    )
    if active_events is not None:
        active_events.connect(lambda *_a, **_k: self._sync_active_btn_enabled())
    self.active_btn.toggled.connect(lambda _c: self._sync_active_btn_enabled())
```

Call `self._sync_active_btn_enabled()` once at the end of `__init__` (after
`_connect_signals()`) to paint the correct initial state.

## Task 4 — Tests (`tests/napari/test_cellpose_segment_track_widget.py`)

Add:

- `test_ch1_buttons_disabled_until_bound` — fresh widget: `ch1_preview_btn` /
  `ch1_seg_btn` disabled, tooltip mentions "Bind Channel 1"; bind an image
  layer → both enabled.
- `test_ch1_track_button_reason_changes_with_state` — fresh widget:
  `ch1_track_btn` disabled, reason "Bind Channel 1 first"; bind Channel 1 →
  still disabled, reason becomes "Segment Channel 1 first."; add a
  `[Channel 1] masks` layer directly + call `w._on_layers_changed()` → enabled.
- `test_ch2_buttons_reason_tracks_which_input_is_missing` — fresh widget:
  `ch2_preview_btn`/`ch2_run_btn` disabled, reason "Bind Channel 1 first.";
  bind Channel 1 only → reason becomes "Bind Channel 2 for joint
  segmentation."; bind Channel 2 too → enabled.
- `test_status_label_shows_idle_hint_on_construction` — fresh widget:
  `status_lbl.text()` mentions "Bind Channel 1" and is visible.
- `test_status_label_next_step_after_bind` — bind Channel 1 → status mentions
  "Preview a frame or Segment".
- `test_segment_done_status_names_next_step` / `test_track_done_status_names_next_step`
  — direct unit tests of the two new pure helpers (no widget, no threading).
- `test_activate_button_disabled_until_labels_layer_selected` — fresh
  standalone widget (`stw.CellposeSegmentTrackWidget`): `cell_correction.active_btn`
  disabled, tooltip is the "Select a Labels layer..." text; set
  `viewer.layers.selection.active` to a real `napari.layers.Labels`, call
  `w.cell_correction._sync_active_btn_enabled()` (fake `_Sel` has no real
  event to fire, matching the existing pattern in
  `test_source_pill_darkens_when_bound_layer_removed`) → enabled.
- `test_activate_button_stays_enabled_once_checked_even_if_selection_moves_away`
  — activate correction on a bound Labels layer, then set
  `viewer.layers.selection.active` to a non-Labels layer, call
  `_sync_active_btn_enabled()` → button stays enabled (still checked) so the
  user can toggle it back off.

Run: `uv run --frozen pytest tests/napari/test_cellpose_segment_track_widget.py -q`
(project note: lockfile is stale, always use `--frozen`).
