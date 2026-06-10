from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerViewState:
    visibility: dict[str, bool]
    active: str | None
    selected: tuple[str, ...]


def capture_layer_view_state(layers: Any) -> LayerViewState:
    """Capture layer visibility and selection before entering correction mode."""
    active = layers.selection.active
    return LayerViewState(
        visibility={layer.name: bool(layer.visible) for layer in layers},
        active=active.name if active is not None else None,
        selected=tuple(layer.name for layer in layers.selection),
    )


def hide_all_layers(layers: Any) -> None:
    """Hide all current layers while correction-mode layers are loaded."""
    for layer in list(layers):
        layer.visible = False


def restore_layer_view_state(layers: Any, state: LayerViewState | None) -> None:
    """Restore captured visibility and selection for layers that still exist."""
    if state is None:
        return
    for name, visible in state.visibility.items():
        if name in layers:
            layers[name].visible = bool(visible)
    layers.selection.clear()
    for name in state.selected:
        if name in layers:
            layers.selection.add(layers[name])
    if state.active is not None and state.active in layers:
        layers.selection.active = layers[state.active]


def remove_owned_layers(layers: Any, owned_layer_names: set[str]) -> None:
    """Remove registered correction-owned layers and clear the ownership set."""
    for name in list(owned_layer_names):
        if name in layers:
            layers.remove(layers[name])
    owned_layer_names.clear()


def detach_higher_dim_stacks(
    layers: Any, *, max_ndim: int, keep_names: Any = ()
) -> list:
    """Remove and return layers with more dimensions than the correction stack.

    Correction shows a ``(T, Y, X)`` stack — a single slider whose axis 0 is the
    frame. A pre-loaded 3D z-stack carried as ``(T, Z, Y, X)`` data has a higher
    rank, so napari (which aligns layer axes to the right) treats its trailing
    ``(T, Y, X)`` as the global ``(Z, Y, X)`` and the frame slider no longer
    scrubs the correction frames. Such higher-rank layers are *removed* — not
    merely hidden, because napari derives the dims sliders from every layer
    regardless of visibility.

    Only layers whose rank *exceeds* ``max_ndim`` (the correction stack's own
    rank) are detached, so same-rank intensity/label layers — even ones with a
    different frame count — are left alone. The detached ``Layer`` objects are
    returned verbatim so they can be re-appended on restore with their data,
    contrast limits and colormap intact (nothing is reloaded from disk).
    ``keep_names`` (e.g. the correction-owned layers) are never detached.
    """
    keep = set(keep_names)
    detached: list = []
    for layer in list(layers):
        if getattr(layer, "name", None) in keep:
            continue
        ndim = getattr(getattr(layer, "data", None), "ndim", None)
        if ndim is not None and ndim > int(max_ndim):
            detached.append(layer)
            layers.remove(layer)
    return detached


def reattach_layers(layers: Any, detached: list) -> None:
    """Re-add layers removed by :func:`detach_higher_dim_stacks`, verbatim.

    Skips any whose name is already back in the list (e.g. re-loaded meanwhile),
    so restore is idempotent.
    """
    for layer in detached:
        if getattr(layer, "name", None) not in layers:
            layers.append(layer)


class CorrectionViewStateMixin:
    """Shared correction-mode view-state + owned-layer handling.

    Mixed into both the cell and nucleus correction widgets, which were carrying
    byte-identical copies of these methods. The host must expose ``self.viewer``
    and ``self.correction_status_lbl``, and track ``self._correction_view_state``
    (captured on activate, restored + cleared on deactivate),
    ``self._correction_owned_layers`` (the layer names the widget added and must
    tear down) and the ``self._correction_dirty`` flag.
    """

    _correction_view_state: LayerViewState | None
    _correction_owned_layers: set[str]
    _correction_dirty: bool

    def _capture_correction_view_state(self) -> None:
        self._correction_view_state = capture_layer_view_state(self.viewer.layers)

    def _restore_correction_view_state(self) -> None:
        restore_layer_view_state(self.viewer.layers, self._correction_view_state)
        self._correction_view_state = None

    def _remove_correction_owned_layers(self) -> None:
        remove_owned_layers(self.viewer.layers, self._correction_owned_layers)

    def _current_t(self) -> int:
        step = getattr(getattr(self.viewer, "dims", None), "current_step", (0,))
        return int(step[0]) if len(step) >= 1 else 0

    def _correction_status(self, msg: str) -> None:
        self.correction_status_lbl.setText(msg)
        self.correction_status_lbl.setVisible(bool(msg))
        lowered = msg.lower()
        if "unsaved" in lowered:
            self._correction_dirty = True
        elif lowered.startswith("saved") or lowered.startswith("loaded"):
            self._correction_dirty = False
        if msg:
            logger.info(msg)
