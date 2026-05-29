from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
