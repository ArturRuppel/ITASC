"""Single all-tracks overlay for the nucleus correction workflow.

This owns one in-canvas napari ``Tracks`` layer that draws *every* track's
trajectory through the per-frame nucleus centroids, plus a small "tip" Points
layer that marks the focused track's position in the current frame with a cross.

Two presentation states share the one tracks layer:

* **overview** (nothing selected) — all tracks coloured by id, moderately
  transparent;
* **focus** (a cell selected) — the selected track recoloured bright
  viridis-by-time while every other track fades to a faint translucent grey.

The focus switch only rewrites the layer's ``properties``/``color_by`` (an
O(track) write), never its geometry, so selecting cells stays cheap. The
geometry is rebuilt (:meth:`refresh`) only when the label stack itself changes.

The pure geometry/colour maths lives in
:mod:`cellflow.napari._correction_track_path`; this controller is the
layer-lifecycle glue.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from cellflow.napari._correction_track_path import build_all_tracks_data
from cellflow.napari._correction_centroids import (
    FOCUS_CROSS_COLOR,
    NEUTRAL_OVERLAY_COLOR,
)

TRACK_LAYER = "[Correction] Nucleus tracks"
TRACK_TIP_LAYER = "[Correction] Track Tip"
TRACK_TAIL_WIDTH = 4
# Trailing history shown behind the current frame, in frames. Short so the
# overview reads as a comet around the current frame instead of the whole path.
TRACK_TAIL_LENGTH = 15
OVERVIEW_COLORMAP = "hsv"
FOCUS_COLORMAP = "viridis"
OVERVIEW_OPACITY = 0.55
# The filled (by-id) view draws the overview fully opaque to match the filled,
# image-hidden labels; the default (outline) view keeps it translucent.
FILLED_OVERVIEW_OPACITY = 1.0
FOCUS_OPACITY = 1.0
TIP_CROSS_SIZE = 5

_NEUTRAL_TRACKS_COLORMAP = None


def _neutral_tracks_colormap():
    """A single-colour colormap so every track vertex reads :data:`NEUTRAL_OVERLAY_COLOR`.

    A ``Tracks`` layer always colours by a property through a colormap; mapping
    every value to one colour (two identical stops) yields a uniform neutral
    overview regardless of ``color_by``. Built lazily and cached.
    """
    global _NEUTRAL_TRACKS_COLORMAP
    if _NEUTRAL_TRACKS_COLORMAP is None:
        from napari.utils.colormaps import Colormap

        rgba = list(NEUTRAL_OVERLAY_COLOR)
        _NEUTRAL_TRACKS_COLORMAP = Colormap([rgba, rgba], name="neutral_overview")
    return _NEUTRAL_TRACKS_COLORMAP

# Backwards-compatible aliases (the layer name is also referenced by the widget).
TRACK_PATH_LAYER = TRACK_LAYER

_CORRECTION_TRACKS_CLS = None


def _correction_tracks_cls():
    """A ``Tracks`` subclass whose temporal tail fade can be forced on/off.

    napari's ``Tracks.use_fade`` is read-only (it fades whenever the time axis
    is hidden, i.e. always in a T/Y/X viewer). We need the focused track to be
    fully opaque end-to-end (``use_fade`` off → the shader pins every vertex's
    alpha to 1.0) while the overview keeps its fading comet, so the layer carries
    an override flag the controller toggles per focus state. Built lazily so the
    module stays import-light. ``use_fade`` is read back by the vispy layer on
    the appearance events (tail/head/color_by) the controller fires anyway.
    """
    global _CORRECTION_TRACKS_CLS
    if _CORRECTION_TRACKS_CLS is None:
        from napari.layers import Tracks

        class _CorrectionTracks(Tracks):
            def __init__(self, *args, **kwargs) -> None:
                self._fade_override: bool | None = None
                super().__init__(*args, **kwargs)

            @property
            def use_fade(self) -> bool:
                if self._fade_override is not None:
                    return self._fade_override
                return 0 in self._slice_input.not_displayed

            @use_fade.setter
            def use_fade(self, value: bool | None) -> None:
                self._fade_override = None if value is None else bool(value)

        _CORRECTION_TRACKS_CLS = _CorrectionTracks
    return _CORRECTION_TRACKS_CLS


class AllTracksController:
    """Own the single all-tracks layer and its current-frame tip cross."""

    def __init__(
        self,
        viewer,
        *,
        tracked_layer_provider: Callable[[], object | None],
        selected_label_provider: Callable[[], int],
        enabled_provider: Callable[[], bool],
        current_t_provider: Callable[[], int],
        status_callback: Callable[[str], None],
        owned_layers: set[str],
    ) -> None:
        self.viewer = viewer
        self._tracked_layer_provider = tracked_layer_provider
        self._selected_label_provider = selected_label_provider
        self._enabled_provider = enabled_provider
        self._current_t_provider = current_t_provider
        self._status = status_callback
        self._owned_layers = owned_layers
        # Cached from the last refresh so focusing a track can slice its vertices
        # out of the full data without rescanning the stack. All share one order.
        self._data: np.ndarray = np.empty((0, 4), dtype=float)
        self._properties: dict[str, np.ndarray] = {}
        self._row_index: dict[int, np.ndarray] = {}
        self._span: int = 1
        # False = default outline view (neutral single colour, translucent);
        # True = filled view (per-id hsv colour, opaque). Only changes the
        # *overview* presentation — a focused track stays viridis-by-time.
        self._filled_mode: bool = False

    # ── lifecycle ──────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Rebuild the tracks layer from the current stack, then re-apply focus."""
        if not self._enabled_provider():
            self.clear()
            return
        layer = self._tracked_layer_provider()
        if layer is None:
            self.clear()
            return
        data = np.asarray(layer.data)
        rows, properties, row_index = build_all_tracks_data(data)
        self._data = rows
        self._properties = properties
        self._row_index = row_index
        if len(rows) == 0:
            self.clear()
            return
        self._span = max(int(data.shape[0]) if data.ndim == 3 else 1, 1)
        self._ensure_layer()
        self.set_focus(int(self._selected_label_provider() or 0))
        self._status(f"Tracks: {len(row_index)} track(s).")

    def clear(self) -> None:
        """Remove the tracks + tip layers from the viewer."""
        for name in (TRACK_LAYER, TRACK_TIP_LAYER):
            if name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[name])
            self._owned_layers.discard(name)
        self._data = np.empty((0, 4), dtype=float)
        self._properties = {}
        self._row_index = {}

    # ── view mode (outline-neutral ↔ filled-by-id overview) ───────────────────

    def set_filled_mode(self, filled: bool) -> None:
        """Switch the overview between neutral (outline view) and by-id (filled view).

        Re-applies the current focus so the live layer picks up the new overview
        colour/opacity; a focused single track is unaffected (still by time).
        """
        self._filled_mode = bool(filled)
        if self._tracks_layer() is not None:
            self.set_focus(int(self._selected_label_provider() or 0))

    def _overview_opacity(self) -> float:
        return FILLED_OVERVIEW_OPACITY if self._filled_mode else OVERVIEW_OPACITY

    def _apply_overview_coloring(self, layer) -> None:
        """Colour the overview by mode: per-id hsv (filled) or one neutral colour.

        A ``Tracks`` layer's ``colormap`` is a *registered name*, so the single
        neutral colour is supplied through ``colormaps_dict`` (keyed by the
        ``color_by`` property, which bypasses the name registry and the
        property normalisation). Assigning ``color_by`` re-runs the recolour.
        """
        if self._filled_mode:
            layer.colormaps_dict = {}
            try:
                layer.colormap = OVERVIEW_COLORMAP
            except Exception:
                pass
        else:
            layer.colormaps_dict = {"track_id": _neutral_tracks_colormap()}
            layer.color_by = "track_id"

    # ── focus / current-frame presentation ───────────────────────────────────

    def set_focus(self, lab: int) -> None:
        """Switch the layer between the overview and a single focused track.

        Focus mode draws *only* the selected track (the surrounding tracks are
        dropped from the layer entirely — a Tracks layer has no per-track
        visibility) coloured by time and fully opaque, image-like. ``lab == 0``
        (or an absent track) restores the all-tracks overview and hides the tip
        cross. A no-op when the tracks layer is not present (toggle off).
        """
        lab = int(lab or 0)
        layer = self._tracks_layer()
        if layer is None:
            return
        if not lab or lab not in self._row_index:
            # Overview: every track, a short trailing comet that fades out behind
            # the current frame (no forward head), all dimmed together. Park
            # color_by on the always-present 'track_id' before the data swap so
            # napari doesn't warn about the old key vanishing.
            layer.use_fade = None  # default (fade on, scaled by tail_length)
            layer.color_by = "track_id"
            layer.data = self._data
            layer.properties = self._properties
            layer.color_by = "track_id"
            self._apply_overview_coloring(layer)
            layer.tail_length = min(TRACK_TAIL_LENGTH, self._span)
            layer.head_length = 0
            layer.opacity = self._overview_opacity()
            self._hide_tip()
            return

        # Focus: keep only the selected track's vertices, coloured by its time
        # gradient. use_fade off pins every vertex's alpha to 1.0 so the whole
        # trajectory reads fully opaque (image-like) on every frame; set it before
        # the tail/head/color_by writes the vispy layer reads use_fade back from.
        rows = self._row_index[lab]
        layer.use_fade = False
        # Park color_by on 'track_id' (always present) across the data swap so
        # napari doesn't warn about the previous key vanishing, then colour by time.
        layer.color_by = "track_id"
        layer.data = self._data[rows]
        layer.properties = {
            "track_id": self._properties["track_id"][rows],
            "time": self._properties["time"][rows],
        }
        layer.color_by = "time"
        try:
            layer.colormap = FOCUS_COLORMAP
        except Exception:
            pass
        layer.tail_length = self._span
        layer.head_length = self._span
        layer.opacity = FOCUS_OPACITY
        self.set_current_frame(int(self._current_t_provider()))
        self._restore_active_layer()

    def set_current_frame(self, t: int) -> None:
        """Move/show the tip cross at the focused track's centroid in frame ``t``."""
        lab = int(self._selected_label_provider() or 0)
        if not self._enabled_provider() or not lab:
            self._hide_tip()
            return
        layer = self._tracked_layer_provider()
        if layer is None:
            self._hide_tip()
            return
        data = np.asarray(layer.data)
        if data.ndim != 3 or not (0 <= int(t) < data.shape[0]):
            self._hide_tip()
            return
        mask = data[int(t)] == lab
        if not mask.any():
            self._hide_tip()
            return
        ys, xs = np.nonzero(mask)
        self._update_tip(np.asarray([[float(ys.mean()), float(xs.mean())]]))

    def spotlight_mask(self, _t: int, lab: int, _default_mask):
        """Widen the inner widget's spotlight to the selected track's whole union."""
        if not self._enabled_provider() or not lab:
            return None
        layer = self._tracked_layer_provider()
        if layer is None:
            return None
        data = np.asarray(layer.data)
        if data.ndim != 3:
            return None
        union = np.any(data == int(lab), axis=0)
        return union if union.any() else None

    # ── layer plumbing ───────────────────────────────────────────────────────

    def _tracks_layer(self):
        from napari.layers import Tracks

        if TRACK_LAYER in self.viewer.layers:
            layer = self.viewer.layers[TRACK_LAYER]
            if isinstance(layer, Tracks):
                return layer
        return None

    def _ensure_layer(self) -> None:
        """Create the tracks layer if absent; ``set_focus`` sets its data/colours.

        Built from the cached full data with overview defaults so there's no
        flash before the ``set_focus`` call that follows every refresh.
        """
        if self._tracks_layer() is not None:
            return
        if TRACK_LAYER in self.viewer.layers:
            self.viewer.layers.remove(TRACK_LAYER)
        # Build the fade-controllable Tracks subclass directly (rather than
        # viewer.add_tracks, which always makes a plain Tracks) so focus mode can
        # pin the focused track fully opaque.
        layer = _correction_tracks_cls()(
            self._data,
            name=TRACK_LAYER,
            properties=self._properties,
            color_by="track_id",
            colormap=OVERVIEW_COLORMAP,
            tail_width=TRACK_TAIL_WIDTH,
            tail_length=min(TRACK_TAIL_LENGTH, self._span),
            head_length=0,
            blending="translucent",
            opacity=self._overview_opacity(),
        )
        self.viewer.add_layer(layer)
        # Construction takes a registered colormap *name*; apply the mode's real
        # overview colouring (the neutral colormaps_dict, etc.) now. The
        # ``set_focus`` that follows ``refresh`` re-applies it for the live state.
        self._apply_overview_coloring(layer)
        self._owned_layers.add(TRACK_LAYER)
        self._restore_active_layer()

    def _update_tip(self, points: np.ndarray) -> None:
        if TRACK_TIP_LAYER in self.viewer.layers:
            layer = self.viewer.layers[TRACK_TIP_LAYER]
            layer.data = points
            layer.visible = True
        else:
            self.viewer.add_points(
                points,
                name=TRACK_TIP_LAYER,
                ndim=2,
                symbol="cross",
                size=TIP_CROSS_SIZE,
                face_color=[FOCUS_CROSS_COLOR],
                border_color=[FOCUS_CROSS_COLOR],
                opacity=1.0,
            )
        self._owned_layers.add(TRACK_TIP_LAYER)
        self._restore_active_layer()

    def _hide_tip(self) -> None:
        if TRACK_TIP_LAYER in self.viewer.layers:
            self.viewer.layers[TRACK_TIP_LAYER].visible = False

    def _restore_active_layer(self) -> None:
        try:
            self.viewer.layers.selection.active = self._tracked_layer_provider()
        except Exception:
            pass
