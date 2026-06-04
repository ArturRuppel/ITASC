"""Docked candidate-gallery state for the correction workspace.

Owns the :class:`CandidateGalleryPanel` and keeps its three columns
(extend-backward · swap · extend-forward) in sync with the selected track and
current frame: on :meth:`refresh` it enumerates the swap lineage
(``list_swap_candidates``) and the extend shortlist in each direction
(``list_extend_candidates``), renders each as a thumbnail strip over the
intensity frame (via the pure :func:`build_candidate_strip`), and caches the
candidates by key. A thumbnail click is routed back through the injected
``apply_swap`` / ``apply_extend`` callbacks — the host widget owns the actual
label mutation, history and protection — after which the gallery refreshes.

The DB enumerators are injected so the click-routing and strip-building are
unit-testable with fakes; the colormap/outline adaptation mirrors
:class:`~cellflow.napari.lineage_canvas_controller.LineageCanvasController`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from collections.abc import Callable

import numpy as np

from cellflow.napari._correction_candidates import (
    CandidateSpec,
    CandidateStrip,
    build_candidate_strip,
)
from cellflow.napari._correction_candidate_panel import CandidateGalleryPanel
from cellflow.tracking_ultrack.extend import list_extend_candidates
from cellflow.tracking_ultrack.swap_candidate import list_swap_candidates

logger = logging.getLogger(__name__)

_NEUTRAL_OUTLINE = (0.75, 0.75, 0.75)

# direction-keyed extend columns
_EXTEND_DIRECTION = {
    CandidateGalleryPanel.EXTEND_BACKWARD: "backward",
    CandidateGalleryPanel.EXTEND_FORWARD: "forward",
}


class CandidateGalleryController:
    """Own the docked candidate galleries and apply a clicked candidate."""

    def __init__(
        self,
        viewer,
        *,
        tracked_data_provider: Callable[[], np.ndarray | None],
        tracked_layer_provider: Callable[[], object | None],
        intensity_layer_provider: Callable[[], object | None],
        selected_label_provider: Callable[[], int],
        current_t_provider: Callable[[], int],
        db_path_provider: Callable[[], Path | None],
        protected_mask_provider: Callable[[int], np.ndarray | None],
        extend_kwargs_provider: Callable[[], dict],
        apply_swap: Callable[[object], None],
        apply_extend: Callable[[str, int, object], None],
        status_callback: Callable[[str], None] = lambda _msg: None,
        list_swap: Callable[..., list] = list_swap_candidates,
        list_extend: Callable[..., object] = list_extend_candidates,
    ) -> None:
        self.viewer = viewer
        self._tracked_data_provider = tracked_data_provider
        self._tracked_layer_provider = tracked_layer_provider
        self._intensity_layer_provider = intensity_layer_provider
        self._selected_label_provider = selected_label_provider
        self._current_t_provider = current_t_provider
        self._db_path_provider = db_path_provider
        self._protected_mask_provider = protected_mask_provider
        self._extend_kwargs_provider = extend_kwargs_provider
        self._apply_swap = apply_swap
        self._apply_extend = apply_extend
        self._status = status_callback
        self._list_swap = list_swap
        self._list_extend = list_extend

        self._panel: CandidateGalleryPanel | None = None
        # key caches for click routing, rebuilt each refresh
        self._swap_by_key: dict[int, object] = {}
        self._extend_cache: dict[str, tuple[int, dict[int, object]]] = {}

    # -- lifecycle ----------------------------------------------------------
    def _ensure_panel(self) -> CandidateGalleryPanel:
        if self._panel is not None:
            return self._panel
        panel = CandidateGalleryPanel()
        panel.candidate_activated.connect(self._on_activated)
        self._panel = panel
        return panel

    def widget(self) -> CandidateGalleryPanel:
        """The gallery widget (created on first access) to embed in the splitter.

        ``refresh`` skips populating until a cell is selected, so the workspace
        layout calls this to materialise the (possibly empty) panel eagerly and
        slot it into the splitter strip before any selection exists.
        """
        return self._ensure_panel()

    def teardown(self) -> None:
        """Forget the panel (next refresh re-creates it).

        The panel is embedded as a bare widget in the host's workspace splitter
        and is deleted when that dock is torn down; here we only drop the
        reference so a later re-activate rebuilds it.
        """
        self._panel = None
        self._swap_by_key = {}
        self._extend_cache = {}

    def clear(self) -> None:
        if self._panel is not None:
            self._panel.clear()
        self._swap_by_key = {}
        self._extend_cache = {}

    # -- refresh ------------------------------------------------------------
    def refresh(self) -> None:
        """Recompute and repaint the three candidate columns for the selection."""
        lab = int(self._selected_label_provider() or 0)
        tracked = self._tracked_data_provider()
        db_path = self._db_path_provider()
        if not lab or tracked is None or db_path is None:
            self.clear()
            return
        tracked = np.asarray(tracked)
        if tracked.ndim != 3:
            self.clear()
            return
        t = int(self._current_t_provider())
        if t < 0 or t >= tracked.shape[0]:
            self.clear()
            return

        panel = self._ensure_panel()
        intensity_2d, colormap = self._intensity_frame(t, tracked.shape[1:])
        outline = self._track_color(lab)

        self._populate_swap(panel, tracked, t, lab, intensity_2d, colormap, outline, db_path)
        self._populate_extend(panel, tracked, t, lab, intensity_2d, colormap, outline, db_path)

    def _populate_swap(
        self, panel, tracked, t, lab, intensity_2d, colormap, outline, db_path
    ) -> None:
        source_mask = tracked[t] == lab
        self._swap_by_key = {}
        if not source_mask.any():
            panel.set_column(panel.SWAP, CandidateStrip())
            return
        try:
            candidates = self._list_swap(
                db_path=db_path,
                frame=t,
                source_mask=source_mask,
                frame_shape=tuple(tracked.shape[1:]),
                protected_mask=self._protected_mask_provider(t),
            )
        except Exception:
            logger.exception("swap candidate enumeration failed")
            candidates = []
        specs = [
            CandidateSpec(key=int(c.node_id), mask=c.mask_2d, caption=f"{int(c.area)} px")
            for c in candidates
        ]
        self._swap_by_key = {int(c.node_id): c for c in candidates}
        panel.set_column(
            panel.SWAP,
            build_candidate_strip(
                intensity_2d, specs, colormap=colormap, outline_color=outline
            ),
        )

    def _populate_extend(
        self, panel, tracked, t, lab, intensity_2d, colormap, outline, db_path
    ) -> None:
        extend_kwargs = self._extend_kwargs_provider() or {}
        self._extend_cache = {}
        for which, direction in _EXTEND_DIRECTION.items():
            try:
                result = self._list_extend(
                    source_id=lab,
                    source_frame=t,
                    direction=direction,
                    tracked_labels=tracked,
                    db_path=db_path,
                    **extend_kwargs,
                )
            except Exception:
                logger.exception("extend candidate enumeration failed (%s)", direction)
                panel.set_column(which, CandidateStrip())
                continue
            assignments = tuple(getattr(result, "assignments", ()) or ())
            target_frame = int(getattr(result, "target_frame", -1))
            specs = [
                CandidateSpec(
                    key=int(a.candidate_label),
                    mask=a.mask_2d,
                    caption=self._extend_caption(a),
                )
                for a in assignments
            ]
            self._extend_cache[which] = (
                target_frame,
                {int(a.candidate_label): a for a in assignments},
            )
            # Extend candidates live on the *adjacent* frame, so crop their masks
            # over that frame's intensity rather than the current one.
            adj_intensity, adj_cmap = (
                self._intensity_frame(target_frame, tracked.shape[1:])
                if 0 <= target_frame < tracked.shape[0]
                else (intensity_2d, colormap)
            )
            panel.set_column(
                which,
                build_candidate_strip(
                    adj_intensity, specs, colormap=adj_cmap, outline_color=outline
                ),
            )

    @staticmethod
    def _extend_caption(assignment) -> str:
        iou = float(getattr(assignment, "centroid_corrected_iou", 0.0))
        dist = float(getattr(assignment, "centroid_distance", 0.0))
        return f"iou {iou:.2f} · {dist:.0f}px"

    # -- click routing ------------------------------------------------------
    def _on_activated(self, which: str, key: int) -> None:
        try:
            if which == CandidateGalleryPanel.SWAP:
                candidate = self._swap_by_key.get(int(key))
                if candidate is not None:
                    self._apply_swap(candidate)
            else:
                target_frame, mapping = self._extend_cache.get(which, (-1, {}))
                assignment = mapping.get(int(key))
                if assignment is not None:
                    self._apply_extend(which, target_frame, assignment)
        except Exception:
            logger.exception("applying candidate failed (%s, %s)", which, key)
            return
        self.refresh()

    # -- intensity / colour helpers ----------------------------------------
    def _intensity_frame(self, t: int, frame_shape) -> tuple[np.ndarray, object]:
        layer = self._intensity_layer_provider()
        if layer is None:
            return np.zeros(frame_shape, dtype=np.float32), None
        data = np.asarray(layer.data)
        plane = data[t] if data.ndim >= 3 else data
        while plane.ndim > 2:
            plane = plane[0]
        return plane, self._adapt_colormap(layer)

    @staticmethod
    def _adapt_colormap(layer):
        cmap = getattr(layer, "colormap", None)
        if cmap is None or not hasattr(cmap, "map"):
            return None

        def _map(values: np.ndarray) -> np.ndarray:
            flat = np.asarray(values, dtype=float).ravel()
            mapped = np.asarray(cmap.map(flat), dtype=float)
            return mapped.reshape(values.shape + (mapped.shape[-1],))

        return _map

    def _track_color(self, cell_id: int):
        layer = self._tracked_layer_provider()
        color_dict = getattr(getattr(layer, "colormap", None), "color_dict", None)
        try:
            raw = color_dict.get(int(cell_id)) if color_dict is not None else None
        except Exception:
            raw = None
        if raw is None or isinstance(raw, str):
            return _NEUTRAL_OUTLINE
        rgba = np.asarray(raw, dtype=float).ravel()
        if rgba.size < 3:
            return _NEUTRAL_OUTLINE
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]))


__all__ = ["CandidateGalleryController"]
