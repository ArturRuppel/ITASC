"""Shared click-to-load: a picked plot point's identity -> input labels in the
viewer. Used by the Shape and Track-dynamics plugins; the underscore keeps it out
of plugin auto-discovery (see ``plugins/__init__.py``)."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import tifffile

from cellflow.napari._spotlight import spotlight_rgba
from cellflow.napari.aggregate_quantification.plot_panel import LoadTarget

#: Overlay layer painting the picked cell's spotlight + yellow border, mirroring
#: the correction studio's selection highlight.
_SPOTLIGHT_LAYER = "picked cell"


class ClickToLoad:
    """Resolves picked points to input labels and loads them, replacing the
    previously loaded layer each time (one position shown at a time).

    The whole labels movie is shown; the picked cell is highlighted in place with
    the correction studio's spotlight (everything else dimmed) plus a yellow
    border, rather than hiding the rest of the segmentation."""

    def __init__(self, viewer) -> None:
        self._viewer = viewer
        self._layer = None
        self._spotlight = None
        self._labels = None
        self._cell_id = None
        self._dims_connected = False
        #: True while :meth:`load` is mutating layers. Adding/removing layers and
        #: changing ``dims.ndim`` re-emits ``current_step``; reacting to it then
        #: (re)adds the spotlight mid-insert and napari reorders layers that are
        #: only half-registered → ``KeyError`` in ``_reorder_layers``. The guard
        #: makes ``_on_dims_change`` a no-op until the load settles.
        self._loading = False

    def resolver(
        self, records: list[dict], label_field: str
    ) -> Callable[[dict], LoadTarget | None]:
        """Closure: identity dict -> LoadTarget for *label_field*'s TIFF, or None
        when the position is unknown or has no labels of that scope.

        ``position_id`` (the catalogue ``id``) is reused across experiments —
        ``pos00`` exists on every date — so it cannot key records on its own: a
        plain ``{id: record}`` dict silently keeps only the last same-named
        position, so a picked point would resolve to the wrong experiment's movie.
        The pooled identity carries ``date`` and ``(date, position_id)`` is unique,
        so key on the pair; fall back to ``id`` alone only for date-less identities
        (older snapshots), where no better key exists."""
        by_key = {(str(r.get("date")), str(r.get("id"))): r for r in records}
        by_id = {str(r.get("id")): r for r in records}

        def resolve(identity: dict) -> LoadTarget | None:
            position_id = str(identity.get("position_id"))
            date = identity.get("date")
            record = (
                by_key.get((str(date), position_id)) if date is not None else None
            )
            if record is None:
                record = by_id.get(position_id)
            if record is None:
                return None
            path = record.get(label_field)
            if not path:
                return None
            frame = identity.get("frame")
            if frame is None:
                frame = identity.get("frame_start")
            cell_id = identity.get("cell_id")
            return LoadTarget(
                path=Path(path),
                kind="labels",
                frame=None if frame is None else int(frame),
                cell_id=None if cell_id is None else int(cell_id),
                identity=identity,
            )

        return resolve

    def load(self, target: LoadTarget) -> None:
        """Replace the loaded layer with *target*'s full labels movie, jump to its
        frame, and spotlight + outline its cell over the rest of the labels."""
        labels = np.asarray(tifffile.imread(target.path))
        # Mutating layers below re-emits dims events; ignore them until the load
        # settles so the spotlight is (re)built exactly once, after the labels
        # layer is fully registered.
        self._loading = True
        try:
            self._remove(self._spotlight)
            self._spotlight = None
            self._remove(self._layer)
            self._labels = labels
            self._cell_id = None if target.cell_id is None else int(target.cell_id)
            self._layer = self._viewer.add_labels(
                labels, name=f"input · {target.path.parent.name}"
            )

            if target.frame is not None and labels.ndim >= 3:
                self._viewer.dims.set_current_step(0, int(target.frame))
            if self._cell_id is not None:
                self._render_spotlight(target.frame)
                self._center_on_cell(labels, target.frame, self._cell_id)
                self._ensure_dims_connection()
        finally:
            self._loading = False

    def _remove(self, layer) -> None:
        if layer is not None and layer in list(self._viewer.layers):
            self._viewer.layers.remove(layer)

    def _frame_plane(self, frame: int | None) -> np.ndarray:
        """The 2D label plane the spotlight is drawn on for *frame*."""
        labels = self._labels
        if labels.ndim >= 3:
            idx = 0 if frame is None else max(0, min(int(frame), labels.shape[0] - 1))
            return labels[idx]
        return labels

    def _render_spotlight(self, frame: int | None) -> None:
        """Paint (or refresh) the spotlight overlay for the picked cell at *frame*.

        Reuses the correction studio's renderer: the cell stays bright, the rest
        of the frame dims, and a yellow border outlines it. The overlay hides
        itself on frames where the cell is absent."""
        mask = self._frame_plane(frame) == self._cell_id
        data = spotlight_rgba(mask, dim=True, border=True)
        if self._spotlight is None or self._spotlight not in list(self._viewer.layers):
            self._spotlight = self._viewer.add_image(
                data, name=_SPOTLIGHT_LAYER, rgb=True, blending="translucent"
            )
        else:
            self._spotlight.data = data
        self._spotlight.visible = bool(mask.any())

    def _ensure_dims_connection(self) -> None:
        """Keep the spotlight on the picked cell as the user scrubs frames."""
        if self._dims_connected:
            return
        try:
            self._viewer.dims.events.current_step.connect(self._on_dims_change)
            self._dims_connected = True
        except (AttributeError, TypeError):  # fake/headless viewers expose no events
            pass

    def _on_dims_change(self, event=None) -> None:
        if self._loading:
            return
        if self._cell_id is None or self._labels is None or self._labels.ndim < 3:
            return
        try:
            frame = int(self._viewer.dims.current_step[0])
        except (AttributeError, IndexError, TypeError):
            return
        self._render_spotlight(frame)

    def _center_on_cell(self, labels: np.ndarray, frame: int | None, cell_id: int) -> None:
        plane = labels[frame] if (frame is not None and labels.ndim >= 3) else labels
        ys, xs = np.nonzero(plane == cell_id)
        if ys.size:
            try:
                self._viewer.camera.center = (float(ys.mean()), float(xs.mean()))
            except Exception:  # camera centering is best-effort across napari versions
                pass
