"""Shared click-to-load: a picked plot point's identity -> input labels in the
viewer. Used by the Shape and Track-dynamics plugins; the underscore keeps it out
of plugin auto-discovery (see ``plugins/__init__.py``)."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import tifffile

from cellflow.napari.aggregate_quantification.plot_panel import LoadTarget


class ClickToLoad:
    """Resolves picked points to input labels and loads them, replacing the
    previously loaded layer each time (one position shown at a time)."""

    def __init__(self, viewer) -> None:
        self._viewer = viewer
        self._layer = None

    def resolver(
        self, records: list[dict], label_field: str
    ) -> Callable[[dict], LoadTarget | None]:
        """Closure: identity dict -> LoadTarget for *label_field*'s TIFF, or None
        when the position is unknown or has no labels of that scope."""
        by_id = {str(r.get("id")): r for r in records}

        def resolve(identity: dict) -> LoadTarget | None:
            record = by_id.get(str(identity.get("position_id")))
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
        """Replace the loaded layer with *target*'s labels, jump to its frame, and
        select + center its cell."""
        labels = np.asarray(tifffile.imread(target.path))
        if self._layer is not None and self._layer in list(self._viewer.layers):
            self._viewer.layers.remove(self._layer)
        self._layer = self._viewer.add_labels(labels, name=f"input · {target.path.parent.name}")

        if target.frame is not None and labels.ndim >= 3:
            self._viewer.dims.set_current_step(0, int(target.frame))
        if target.cell_id is not None:
            self._layer.selected_label = int(target.cell_id)
            self._layer.show_selected_label = True
            self._center_on_cell(labels, target.frame, int(target.cell_id))

    def _center_on_cell(self, labels: np.ndarray, frame: int | None, cell_id: int) -> None:
        plane = labels[frame] if (frame is not None and labels.ndim >= 3) else labels
        ys, xs = np.nonzero(plane == cell_id)
        if ys.size:
            try:
                self._viewer.camera.center = (float(ys.mean()), float(xs.mean()))
            except Exception:  # camera centering is best-effort across napari versions
                pass
