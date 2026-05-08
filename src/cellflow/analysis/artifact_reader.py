from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np


def _read_dataset(dataset: h5py.Dataset) -> np.ndarray:
    if h5py.check_string_dtype(dataset.dtype) is not None:
        return dataset.asstr()[:]
    return dataset[:]


def _read_table(group: h5py.Group) -> dict[str, np.ndarray]:
    return {name: _read_dataset(dataset) for name, dataset in group.items()}


@dataclass(frozen=True)
class PositionArtifactData:
    cells: dict[str, np.ndarray]
    edges: dict[str, np.ndarray]
    t1_events: dict[str, np.ndarray]
    cell_tracked_labels_path: str
    nucleus_tracked_labels_path: str
    _edge_coord_y: np.ndarray = field(repr=False)
    _edge_coord_x: np.ndarray = field(repr=False)

    @property
    def coord_y(self) -> np.ndarray:
        return self._edge_coord_y

    @property
    def coord_x(self) -> np.ndarray:
        return self._edge_coord_x

    def edge_lines(self) -> list[np.ndarray]:
        lines: list[np.ndarray] = []
        offsets = self.edges.get("coord_offset", np.asarray([], dtype=np.int64))
        counts = self.edges.get("coord_count", np.asarray([], dtype=np.int64))
        frames = self.edges.get("frame", np.asarray([], dtype=np.int64))
        for frame, offset, count in zip(frames, offsets, counts):
            start = int(offset)
            stop = start + int(count)
            y = self._edge_coord_y[start:stop]
            x = self._edge_coord_x[start:stop]
            if len(y) == 0:
                lines.append(np.empty((0, 3), dtype=float))
                continue
            frame_col = np.full(len(y), float(frame), dtype=float)
            lines.append(np.column_stack([frame_col, y.astype(float, copy=False), x.astype(float, copy=False)]))
        return lines

    def centroid_points(self) -> np.ndarray:
        frames = self.cells.get("frame", np.asarray([], dtype=np.int64))
        ys = self.cells.get("centroid_y", np.asarray([], dtype=float))
        xs = self.cells.get("centroid_x", np.asarray([], dtype=float))
        if len(frames) == 0:
            return np.empty((0, 3), dtype=float)
        return np.column_stack([frames.astype(float, copy=False), ys.astype(float, copy=False), xs.astype(float, copy=False)])


def read_position_artifact(path: str | Path) -> PositionArtifactData:
    path = Path(path)
    with h5py.File(path, "r") as h5:
        provenance = h5["provenance"].attrs
        cell_tracked_labels_path = str(provenance["cell_tracked_labels_path"])
        nucleus_tracked_labels_path = str(provenance["nucleus_tracked_labels_path"])
        cells = _read_table(h5["cells/table"])
        edges = _read_table(h5["edges/table"])
        t1_events = _read_table(h5["t1_events/table"])
        edge_coord_y = h5["edges/coordinates/y"][:]
        edge_coord_x = h5["edges/coordinates/x"][:]
    return PositionArtifactData(
        cells=cells,
        edges=edges,
        t1_events=t1_events,
        cell_tracked_labels_path=cell_tracked_labels_path,
        nucleus_tracked_labels_path=nucleus_tracked_labels_path,
        _edge_coord_y=edge_coord_y,
        _edge_coord_x=edge_coord_x,
    )
