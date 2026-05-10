from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import h5py
import numpy as np
import tifffile
from skimage.measure import regionprops


NULL_INT = -1


@dataclass(frozen=True)
class EdgeRecord:
    frame: int
    pair: tuple[int, int]
    kind: str
    edge_label: str
    length: float
    midpoint_y: float
    midpoint_x: float
    coordinates: np.ndarray


@dataclass(frozen=True)
class T1Record:
    t1_event_id: int
    frame: int
    edge_id: int
    losing_pair: tuple[int, int]
    gaining_pair: tuple[int, int]
    location_y: float
    location_x: float

    @property
    def losing_cell_a(self) -> int:
        return self.losing_pair[0]

    @property
    def losing_cell_b(self) -> int:
        return self.losing_pair[1]

    @property
    def gaining_cell_a(self) -> int:
        return self.gaining_pair[0]

    @property
    def gaining_cell_b(self) -> int:
        return self.gaining_pair[1]


def build_position_analysis_artifact(
    position_path: str | Path,
    output_path: str | Path,
    *,
    cell_tracked_labels_path: str | Path | None = None,
    nucleus_tracked_labels_path: str | Path | None = None,
    edge_extraction_params: dict | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Build the canonical per-position analysis HDF5 artifact."""
    total = 6
    position_path = Path(position_path)
    output_path = Path(output_path)
    params = dict(edge_extraction_params or {})

    cell_labels_path = Path(cell_tracked_labels_path) if cell_tracked_labels_path else (
        position_path / "cell" / "tracked_labels.tif"
    )
    nucleus_labels_path = (
        Path(nucleus_tracked_labels_path)
        if nucleus_tracked_labels_path
        else position_path / "nucleus" / "tracked_labels.tif"
    )
    cell_stack = _read_label_stack(cell_labels_path)
    nucleus_stack = _read_label_stack(nucleus_labels_path)
    _report_progress(progress_cb, 1, total, "read labels")
    _validate_cell_nucleus_identity(cell_stack, nucleus_stack)
    _report_progress(progress_cb, 2, total, "validate IDs")

    cell_columns = _extract_cell_columns(cell_stack)
    _report_progress(progress_cb, 3, total, "extract cells")
    edge_records = _extract_edges(cell_stack)
    _report_progress(progress_cb, 4, total, "extract edges")
    assignments, t1_events = _assign_ids_to_records(edge_records)
    _report_progress(progress_cb, 5, total, "assign edge IDs/T1")
    edge_columns, coord_y, coord_x = _extract_edge_columns(edge_records, assignments, t1_events)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5:
        provenance = h5.create_group("provenance")
        provenance.attrs["source_position_path"] = str(position_path)
        provenance.attrs["cell_tracked_labels_path"] = str(cell_labels_path)
        provenance.attrs["nucleus_tracked_labels_path"] = str(nucleus_labels_path)
        provenance.attrs["edge_extraction_params_json"] = json.dumps(
            params, sort_keys=True, separators=(",", ":")
        )
        provenance.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
        provenance.attrs["cellflow_version"] = _cellflow_version()

        _write_column_group(h5.create_group("cells/table", track_order=True), cell_columns)
        h5.create_group("cells/measurements")

        edges_group = h5.create_group("edges/table", track_order=True)
        _write_column_group(edges_group, edge_columns)
        edges_group["t1_event_id"].attrs["null_sentinel"] = NULL_INT

        coords = h5.create_group("edges/coordinates", track_order=True)
        coords.create_dataset("y", data=coord_y)
        coords.create_dataset("x", data=coord_x)
        h5.create_group("edges/measurements")

        _write_t1_table(h5.create_group("t1_events/table", track_order=True), t1_events)
    _report_progress(progress_cb, 6, total, "write HDF5")

    return output_path


def assign_persistent_edge_ids(
    frame_edges: list[Iterable[tuple[int, int]]],
) -> tuple[list[dict[tuple[int, int], int]], list[T1Record]]:
    """Assign deterministic edge IDs, linking losing/gaining pairs through T1s."""
    assignments: list[dict[tuple[int, int], int]] = []
    pair_to_id: dict[tuple[int, int], int] = {}
    events: list[T1Record] = []
    next_edge_id = 1

    normalized_frames = [
        {tuple(sorted((int(a), int(b)))) for a, b in pairs}
        for pairs in frame_edges
    ]

    for frame_idx, pairs in enumerate(normalized_frames):
        if frame_idx > 0:
            prev = normalized_frames[frame_idx - 1]
            for losing, gaining in _detect_t1_pairs(prev, pairs):
                edge_id = pair_to_id.get(losing)
                if edge_id is None:
                    edge_id = next_edge_id
                    next_edge_id += 1
                    pair_to_id[losing] = edge_id
                pair_to_id[gaining] = edge_id
                events.append(
                    T1Record(
                        t1_event_id=len(events) + 1,
                        frame=frame_idx - 1,
                        edge_id=edge_id,
                        losing_pair=losing,
                        gaining_pair=gaining,
                        location_y=np.nan,
                        location_x=np.nan,
                    )
                )

        frame_assignment: dict[tuple[int, int], int] = {}
        for pair in sorted(pairs):
            if pair not in pair_to_id:
                pair_to_id[pair] = next_edge_id
                next_edge_id += 1
            frame_assignment[pair] = pair_to_id[pair]
        assignments.append(frame_assignment)

    return assignments, events


def _read_label_stack(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.asarray(tifffile.imread(path))
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected a 2-D or 3-D tracked label TIFF at {path}, got shape {arr.shape}")
    return arr.astype(np.int64, copy=False)


def _validate_cell_nucleus_identity(cell_stack: np.ndarray, nucleus_stack: np.ndarray) -> None:
    if cell_stack.shape != nucleus_stack.shape:
        raise ValueError(
            "cell_id == nucleus_id invariant requires matching cell and nucleus stack shapes"
        )
    for frame_idx in range(cell_stack.shape[0]):
        cell_ids = set(np.unique(cell_stack[frame_idx]).astype(int))
        nucleus_ids = set(np.unique(nucleus_stack[frame_idx]).astype(int))
        cell_ids.discard(0)
        nucleus_ids.discard(0)
        if cell_ids != nucleus_ids:
            cell_only = sorted(int(label) for label in cell_ids - nucleus_ids)
            nucleus_only = sorted(int(label) for label in nucleus_ids - cell_ids)
            raise ValueError(
                "cell_id == nucleus_id invariant failed for frame "
                f"{frame_idx}: cell labels present only in cell stack: {cell_only}; "
                f"nucleus labels present only in nucleus stack: {nucleus_only}"
            )


def _extract_cell_columns(label_stack: np.ndarray) -> dict[str, np.ndarray]:
    rows: list[dict[str, int | float | str]] = []
    for frame_idx, frame in enumerate(label_stack):
        for prop in sorted(regionprops(frame), key=lambda item: item.label):
            min_y, min_x, max_y, max_x = prop.bbox
            rows.append(
                {
                    "frame": frame_idx,
                    "cell_id": int(prop.label),
                    "class_label": "",
                    "area": float(prop.area),
                    "centroid_y": float(prop.centroid[0]),
                    "centroid_x": float(prop.centroid[1]),
                    "perimeter": float(prop.perimeter),
                    "bbox_min_y": int(min_y),
                    "bbox_min_x": int(min_x),
                    "bbox_max_y": int(max_y),
                    "bbox_max_x": int(max_x),
                }
            )
    return _columns_from_rows(
        rows,
        [
            "frame",
            "cell_id",
            "class_label",
            "area",
            "centroid_y",
            "centroid_x",
            "perimeter",
            "bbox_min_y",
            "bbox_min_x",
            "bbox_max_y",
            "bbox_max_x",
        ],
    )


def _extract_edges(label_stack: np.ndarray) -> list[EdgeRecord]:
    records: list[EdgeRecord] = []
    for frame_idx, frame in enumerate(label_stack):
        records.extend(_extract_frame_cell_edges(frame, frame_idx))
        records.extend(_extract_frame_border_edges(frame, frame_idx))
    return sorted(records, key=lambda row: (row.frame, row.kind != "cell_cell", row.pair))


def _extract_frame_cell_edges(frame: np.ndarray, frame_idx: int) -> list[EdgeRecord]:
    points_by_pair: dict[tuple[int, int], list[tuple[float, float]]] = {}

    left = frame[:, :-1]
    right = frame[:, 1:]
    ys, xs = np.where((left != right) & (left > 0) & (right > 0))
    for y, x in zip(ys, xs):
        pair = tuple(sorted((int(left[y, x]), int(right[y, x]))))
        points_by_pair.setdefault(pair, []).append((float(y), float(x) + 0.5))

    top = frame[:-1, :]
    bottom = frame[1:, :]
    ys, xs = np.where((top != bottom) & (top > 0) & (bottom > 0))
    for y, x in zip(ys, xs):
        pair = tuple(sorted((int(top[y, x]), int(bottom[y, x]))))
        points_by_pair.setdefault(pair, []).append((float(y) + 0.5, float(x)))

    rows = []
    for pair in sorted(points_by_pair):
        for segment in _coordinate_segments(np.asarray(points_by_pair[pair], dtype=float)):
            coords = _order_coordinates(segment)
            rows.append(_edge_record(frame_idx, pair, "cell_cell", "", coords))
    return rows


def _extract_frame_border_edges(frame: np.ndarray, frame_idx: int) -> list[EdgeRecord]:
    points_by_cell: dict[int, list[tuple[float, float]]] = {}
    h, w = frame.shape

    for y in range(h):
        for x in range(w):
            cell_id = int(frame[y, x])
            if cell_id <= 0:
                continue
            if y == 0:
                points_by_cell.setdefault(cell_id, []).append((float(y) - 0.5, float(x)))
            if y == h - 1:
                points_by_cell.setdefault(cell_id, []).append((float(y) + 0.5, float(x)))
            if x == 0:
                points_by_cell.setdefault(cell_id, []).append((float(y), float(x) - 0.5))
            if x == w - 1:
                points_by_cell.setdefault(cell_id, []).append((float(y), float(x) + 0.5))

            if y > 0 and frame[y - 1, x] == 0:
                points_by_cell.setdefault(cell_id, []).append((float(y) - 0.5, float(x)))
            if y < h - 1 and frame[y + 1, x] == 0:
                points_by_cell.setdefault(cell_id, []).append((float(y) + 0.5, float(x)))
            if x > 0 and frame[y, x - 1] == 0:
                points_by_cell.setdefault(cell_id, []).append((float(y), float(x) - 0.5))
            if x < w - 1 and frame[y, x + 1] == 0:
                points_by_cell.setdefault(cell_id, []).append((float(y), float(x) + 0.5))

    rows = []
    for cell_id in sorted(points_by_cell):
        for segment in _coordinate_segments(np.asarray(points_by_cell[cell_id], dtype=float)):
            coords = _order_coordinates(segment)
            rows.append(_edge_record(frame_idx, (cell_id, 0), "border", "border", coords))
    return rows


def _edge_record(
    frame_idx: int,
    pair: tuple[int, int],
    kind: str,
    edge_label: str,
    coords: np.ndarray,
) -> EdgeRecord:
    midpoint = coords[len(coords) // 2] if len(coords) else np.array([np.nan, np.nan])
    return EdgeRecord(
        frame=frame_idx,
        pair=pair,
        kind=kind,
        edge_label=edge_label,
        length=_path_length(coords),
        midpoint_y=float(midpoint[0]),
        midpoint_x=float(midpoint[1]),
        coordinates=coords,
    )


def _assign_ids_to_records(
    records: list[EdgeRecord],
) -> tuple[list[dict[tuple[int, int], int]], list[T1Record]]:
    max_frame = max((record.frame for record in records), default=-1)
    frame_cell_edges: list[set[tuple[int, int]]] = []
    for frame_idx in range(max_frame + 1):
        frame_cell_edges.append(
            {
                record.pair
                for record in records
                if record.frame == frame_idx and record.kind == "cell_cell"
            }
        )
    assignments, events = assign_persistent_edge_ids(frame_cell_edges)
    border_next_id = (
        max((edge_id for frame in assignments for edge_id in frame.values()), default=0) + 1
    )
    border_ids: dict[tuple[int, int], int] = {}
    for record in records:
        if record.kind != "border" or record.pair in border_ids:
            continue
        border_ids[record.pair] = border_next_id
        border_next_id += 1
    merged = []
    for frame_idx, frame_assignment in enumerate(assignments):
        full = dict(frame_assignment)
        for pair, edge_id in border_ids.items():
            if any(record.frame == frame_idx and record.pair == pair for record in records):
                full[pair] = edge_id
        merged.append(full)

    t1_by_key = {(event.frame, event.losing_pair): event for event in events}
    records_by_key = {(record.frame, record.pair): record for record in records}
    resolved_events = []
    for event in events:
        rec = records_by_key.get((event.frame, event.losing_pair))
        if rec is None:
            resolved_events.append(event)
            continue
        resolved_events.append(
            T1Record(
                t1_event_id=event.t1_event_id,
                frame=event.frame,
                edge_id=event.edge_id,
                losing_pair=event.losing_pair,
                gaining_pair=event.gaining_pair,
                location_y=rec.midpoint_y,
                location_x=rec.midpoint_x,
            )
        )
    return merged, resolved_events


def _extract_edge_columns(
    records: list[EdgeRecord],
    assignments: list[dict[tuple[int, int], int]],
    t1_events: list[T1Record],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    rows = []
    coord_y: list[float] = []
    coord_x: list[float] = []
    t1_by_key = {(event.frame, event.losing_pair): event for event in t1_events}

    for record in records:
        offset = len(coord_y)
        coords = record.coordinates
        coord_y.extend(coords[:, 0].astype(float).tolist())
        coord_x.extend(coords[:, 1].astype(float).tolist())
        event = t1_by_key.get((record.frame, record.pair))
        cell_a, cell_b = record.pair
        rows.append(
            {
                "frame": record.frame,
                "edge_id": assignments[record.frame][record.pair],
                "cell_a": int(cell_a),
                "cell_b": int(cell_b),
                "kind": record.kind,
                "edge_label": record.edge_label,
                "is_t1_frame": event is not None,
                "t1_event_id": event.t1_event_id if event is not None else NULL_INT,
                "length": record.length,
                "midpoint_y": record.midpoint_y,
                "midpoint_x": record.midpoint_x,
                "coord_offset": offset,
                "coord_count": len(coords),
            }
        )
    return (
        _columns_from_rows(
            rows,
            [
                "frame",
                "edge_id",
                "cell_a",
                "cell_b",
                "kind",
                "edge_label",
                "is_t1_frame",
                "t1_event_id",
                "length",
                "midpoint_y",
                "midpoint_x",
                "coord_offset",
                "coord_count",
            ],
        ),
        np.asarray(coord_y, dtype=float),
        np.asarray(coord_x, dtype=float),
    )


def _detect_t1_pairs(
    prev_edges: set[tuple[int, int]],
    next_edges: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    removed = sorted(prev_edges - next_edges)
    added = sorted(next_edges - prev_edges)
    events = []
    used_removed = set()
    used_added = set()
    for lost in removed:
        for gained in added:
            if lost in used_removed or gained in used_added:
                continue
            if _is_valid_t1(lost, gained, prev_edges, next_edges):
                events.append((lost, gained))
                used_removed.add(lost)
                used_added.add(gained)
    return events


def _is_valid_t1(
    lost: tuple[int, int],
    gained: tuple[int, int],
    prev_edges: set[tuple[int, int]],
    next_edges: set[tuple[int, int]],
) -> bool:
    all_cells = set(lost) | set(gained)
    if len(all_cells) != 4:
        return False
    lost_cells = tuple(sorted(lost))
    gained_cells = tuple(sorted(gained))
    connectors = [
        tuple(sorted((lost_cells[0], gained_cells[0]))),
        tuple(sorted((lost_cells[0], gained_cells[1]))),
        tuple(sorted((lost_cells[1], gained_cells[0]))),
        tuple(sorted((lost_cells[1], gained_cells[1]))),
    ]
    return all(edge in prev_edges and edge in next_edges for edge in connectors)


def _order_coordinates(coords: np.ndarray) -> np.ndarray:
    if len(coords) <= 2:
        return coords[np.lexsort((coords[:, 1], coords[:, 0]))]

    coords = np.unique(np.asarray(coords, dtype=float), axis=0)
    neighbors = _coordinate_neighbors(coords)
    endpoints = [idx for idx, adjacent in enumerate(neighbors) if len(adjacent) == 1]
    start = min(endpoints or range(len(coords)), key=lambda idx: (coords[idx, 0], coords[idx, 1]))

    ordered_indices = [start]
    visited = {start}
    prev_idx: int | None = None
    current_idx = start
    while len(visited) < len(coords):
        candidates = [idx for idx in neighbors[current_idx] if idx not in visited]
        if candidates:
            next_idx = min(
                candidates,
                key=lambda idx: (
                    _turn_cost(coords[prev_idx], coords[current_idx], coords[idx])
                    if prev_idx is not None
                    else 0.0,
                    coords[idx, 0],
                    coords[idx, 1],
                ),
            )
        else:
            remaining = [idx for idx in range(len(coords)) if idx not in visited]
            next_idx = min(
                remaining,
                key=lambda idx: (
                    float(np.linalg.norm(coords[current_idx] - coords[idx])),
                    coords[idx, 0],
                    coords[idx, 1],
                ),
            )
        ordered_indices.append(next_idx)
        visited.add(next_idx)
        prev_idx = current_idx
        current_idx = next_idx
    return coords[np.asarray(ordered_indices, dtype=np.intp)]


def _coordinate_neighbors(coords: np.ndarray) -> list[list[int]]:
    scaled = np.rint(coords * 2.0).astype(np.int64)
    point_to_idx = {tuple(point): idx for idx, point in enumerate(scaled)}
    neighbor_offsets = [
        (dy, dx)
        for dy in range(-2, 3)
        for dx in range(-2, 3)
        if (dy or dx) and (dy * dy + dx * dx) <= 4
    ]
    neighbors: list[list[int]] = []
    for y, x in scaled:
        adjacent = []
        for dy, dx in neighbor_offsets:
            idx = point_to_idx.get((int(y + dy), int(x + dx)))
            if idx is not None:
                adjacent.append(idx)
        neighbors.append(sorted(adjacent, key=lambda idx: (coords[idx, 0], coords[idx, 1])))
    return neighbors


def _turn_cost(prev: np.ndarray, current: np.ndarray, candidate: np.ndarray) -> float:
    incoming = current - prev
    outgoing = candidate - current
    incoming_norm = float(np.linalg.norm(incoming))
    outgoing_norm = float(np.linalg.norm(outgoing))
    if incoming_norm == 0.0 or outgoing_norm == 0.0:
        return 0.0
    return float(1.0 - np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm))


def _coordinate_segments(coords: np.ndarray) -> list[np.ndarray]:
    if len(coords) <= 1:
        return [coords]

    coords = np.unique(np.asarray(coords, dtype=float), axis=0)
    neighbors = _coordinate_neighbors(coords)
    unused_edges = {
        tuple(sorted((idx, neighbor_idx)))
        for idx, adjacent in enumerate(neighbors)
        for neighbor_idx in adjacent
        if idx != neighbor_idx
    }
    segments: list[np.ndarray] = []

    while unused_edges:
        start_idx = _next_trail_start(coords, neighbors, unused_edges)
        next_idx = _next_unused_neighbor(coords, start_idx, None, neighbors, unused_edges)
        if next_idx is None:
            unused_edges = {edge for edge in unused_edges if start_idx not in edge}
            continue

        path = [start_idx]
        prev_idx = start_idx
        current_idx = next_idx
        unused_edges.remove(tuple(sorted((prev_idx, current_idx))))
        path.append(current_idx)

        while len(neighbors[current_idx]) == 2:
            next_idx = _next_unused_neighbor(
                coords, current_idx, prev_idx, neighbors, unused_edges
            )
            if next_idx is None:
                break
            unused_edges.remove(tuple(sorted((current_idx, next_idx))))
            path.append(next_idx)
            prev_idx, current_idx = current_idx, next_idx

        segments.append(coords[np.asarray(path, dtype=np.intp)])

    return sorted(segments, key=lambda segment: (segment[0, 0], segment[0, 1], len(segment)))


def _next_trail_start(
    coords: np.ndarray,
    neighbors: list[list[int]],
    unused_edges: set[tuple[int, int]],
) -> int:
    incident = {idx for edge in unused_edges for idx in edge}
    preferred = [idx for idx in incident if len(neighbors[idx]) != 2]
    return min(preferred or incident, key=lambda idx: (coords[idx, 0], coords[idx, 1]))


def _next_unused_neighbor(
    coords: np.ndarray,
    current_idx: int,
    prev_idx: int | None,
    neighbors: list[list[int]],
    unused_edges: set[tuple[int, int]],
) -> int | None:
    candidates = [
        idx
        for idx in neighbors[current_idx]
        if tuple(sorted((current_idx, idx))) in unused_edges
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda idx: (
            _turn_cost(coords[prev_idx], coords[current_idx], coords[idx])
            if prev_idx is not None
            else 0.0,
            coords[idx, 0],
            coords[idx, 1],
        ),
    )


def _path_length(coords: np.ndarray) -> float:
    if len(coords) < 2:
        return 0.0
    diffs = np.diff(coords, axis=0)
    return float(np.sqrt(np.sum(diffs * diffs, axis=1)).sum())


def _columns_from_rows(rows: list[dict], names: list[str]) -> dict[str, np.ndarray]:
    columns = {}
    for name in names:
        values = [row[name] for row in rows]
        if name in {"kind", "edge_label", "class_label"}:
            columns[name] = np.asarray(values, dtype=object)
        elif name == "is_t1_frame":
            columns[name] = np.asarray(values, dtype=bool)
        elif name in {
            "area",
            "centroid_y",
            "centroid_x",
            "perimeter",
            "length",
            "midpoint_y",
            "midpoint_x",
            "location_y",
            "location_x",
        }:
            columns[name] = np.asarray(values, dtype=float)
        else:
            columns[name] = np.asarray(values, dtype=np.int64)
    return columns


def _write_column_group(group: h5py.Group, columns: dict[str, np.ndarray]) -> None:
    string_dtype = h5py.string_dtype(encoding="utf-8")
    for name, values in columns.items():
        if values.dtype == object:
            group.create_dataset(name, data=values, dtype=string_dtype)
        else:
            group.create_dataset(name, data=values)


def _write_t1_table(group: h5py.Group, events: list[T1Record]) -> None:
    rows = [
        {
            "t1_event_id": event.t1_event_id,
            "frame": event.frame,
            "edge_id": event.edge_id,
            "losing_cell_a": event.losing_cell_a,
            "losing_cell_b": event.losing_cell_b,
            "gaining_cell_a": event.gaining_cell_a,
            "gaining_cell_b": event.gaining_cell_b,
            "location_y": event.location_y,
            "location_x": event.location_x,
        }
        for event in events
    ]
    columns = _columns_from_rows(
        rows,
        [
            "t1_event_id",
            "frame",
            "edge_id",
            "losing_cell_a",
            "losing_cell_b",
            "gaining_cell_a",
            "gaining_cell_b",
            "location_y",
            "location_x",
        ],
    )
    _write_column_group(group, columns)


def _report_progress(
    progress_cb: Callable[[int, int, str], None] | None,
    done: int,
    total: int,
    message: str,
) -> None:
    if progress_cb is not None:
        progress_cb(done, total, message)


def _cellflow_version() -> str:
    try:
        from importlib.metadata import version

        return version("cellflow")
    except Exception:
        return "unknown"
