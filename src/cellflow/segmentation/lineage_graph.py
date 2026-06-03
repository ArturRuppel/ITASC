"""Node/edge structure for the unified lineage canvas (TrackScheme-with-thumbnails).

This is the data half of the combined correction visualization: a per-track,
per-frame node graph where each node is a cell at a frame and each edge links a
cell's consecutive present frames (so a gap reads as an edge that skips a row).
It carries structure only — the Qt canvas places nodes by track column and frame
row and renders each node as that frame's film-strip crop.

Array-only, like :mod:`cellflow.segmentation.lineage` — parent/daughter division
edges live in the Ultrack database and are layered on top later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One cell at one frame."""

    cell_id: int
    t: int


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """A link between a cell's nodes at ``t0`` and the next frame it appears."""

    cell_id: int
    t0: int
    t1: int


@dataclass(frozen=True, slots=True)
class LineageGraph:
    """Nodes + edges for every track."""

    n_frames: int
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

    def nodes_by_track(self) -> dict[int, list[GraphNode]]:
        """Per-track nodes, each list sorted by frame."""
        out: dict[int, list[GraphNode]] = {}
        for node in self.nodes:
            out.setdefault(node.cell_id, []).append(node)
        for nodes in out.values():
            nodes.sort(key=lambda n: n.t)
        return out


def _as_tyx(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 4 and a.shape[1] == 1:
        a = a[:, 0]
    if a.ndim == 2:
        a = a[np.newaxis]
    if a.ndim != 3:
        raise ValueError(f"{name} must be (T, Y, X); got shape {a.shape}")
    return a


def build_lineage_graph(tracked: np.ndarray) -> LineageGraph:
    """Build the per-track node/edge graph from a ``(T, Y, X)`` label stack."""
    arr = _as_tyx(tracked, "tracked")
    n_t = arr.shape[0]

    frames_of: dict[int, list[int]] = {}
    for t in range(n_t):
        for cell_id in np.unique(arr[t]).tolist():
            if cell_id == 0:
                continue
            frames_of.setdefault(int(cell_id), []).append(t)

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    for cell_id, frames in frames_of.items():
        frames.sort()
        for t in frames:
            nodes.append(GraphNode(cell_id=cell_id, t=t))
        # Link consecutive present frames; a gap just makes the edge skip rows.
        for t0, t1 in zip(frames, frames[1:]):
            edges.append(GraphEdge(cell_id=cell_id, t0=t0, t1=t1))

    return LineageGraph(n_frames=n_t, nodes=tuple(nodes), edges=tuple(edges))


def assign_columns(graph: LineageGraph) -> dict[int, int]:
    """Map each track id to a canvas column, ordered by first appearance then id.

    A simple one-column-per-track packing — enough for the v1 canvas, where each
    track is an independent vertical chain. (Division-aware packing that places
    daughters under their parent is a later, DB-backed concern.)
    """
    first_frame: dict[int, int] = {}
    for cell_id, nodes in graph.nodes_by_track().items():
        first_frame[cell_id] = nodes[0].t
    order = sorted(first_frame, key=lambda cid: (first_frame[cid], cid))
    return {cell_id: col for col, cell_id in enumerate(order)}


__all__ = [
    "GraphEdge",
    "GraphNode",
    "LineageGraph",
    "assign_columns",
    "build_lineage_graph",
]
