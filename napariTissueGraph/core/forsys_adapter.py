"""Bridge between napariTissueGraph data structures and ForSys.

ForSys (Borges et al., iScience 2025) infers membrane tensions and cell
pressures from tissue geometry.  This module converts our TissueGraphFrame
into the ForSys Frame representation and maps results back.

The conversion uses junction topology directly: junction endpoints are
clustered to find triple junctions, and each junction maps to exactly one
ForSys BigEdge.  This avoids duplicate BigEdges that arise from
tolerance-based vertex deduplication.

All ForSys imports are guarded so the rest of the codebase works without
ForSys installed.
"""
import logging
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..structures import CellData, JunctionData, TissueGraphFrame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
_FORSYS_AVAILABLE = False
try:
    import forsys
    import forsys.vertex as fvertex
    import forsys.edge as fedge
    import forsys.cell as fcell
    import forsys.frames as fframes
    import forsys.virtual_edges as fvirtual
    _FORSYS_AVAILABLE = True
except ImportError:
    pass


def forsys_available() -> bool:
    """Return True if the forsys package is importable."""
    return _FORSYS_AVAILABLE


def _require_forsys():
    if not _FORSYS_AVAILABLE:
        raise ImportError(
            "forsys is not installed. Install it with: "
            "pip install napariTissueGraph[forces]"
        )


# ---------------------------------------------------------------------------
# Tissue → ForSys conversion  (junction topology path)
# ---------------------------------------------------------------------------

def tissue_frame_to_forsys(
    frame: TissueGraphFrame,
    vertices_per_edge: int = 6,
    endpoint_cluster_tol: float = 3.0,
) -> "fframes.Frame":
    """Convert a TissueGraphFrame into a ForSys Frame.

    Uses junction topology directly: each junction in the TissueGraphFrame
    maps to exactly one ForSys BigEdge.  Junction endpoints are clustered
    to identify triple junctions (shared ForSys Vertex objects), guaranteeing
    no duplicate BigEdges.

    Parameters
    ----------
    frame : TissueGraphFrame
        A single timepoint from napariTissueGraph.
    vertices_per_edge : int
        Target number of vertices per BigEdge (including TJ endpoints).
        Junction coordinates are subsampled to this count.
    endpoint_cluster_tol : float
        Distance tolerance (pixels) for clustering junction endpoints
        into triple junctions.

    Returns
    -------
    forsys.frames.Frame
        Ready for the ForSys solver.
    """
    _require_forsys()

    # --- Step 1: Collect internal junctions ---
    internal_junctions = {}
    for key, jd in frame.junctions.items():
        if "border" in jd.tags or 0 in key:
            continue
        if jd.coordinates is None or len(jd.coordinates) < 2:
            continue
        internal_junctions[key] = jd

    if not internal_junctions:
        raise ValueError("No internal junctions found in frame.")

    # --- Step 2: Cluster junction endpoints → triple junctions ---
    all_endpoints = []
    endpoint_info = []  # (junction_key, 'start'|'end')
    for key, jd in internal_junctions.items():
        all_endpoints.append(jd.coordinates[0])
        endpoint_info.append((key, 'start'))
        all_endpoints.append(jd.coordinates[-1])
        endpoint_info.append((key, 'end'))

    all_endpoints = np.array(all_endpoints)
    ep_tree = cKDTree(all_endpoints)

    n_ep = len(all_endpoints)
    ep_label = np.full(n_ep, -1, dtype=int)
    next_cluster = 0
    for i in range(n_ep):
        if ep_label[i] >= 0:
            continue
        neighbors = ep_tree.query_ball_point(all_endpoints[i], endpoint_cluster_tol)
        for j in neighbors:
            if ep_label[j] < 0:
                ep_label[j] = next_cluster
        next_cluster += 1

    # Cluster centres = triple junction positions (y, x)
    tj_positions = np.zeros((next_cluster, 2))
    tj_counts = np.zeros(next_cluster)
    for i in range(n_ep):
        tj_positions[ep_label[i]] += all_endpoints[i]
        tj_counts[ep_label[i]] += 1
    tj_positions /= tj_counts[:, None]

    # Map each junction to its endpoint TJ IDs
    junction_tjs = {}  # key -> (start_tj_id, end_tj_id)
    for i, (key, which) in enumerate(endpoint_info):
        tj_id = int(ep_label[i])
        if key not in junction_tjs:
            junction_tjs[key] = [None, None]
        if which == 'start':
            junction_tjs[key][0] = tj_id
        else:
            junction_tjs[key][1] = tj_id

    num_tjs = next_cluster
    logger.info(
        f"Frame {frame.frame}: {len(internal_junctions)} junctions, "
        f"{num_tjs} triple junctions"
    )

    # --- Step 3: Create ForSys Vertex objects for TJ positions ---
    # TJ vertices get IDs 0 .. num_tjs-1; interior vertices continue from there.
    # ForSys uses (x, y); our coordinates are (y, x).
    fs_vertices = {}
    for tj_id in range(num_tjs):
        y, x = tj_positions[tj_id]
        fs_vertices[tj_id] = fvertex.Vertex(id=tj_id, x=float(x), y=float(y))

    next_vid = num_tjs

    # --- Step 4: Build vertex chain + SmallEdges for each junction ---
    junction_chains = {}  # key -> list of vertex IDs
    fs_edges = {}
    next_eid = 0

    for key, jd in internal_junctions.items():
        start_tj, end_tj = junction_tjs[key]

        # Skip degenerate self-loops
        if start_tj == end_tj:
            logger.debug(f"Junction {key}: same TJ at both ends, skipping")
            continue

        coords = jd.coordinates  # Nx2 (y, x)
        n_pts = len(coords)

        # Subsample to vertices_per_edge evenly-spaced points along the path.
        # Use linspace over the full range so interior vertices are well-
        # separated from the TJ endpoints (not just 0.5 px away).
        n_interior = max(0, vertices_per_edge - 2)
        if n_pts <= vertices_per_edge:
            interior_coords = coords[1:-1] if n_pts > 2 else np.empty((0, 2))
        else:
            all_indices = np.round(
                np.linspace(0, n_pts - 1, vertices_per_edge)
            ).astype(int)
            # First and last are TJ endpoints (handled separately); take middle
            interior_coords = coords[all_indices[1:-1]]

        # Chain: start_tj → interior vertices → end_tj
        chain = [start_tj]
        for pt in interior_coords:
            y, x = pt
            fs_vertices[next_vid] = fvertex.Vertex(
                id=next_vid, x=float(x), y=float(y),
            )
            chain.append(next_vid)
            next_vid += 1
        chain.append(end_tj)
        junction_chains[key] = chain

        # SmallEdges between consecutive vertices
        for i in range(len(chain) - 1):
            v1_id, v2_id = chain[i], chain[i + 1]
            if v1_id != v2_id:
                fs_edges[next_eid] = fedge.SmallEdge(
                    id=next_eid,
                    v1=fs_vertices[v1_id],
                    v2=fs_vertices[v2_id],
                )
                next_eid += 1

    # Remove junctions that were skipped (self-loops)
    internal_junctions = {k: v for k, v in internal_junctions.items()
                         if k in junction_chains}

    # --- Step 5: Build Cell objects ---
    # Group junctions by participating cell
    cell_junc_keys = defaultdict(list)
    for key in internal_junctions:
        for cid in key:
            cell_junc_keys[cid].append(key)

    fs_cells = {}
    for cell_id, jkeys in cell_junc_keys.items():
        if len(jkeys) < 2:
            continue

        # Build adjacency: TJ → junctions touching it (for this cell)
        tj_to_juncs = defaultdict(list)
        for jk in jkeys:
            s, e = junction_tjs[jk]
            tj_to_juncs[s].append(jk)
            tj_to_juncs[e].append(jk)

        # Walk around the cell following shared TJ vertices
        ordered = []  # list of (jkey, forward: bool)
        visited = set()

        current_jk = jkeys[0]
        visited.add(current_jk)
        s, e = junction_tjs[current_jk]
        ordered.append((current_jk, True))  # forward: s → e
        exit_tj = e

        for _ in range(len(jkeys) - 1):
            found = False
            for neighbor_jk in tj_to_juncs[exit_tj]:
                if neighbor_jk in visited:
                    continue
                s_n, e_n = junction_tjs[neighbor_jk]
                if s_n == exit_tj:
                    ordered.append((neighbor_jk, True))
                    exit_tj = e_n
                else:
                    ordered.append((neighbor_jk, False))
                    exit_tj = s_n
                visited.add(neighbor_jk)
                current_jk = neighbor_jk
                found = True
                break
            if not found:
                break

        if len(ordered) < 2:
            logger.debug(
                f"Cell {cell_id}: only {len(ordered)} connected junctions, "
                f"skipping"
            )
            continue

        # Build vertex ring from ordered junction chains
        cell_vids = []
        for jk, forward in ordered:
            chain = junction_chains[jk]
            if forward:
                segment = chain[:-1]  # exclude last (shared with next junction)
            else:
                segment = chain[::-1][:-1]
            cell_vids.extend(segment)

        # Remove consecutive duplicates
        deduped = [cell_vids[0]]
        for vid in cell_vids[1:]:
            if vid != deduped[-1]:
                deduped.append(vid)
        if len(deduped) > 1 and deduped[-1] == deduped[0]:
            deduped.pop()

        if len(deduped) < 3:
            continue

        cell_vertices = [fs_vertices[vid] for vid in deduped]
        fs_cells[cell_id] = fcell.Cell(
            id=cell_id,
            vertices=cell_vertices,
            is_border=False,
        )

    if not fs_cells:
        raise ValueError("No valid cells could be constructed.")

    logger.info(
        f"Frame {frame.frame}: built {len(fs_cells)} cells from "
        f"{len(internal_junctions)} junctions"
    )

    # --- Step 6: Build Frame manually (bypass __post_init__) ---
    frame_obj = object.__new__(fframes.Frame)
    frame_obj.frame_id = frame.frame
    frame_obj.vertices = fs_vertices
    frame_obj.edges = fs_edges
    frame_obj.cells = fs_cells
    frame_obj.time = 0.0
    frame_obj.gt = False
    frame_obj.big_edge_gt_tension = {}
    frame_obj.big_edge_tension = {}
    frame_obj.big_edges = {}

    # BigEdge list: one per junction chain (guaranteed unique by construction)
    big_edges_list = []
    junction_key_order = list(internal_junctions.keys())
    for key in junction_key_order:
        big_edges_list.append(junction_chains[key])

    skipped = 0
    for bid, chain in enumerate(big_edges_list):
        try:
            vertex_objects = [fs_vertices[vid] for vid in chain]
            frame_obj.big_edges[bid] = fedge.BigEdge(bid, vertex_objects)
        except (IndexError, KeyError, AssertionError) as exc:
            for vid in chain:
                v = fs_vertices.get(vid)
                if v is not None and bid in v.own_big_edges:
                    v.own_big_edges.remove(bid)
            skipped += 1
            logger.debug(f"BigEdge {bid} skipped: {exc}")

    if skipped > 0:
        logger.info(
            f"Frame {frame.frame}: skipped {skipped}/{len(big_edges_list)} "
            f"BigEdges"
        )

    frame_obj.big_edges_list = big_edges_list

    # Identify external edges
    frame_obj.external_edges_id = []
    try:
        border_edges = fvirtual.get_border_edge(big_edges_list, fs_vertices)
        frame_obj.external_edges_id = [
            big_edges_list.index(e) for e in border_edges
        ]
    except Exception:
        pass

    # Determine which TJ vertices are "valid" for force balance:
    # a vertex needs ≥3 non-external BigEdges to produce a meaningful
    # equation (ForSys requires ≥3 non-zero entries per row).
    def _is_valid_tj(v):
        if len(v.ownCells) <= 2:
            return False
        n_non_ext = sum(
            1 for beid in v.own_big_edges
            if beid in frame_obj.big_edges
            and not frame_obj.big_edges[beid].external
        )
        return n_non_ext >= 3

    valid_tjs = {vid for vid, v in fs_vertices.items() if _is_valid_tj(v)}

    # Build internal BigEdges list: at least one endpoint must be a valid TJ
    def _is_internal(eid, chain):
        if eid not in frame_obj.big_edges:
            return False
        if eid in frame_obj.external_edges_id:
            return False
        v0 = fs_vertices.get(chain[0])
        vn = fs_vertices.get(chain[-1])
        if v0 is None or vn is None:
            return False
        if chain[0] not in valid_tjs and chain[-1] not in valid_tjs:
            return False
        if len(v0.own_big_edges) < 2 or len(vn.own_big_edges) < 2:
            return False
        return True

    frame_obj.internal_big_edges_vertices = [
        chain for eid, chain in enumerate(big_edges_list)
        if _is_internal(eid, chain)
    ]
    frame_obj.internal_big_edges = [
        frame_obj.big_edges[eid]
        for eid, chain in enumerate(big_edges_list)
        if _is_internal(eid, chain)
    ]

    # Calculate cell neighbors
    for cell in fs_cells.values():
        try:
            cell.calculate_neighbors()
        except Exception:
            pass

    frame_obj.border_vertices = frame_obj.get_external_edges()

    n_internal = len(frame_obj.internal_big_edges)
    n_total = len(frame_obj.big_edges)
    logger.info(
        f"Frame {frame.frame}: {len(fs_vertices)} vertices, "
        f"{n_total} BigEdges ({n_internal} internal), "
        f"{len(fs_cells)} cells"
    )

    return frame_obj


# ---------------------------------------------------------------------------
# ForSys → Tissue result mapping
# ---------------------------------------------------------------------------

def _match_big_edge_to_junction(
    big_edge,
    junctions: Dict,
    cells: Dict[int, CellData],
    tol: float = 5.0,
) -> Optional[frozenset]:
    """Find the TissueGraphFrame junction that corresponds to a ForSys BigEdge.

    First tries matching by the pair of cells the big edge separates.
    Falls back to midpoint distance matching.

    Returns the junction dict key (frozenset) or None.
    """
    own_cells = big_edge.own_cells
    if len(own_cells) == 2:
        pair = frozenset(own_cells)
        if pair in junctions:
            return pair

    # Midpoint matching fallback
    be_midx = np.mean(big_edge.xs)
    be_midy = np.mean(big_edge.ys)

    best_key = None
    best_dist = float("inf")
    for key, jd in junctions.items():
        if 0 in key:
            continue
        # Our midpoint is (y, x), ForSys uses (x, y)
        dist = np.sqrt((jd.midpoint[0] - be_midy) ** 2 +
                        (jd.midpoint[1] - be_midx) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_key = key

    if best_dist < tol:
        return best_key
    return None


def forsys_results_to_tissue(
    forsys_frame: "fframes.Frame",
    tissue_frame: TissueGraphFrame,
) -> None:
    """Write ForSys tension and pressure results back into a TissueGraphFrame.

    Modifies tissue_frame in place.
    """
    _require_forsys()

    # --- Map tensions from BigEdges to JunctionData ---
    matched = 0
    for big_edge in forsys_frame.big_edges.values():
        if big_edge.external:
            continue

        key = _match_big_edge_to_junction(
            big_edge, tissue_frame.junctions, tissue_frame.cells
        )
        if key is not None:
            tissue_frame.junctions[key].tension = float(big_edge.tension)
            matched += 1

    n_internal = sum(1 for be in forsys_frame.big_edges.values() if not be.external)
    logger.info(
        f"Frame {tissue_frame.frame}: matched {matched}/{n_internal} "
        f"internal edges to junctions"
    )

    # --- Map pressures from Cells to CellData ---
    for cell_id, fs_cell in forsys_frame.cells.items():
        if cell_id in tissue_frame.cells and fs_cell.pressure is not None:
            tissue_frame.cells[cell_id].pressure = float(fs_cell.pressure)
