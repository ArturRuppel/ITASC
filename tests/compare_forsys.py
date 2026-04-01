"""Compare native ForSys pipeline vs cellflow adapter.

Runs both pipelines on the same label image (t_0_cells.tif) and compares
the inferred tensions and pressures.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Load the label image ──────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "forsys_test"
LABEL_PATH = DATA_DIR / "t_0_cells.tif"
SKELETON_PATH = DATA_DIR / "forsys_input" / "t_0" / "handCorrection.tif"

from PIL import Image

label_img = np.array(Image.open(LABEL_PATH))
print(f"Label image shape: {label_img.shape}, dtype: {label_img.dtype}")
print(f"Unique labels: {len(np.unique(label_img))} (including background={0 in label_img})")
print()

# ══════════════════════════════════════════════════════════════════════
# 1.  NATIVE ForSys pipeline  (Skeleton → Frame → solver)
# ══════════════════════════════════════════════════════════════════════
import forsys
import forsys.skeleton as fskeleton
import forsys.frames as fframes
import forsys.virtual_edges as fvirtual

print("=" * 70)
print("NATIVE ForSys (from skeleton TIF)")
print("=" * 70)

skel = fskeleton.Skeleton(str(SKELETON_PATH))
vertices, edges, cells = skel.create_lattice()
print(f"  Vertices: {len(vertices)}, SmallEdges: {len(edges)}, Cells: {len(cells)}")

native_frame = fframes.Frame(0, vertices, edges, cells)
print(f"  BigEdges total: {len(native_frame.big_edges)}")
print(f"  BigEdges internal: {len(native_frame.internal_big_edges)}")
print(f"  BigEdges external: {len(native_frame.external_edges_id)}")

fs_native = forsys.ForSys(frames={0: native_frame})
with np.errstate(invalid="ignore"):
    fs_native.build_force_matrix(when=0)
    fs_native.solve_stress(when=0, allow_negatives=False)
    fs_native.build_pressure_matrix(when=0)
    fs_native.solve_pressure(when=0, method="lagrange_pressure")

# Collect native results by cell pair
native_tensions = {}  # frozenset(cell_a, cell_b) → tension
for be in native_frame.big_edges.values():
    if be.external:
        continue
    if len(be.own_cells) == 2:
        pair = frozenset(be.own_cells)
        native_tensions[pair] = be.tension

native_pressures = {}  # cell_id → pressure
for cid, cell in native_frame.cells.items():
    if cell.pressure is not None:
        native_pressures[cid] = cell.pressure

print(f"  Tensions assigned: {len(native_tensions)}")
print(f"  Pressures assigned: {len(native_pressures)}")

t_vals = list(native_tensions.values())
p_vals = list(native_pressures.values())
print(f"  Tension range: [{min(t_vals):.4f}, {max(t_vals):.4f}], "
      f"mean={np.mean(t_vals):.4f}")
print(f"  Pressure range: [{min(p_vals):.4f}, {max(p_vals):.4f}], "
      f"mean={np.mean(p_vals):.4f}")
print()

# ══════════════════════════════════════════════════════════════════════
# 2.  NATIVE ForSys from label image (NPY/contour approach)
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("NATIVE ForSys (from label image, per-cell contours)")
print("=" * 70)

# Replicate what Skeleton does for NPY: per-cell binary mask → contour
import cv2
import forsys.vertex as fvertex
import forsys.edge as fedge
import forsys.cell as fcell

vertex_id = 0
edge_id = 0
cell_id_counter = 0
npy_vertices = {}
npy_edges = {}
npy_cells = {}
coords_to_key = {}
edges_added = []

# Use expand_labels to fill any gaps (as ForSys does for NPY)
from skimage.segmentation import expand_labels
expanded = expand_labels(label_img, distance=2)

cell_ids_sorted = sorted(np.unique(expanded))
cell_ids_sorted = [c for c in cell_ids_sorted if c > 0]  # skip background

for cell_label in cell_ids_sorted:
    binary_mask = (expanded == cell_label).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours:
        continue
    # Use largest contour
    which = np.argmax([c.shape[0] for c in contours])
    polygon = contours[which].astype(int).squeeze()
    if polygon.ndim != 2:
        continue

    cell_vertices_list = []
    for coords in polygon:
        key = tuple(coords)
        if key in coords_to_key:
            vid = coords_to_key[key]
        else:
            vid = vertex_id
            npy_vertices[vid] = fvertex.Vertex(int(vid), float(coords[0]), float(coords[1]))
            coords_to_key[key] = vid
            vertex_id += 1
        cell_vertices_list.append(npy_vertices[vid])

    # Create small edges
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        v1 = npy_vertices[coords_to_key[tuple(polygon[i])]]
        v2 = npy_vertices[coords_to_key[tuple(polygon[j])]]
        if (v1.id, v2.id) not in edges_added and (v2.id, v1.id) not in edges_added:
            npy_edges[edge_id] = fedge.SmallEdge(edge_id, v1, v2)
            edge_id += 1
            edges_added.append((v1.id, v2.id))

    npy_cells[cell_label] = fcell.Cell(cell_label, cell_vertices_list, {})

# Mark border cells/edges
for current_cell in npy_cells.values():
    if np.any([len(v.ownCells) == 1 for v in current_cell.vertices]):
        current_cell.is_border = True
for e in npy_edges.values():
    e.external = (len(e.v1.ownCells) == 1 or len(e.v2.ownCells) == 1)

print(f"  Vertices: {len(npy_vertices)}, SmallEdges: {len(npy_edges)}, Cells: {len(npy_cells)}")

npy_frame = fframes.Frame(0, npy_vertices, npy_edges, npy_cells)
print(f"  BigEdges total: {len(npy_frame.big_edges)}")
print(f"  BigEdges internal: {len(npy_frame.internal_big_edges)}")
print(f"  BigEdges external: {len(npy_frame.external_edges_id)}")

fs_npy = forsys.ForSys(frames={0: npy_frame})
with np.errstate(invalid="ignore"):
    fs_npy.build_force_matrix(when=0)
    fs_npy.solve_stress(when=0, allow_negatives=False)
    fs_npy.build_pressure_matrix(when=0)
    fs_npy.solve_pressure(when=0, method="lagrange_pressure")

# Collect NPY results by cell pair
npy_tensions = {}
for be in npy_frame.big_edges.values():
    if be.external:
        continue
    if len(be.own_cells) == 2:
        pair = frozenset(be.own_cells)
        npy_tensions[pair] = be.tension

npy_pressures = {}
for cid, cell in npy_frame.cells.items():
    if cell.pressure is not None:
        npy_pressures[cid] = cell.pressure

print(f"  Tensions assigned: {len(npy_tensions)}")
print(f"  Pressures assigned: {len(npy_pressures)}")

t_vals2 = list(npy_tensions.values())
p_vals2 = list(npy_pressures.values())
if t_vals2:
    print(f"  Tension range: [{min(t_vals2):.4f}, {max(t_vals2):.4f}], "
          f"mean={np.mean(t_vals2):.4f}")
if p_vals2:
    print(f"  Pressure range: [{min(p_vals2):.4f}, {max(p_vals2):.4f}], "
          f"mean={np.mean(p_vals2):.4f}")
print()

# ══════════════════════════════════════════════════════════════════════
# 3.  OUR ADAPTER pipeline
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("cellflow ADAPTER")
print("=" * 70)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cellflow.backend.graph import build_from_labels
from cellflow.utils.mechanics import infer_forces

label_stack = label_img[np.newaxis, ...]  # (1, H, W)
series = build_from_labels(label_stack)
frame = series.frames[0]

print(f"  Cells: {len(frame.cells)}, Junctions: {len(frame.junctions)}")

import logging
logging.basicConfig(level=logging.INFO)
infer_forces(series, method="static")

# Collect adapter results
adapter_tensions = {}
for pair, jd in frame.junctions.items():
    if jd.tension is not None:
        adapter_tensions[pair] = jd.tension

adapter_pressures = {}
for cid, cd in frame.cells.items():
    if cd.pressure is not None:
        adapter_pressures[cid] = cd.pressure

print(f"  Tensions assigned: {len(adapter_tensions)}")
print(f"  Pressures assigned: {len(adapter_pressures)}")

t_vals3 = list(adapter_tensions.values())
p_vals3 = list(adapter_pressures.values())
if t_vals3:
    print(f"  Tension range: [{min(t_vals3):.4f}, {max(t_vals3):.4f}], "
          f"mean={np.mean(t_vals3):.4f}")
if p_vals3:
    print(f"  Pressure range: [{min(p_vals3):.4f}, {max(p_vals3):.4f}], "
          f"mean={np.mean(p_vals3):.4f}")
print()

# ══════════════════════════════════════════════════════════════════════
# 4.  COMPARISON
# ══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("COMPARISON: native (skeleton) vs adapter (labels)")
print("=" * 70)

# --- Compare by matching cell pairs ---
# The native pipeline uses Skeleton cell IDs (0-indexed),
# while our pipeline uses label-image cell IDs.
# The skeleton pipeline's cell IDs are arbitrary (order of contour detection),
# so we match by cell position instead.

# Build position maps
native_cell_positions = {}
for cid, cell in native_frame.cells.items():
    xs = [v.x for v in cell.vertices]
    ys = [v.y for v in cell.vertices]
    native_cell_positions[cid] = (np.mean(xs), np.mean(ys))

adapter_cell_positions = {}
for cid, cd in frame.cells.items():
    adapter_cell_positions[cid] = (cd.position[1], cd.position[0])  # (x, y)

# Match native→adapter cells by nearest centroid
from scipy.spatial.distance import cdist

native_ids = sorted(native_cell_positions.keys())
adapter_ids = sorted(adapter_cell_positions.keys())

native_pos = np.array([native_cell_positions[c] for c in native_ids])
adapter_pos = np.array([adapter_cell_positions[c] for c in adapter_ids])

D = cdist(native_pos, adapter_pos)
native_to_adapter = {}
for i, nid in enumerate(native_ids):
    j = np.argmin(D[i])
    if D[i, j] < 20.0:  # reasonable matching threshold
        native_to_adapter[nid] = adapter_ids[j]

print(f"\nCell matching: {len(native_to_adapter)}/{len(native_ids)} native cells matched to adapter cells")

# Compare pressures for matched cells
matched_p = []
for nid, aid in native_to_adapter.items():
    if nid in native_pressures and aid in adapter_pressures:
        matched_p.append((nid, aid, native_pressures[nid], adapter_pressures[aid]))

if matched_p:
    print(f"\nPressure comparison ({len(matched_p)} matched cells):")
    native_p = np.array([m[2] for m in matched_p])
    adapter_p = np.array([m[3] for m in matched_p])
    corr = np.corrcoef(native_p, adapter_p)[0, 1] if len(native_p) > 1 else 0
    rmse = np.sqrt(np.mean((native_p - adapter_p) ** 2))
    print(f"  Correlation: {corr:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Native  mean={np.mean(native_p):.4f}, std={np.std(native_p):.4f}")
    print(f"  Adapter mean={np.mean(adapter_p):.4f}, std={np.std(adapter_p):.4f}")

# Compare tensions by converting native cell pairs to adapter cell pairs
matched_t = []
for pair, tension in native_tensions.items():
    pair_list = list(pair)
    if len(pair_list) != 2:
        continue
    a_id0 = native_to_adapter.get(pair_list[0])
    a_id1 = native_to_adapter.get(pair_list[1])
    if a_id0 is not None and a_id1 is not None:
        a_pair = frozenset((a_id0, a_id1))
        if a_pair in adapter_tensions:
            matched_t.append((pair, a_pair, tension, adapter_tensions[a_pair]))

if matched_t:
    print(f"\nTension comparison ({len(matched_t)} matched junctions):")
    native_t = np.array([m[2] for m in matched_t])
    adapter_t = np.array([m[3] for m in matched_t])
    corr = np.corrcoef(native_t, adapter_t)[0, 1] if len(native_t) > 1 else 0
    rmse = np.sqrt(np.mean((native_t - adapter_t) ** 2))
    print(f"  Correlation: {corr:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Native  mean={np.mean(native_t):.4f}, std={np.std(native_t):.4f}")
    print(f"  Adapter mean={np.mean(adapter_t):.4f}, std={np.std(adapter_t):.4f}")

    # Show worst mismatches
    diffs = np.abs(native_t - adapter_t)
    worst = np.argsort(diffs)[::-1][:10]
    print(f"\n  Top 10 worst tension mismatches:")
    print(f"  {'Native pair':>20s}  {'Adapter pair':>20s}  {'Native':>8s}  {'Adapter':>8s}  {'Diff':>8s}")
    for idx in worst:
        m = matched_t[idx]
        print(f"  {str(set(m[0])):>20s}  {str(set(m[1])):>20s}  {m[2]:8.4f}  {m[3]:8.4f}  {diffs[idx]:8.4f}")

# ── Also compare NPY-path native vs adapter ──
print()
print("=" * 70)
print("COMPARISON: native (from labels, NPY-style) vs adapter")
print("=" * 70)

# NPY uses same cell IDs as labels, so direct comparison
common_t_pairs = set(npy_tensions.keys()) & set(adapter_tensions.keys())
print(f"\nCommon tension pairs: {len(common_t_pairs)}")
if common_t_pairs:
    n_t = np.array([npy_tensions[p] for p in common_t_pairs])
    a_t = np.array([adapter_tensions[p] for p in common_t_pairs])
    corr = np.corrcoef(n_t, a_t)[0, 1] if len(n_t) > 1 else 0
    rmse = np.sqrt(np.mean((n_t - a_t) ** 2))
    print(f"  Tension correlation: {corr:.4f}")
    print(f"  Tension RMSE: {rmse:.4f}")
    print(f"  NPY     mean={np.mean(n_t):.4f}, std={np.std(n_t):.4f}")
    print(f"  Adapter mean={np.mean(a_t):.4f}, std={np.std(a_t):.4f}")

common_p_cells = set(npy_pressures.keys()) & set(adapter_pressures.keys())
print(f"\nCommon pressure cells: {len(common_p_cells)}")
if common_p_cells:
    n_p = np.array([npy_pressures[c] for c in common_p_cells])
    a_p = np.array([adapter_pressures[c] for c in common_p_cells])
    corr = np.corrcoef(n_p, a_p)[0, 1] if len(n_p) > 1 else 0
    rmse = np.sqrt(np.mean((n_p - a_p) ** 2))
    print(f"  Pressure correlation: {corr:.4f}")
    print(f"  Pressure RMSE: {rmse:.4f}")
    print(f"  NPY     mean={np.mean(n_p):.4f}, std={np.std(n_p):.4f}")
    print(f"  Adapter mean={np.mean(a_p):.4f}, std={np.std(a_p):.4f}")

print("\nDone.")
