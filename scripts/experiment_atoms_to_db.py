#!/usr/bin/env python
"""Build an ultrack data.db directly from atom-based candidates and round-trip it.

This validates the *database-creation* half of the new pipeline: instead of
ultrack's higra hierarchy (which only stores strictly-nested candidates — for
touching basins A,B,C it has A, B, C, A∪B, A∪B∪C but never B∪C), we

  1. extract atoms (territory split by residual contour ridges),
  2. build the atom adjacency graph (RAG),
  3. enumerate *every* connected union of up to ``--max-atoms`` atoms whose total
     area is ≤ ``--max-area`` — this is the richer structure that contains the
     missing permutations,
  4. write each union as a NodeDB row (pickled ``ultrack`` ``Node``) and every
     pair of unions that share an atom as an OverlapDB mutual-exclusion row,
  5. run ultrack's own linking + ILP solve on the result and export labels.

If linking + solve run and produce a non-trivial tracking, the schema we write
is compatible with the rest of the existing cellflow pipeline.

The DB is written to a *scratch* working dir (``--workdir``), never the
production ``2_nucleus/ultrack_workdir``.

Usage
-----
    python scripts/experiment_atoms_to_db.py --frames 10
    python scripts/experiment_atoms_to_db.py --frames 10 --max-atoms 3 --gui
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time as _time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_residual_atoms import (  # noqa: E402
    CONTOUR_PATH,
    FG_PATH,
    GT_PATH,
    DATA,
    residual,
    extract_atoms_frame,
)


# --------------------------------------------------------------------------- #
# atom graph + connected-union enumeration
# --------------------------------------------------------------------------- #
def atom_adjacency(atoms: np.ndarray) -> dict[int, set[int]]:
    """Region-adjacency graph of a label image: labels sharing a 4-conn border."""
    adj: dict[int, set[int]] = {}
    pairs = set()
    for a, b in (
        (atoms[:-1, :], atoms[1:, :]),
        (atoms[:, :-1], atoms[:, 1:]),
    ):
        m = (a != b) & (a != 0) & (b != 0)
        for u, v in zip(a[m].tolist(), b[m].tolist()):
            pairs.add((u, v) if u < v else (v, u))
    for lbl in np.unique(atoms):
        if lbl != 0:
            adj[int(lbl)] = set()
    for u, v in pairs:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def enum_connected_unions(
    adj: dict[int, set[int]],
    areas: dict[int, int],
    max_atoms: int,
    max_area: int,
) -> list[frozenset[int]]:
    """All connected atom-subsets of size 1..max_atoms with total area ≤ max_area.

    BFS growth with a global ``seen`` set: each connected subset is reached once
    its members are all present, deduped by frozenset. Correct and exhaustive;
    bounded by max_atoms/max_area so it does not explode.
    """
    seen: set[frozenset[int]] = set()
    out: list[frozenset[int]] = []
    frontier: list[tuple[frozenset[int], int]] = []
    for v, ar in areas.items():
        fs = frozenset((v,))
        seen.add(fs)
        out.append(fs)
        if ar <= max_area:
            frontier.append((fs, ar))
    while frontier:
        fs, area = frontier.pop()
        if len(fs) >= max_atoms:
            continue
        nbrs: set[int] = set()
        for v in fs:
            nbrs |= adj[v]
        nbrs -= fs
        for w in nbrs:
            na = area + areas[w]
            if na > max_area:
                continue
            nfs = fs | {w}
            if nfs in seen:
                continue
            seen.add(nfs)
            out.append(nfs)
            frontier.append((nfs, na))
    return out


def union_mask(atoms: np.ndarray, slices, members: list[int]):
    """Boolean crop + bbox (y0,x0,y1,x1) for the union of ``members`` labels."""
    ys0 = min(slices[m - 1][0].start for m in members)
    xs0 = min(slices[m - 1][1].start for m in members)
    ys1 = max(slices[m - 1][0].stop for m in members)
    xs1 = max(slices[m - 1][1].stop for m in members)
    crop = atoms[ys0:ys1, xs0:xs1]
    mask = np.isin(crop, members)
    return mask, np.array([ys0, xs0, ys1, xs1])


# --------------------------------------------------------------------------- #
# DB writing
# --------------------------------------------------------------------------- #
def build_database(
    fg, contour, atoms_stack, *, cfg, working_dir: Path,
    max_atoms: int, max_area: int,
):
    """Populate a fresh ultrack data.db with atom-union nodes + overlap rows."""
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import Base, NodeDB, OverlapDB, clear_all_data
    from ultrack.core.segmentation.node import Node

    from cellflow.tracking_ultrack.ingest import _build_ultrack_config

    ultrack_cfg = _build_ultrack_config(cfg, working_dir)
    db_path = ultrack_cfg.data_config.database_path
    clear_all_data(db_path)
    engine = sqla.create_engine(db_path)
    Base.metadata.create_all(engine)
    ultrack_cfg.data_config.metadata_add({"shape": list(fg.shape)})

    max_seg = cfg.max_segments_per_time
    n_frames = atoms_stack.shape[0]
    total_nodes = total_overlaps = 0
    per_frame_cands = []

    for t in range(n_frames):
        atoms = atoms_stack[t]
        n_atoms = int(atoms.max())
        if n_atoms == 0:
            per_frame_cands.append(0)
            continue
        slices = ndi.find_objects(atoms)
        ids, counts = np.unique(atoms, return_counts=True)
        areas = {int(i): int(c) for i, c in zip(ids, counts) if i != 0}
        adj = atom_adjacency(atoms)

        unions = enum_connected_unions(adj, areas, max_atoms, max_area)
        unions = [u for u in unions if areas_sum(u, areas) >= cfg.min_area or len(u) > 1]

        nodes = []
        # map atom-id -> list of (db_node_id) of candidates that contain it
        atom_members: dict[int, list[int]] = {a: [] for a in areas}
        index = 1
        for members in unions:
            mlist = sorted(members)
            mask, bbox = union_mask(atoms, slices, mlist)
            node = Node.from_mask(t, mask.astype(bool), bbox=bbox)
            nid = index + (t + 1) * max_seg
            node.id = nid
            node.time = t
            z, y, x = (0, *node.centroid) if len(node.centroid) == 2 else node.centroid
            nodes.append(NodeDB(
                id=nid, t_node_id=index, t_hier_id=1, t=t,
                z=int(z), y=int(y), x=int(x), area=int(node.area),
                frontier=-1.0, height=float(len(mlist)),
                pickle=pickle.dumps(node),
            ))
            for a in mlist:
                atom_members[a].append(nid)
            index += 1

        # overlaps: two candidates overlap iff they share an atom (atoms are a
        # disjoint partition, so pixel-overlap ⇔ shared atom). Dedup unordered.
        seen_pairs: set[tuple[int, int]] = set()
        overlaps = []
        for cand_ids in atom_members.values():
            for i in range(len(cand_ids)):
                for j in range(i + 1, len(cand_ids)):
                    a, b = cand_ids[i], cand_ids[j]
                    key = (a, b) if a < b else (b, a)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    overlaps.append(OverlapDB(node_id=key[0], ancestor_id=key[1]))

        with Session(engine) as session:
            session.add_all(nodes)
            session.add_all(overlaps)
            session.commit()
        engine.dispose()

        total_nodes += len(nodes)
        total_overlaps += len(overlaps)
        per_frame_cands.append(len(nodes))
        print(f"  t={t:>3}  atoms={n_atoms:>4}  candidates={len(nodes):>5}  "
              f"overlaps={len(overlaps):>6}")

    return total_nodes, total_overlaps, per_frame_cands


def areas_sum(members, areas):
    return sum(areas[m] for m in members)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--w", type=int, default=51, help="fg residual window")
    ap.add_argument("--wc", type=int, default=51, help="contour residual window")
    ap.add_argument("--fg-thr", type=float, default=0.002,
                    help="residual territory threshold")
    ap.add_argument("--floor", type=float, default=0.01, help="ridge noise cutoff")
    ap.add_argument("--atom-min-area", type=int, default=100,
                    help="merge atoms smaller than this into neighbours")
    ap.add_argument("--max-atoms", type=int, default=3,
                    help="max atoms per candidate union")
    ap.add_argument("--max-area", type=int, default=8000,
                    help="max candidate pixel area")
    ap.add_argument("--min-area", type=int, default=100,
                    help="min candidate area (cfg.min_area)")
    ap.add_argument("--workdir", type=str,
                    default=str(DATA / "2_nucleus" / "atoms_ultrack_workdir"))
    ap.add_argument("--no-score", action="store_true",
                    help="skip node-prob scoring (solve with uniform weights)")
    ap.add_argument("--score-signal", type=str, default=str(FG_PATH),
                    help="intensity image used as the node-prob quality signal")
    ap.add_argument("--no-solve", action="store_true")
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()

    from cellflow.tracking_ultrack.config import TrackingConfig
    cfg = TrackingConfig(
        min_area=args.min_area,
        max_area=args.max_area,
        seg_min_area=args.min_area,
        seg_max_area=args.max_area,
    )

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"workdir = {workdir}")

    fg = tifffile.imread(FG_PATH).astype(np.float32)[:args.frames]
    contour = tifffile.imread(CONTOUR_PATH).astype(np.float32)[:args.frames]
    print(f"frames={fg.shape[0]}  w={args.w} wc={args.wc} fg_thr={args.fg_thr} "
          f"max_atoms={args.max_atoms} max_area={args.max_area}")

    fg_resid = residual(fg, args.w | 1)
    ridge_resid = residual(contour, args.wc | 1)
    territory = fg_resid > args.fg_thr

    print("\nextracting atoms…")
    atoms_stack = np.zeros(fg.shape, np.int32)
    for t in range(fg.shape[0]):
        atoms_stack[t] = extract_atoms_frame(
            ridge_resid[t], territory[t], args.floor, args.atom_min_area)
    print(f"  atoms/frame = {np.mean([a.max() for a in atoms_stack]):.1f}")

    print("\nbuilding database (candidates + overlaps)…")
    t0 = _time.time()
    n_nodes, n_ovl, per_frame = build_database(
        fg, contour, atoms_stack, cfg=cfg, working_dir=workdir,
        max_atoms=args.max_atoms, max_area=args.max_area)
    print(f"\nTOTAL: {n_nodes} candidate nodes, {n_ovl} overlap rows "
          f"({np.mean(per_frame):.0f} cands/frame) in {_time.time()-t0:.1f}s")

    if not args.no_score:
        print("\nscoring node probabilities…")
        from cellflow.tracking_ultrack.seed_prior import write_seed_prior_node_probs
        rep = write_seed_prior_node_probs(workdir, args.score_signal, cfg)
        print(f"  scored {rep.scored} nodes ({rep.seeds} REAL anchors)")

    if args.no_solve:
        print("\n(--no-solve: stopping after DB build)")
        return

    print("\nlinking…")
    from cellflow.tracking_ultrack.linking import run_linking
    from cellflow.tracking_ultrack.solve import run_solve
    for step, total, label in run_linking(workdir, cfg):
        print(f"  [link {step}/{total}] {label}")

    print("\nsolving…")
    for step, total, label in run_solve(workdir, cfg, use_annotations=False):
        print(f"  [solve {step}/{total}] {label}")

    print("\nexporting tracked labels…")
    from cellflow.tracking_ultrack.export import export_tracked_labels
    out_path = workdir / "atoms_tracked_labels.tif"
    labels = export_tracked_labels(workdir, cfg, out_path)
    n_tracks = len(np.unique(labels)) - 1
    print(f"  exported {out_path}")
    print(f"  selected segments/frame = "
          f"{np.mean([len(np.unique(l)) - 1 for l in labels]):.1f}")
    print(f"  distinct track ids = {n_tracks}")

    if not args.gui:
        print("\n(pass --gui to inspect)")
        return

    import napari
    gt = tifffile.imread(GT_PATH)[:args.frames] if GT_PATH.exists() else None
    v = napari.Viewer()
    v.add_image(fg, name="fg", colormap="gray")
    v.add_image(contour, name="contour", colormap="inferno",
                blending="additive", visible=False)
    v.add_labels(atoms_stack, name="atoms", opacity=0.4, visible=False)
    v.add_labels(labels.astype(np.int32), name="atoms_tracked", opacity=0.6)
    if gt is not None:
        v.add_labels(gt, name="GT", opacity=0.3, visible=False)
    napari.run()


if __name__ == "__main__":
    main()
