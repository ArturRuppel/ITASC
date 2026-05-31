#!/usr/bin/env python
"""Deduplicate a merged multi-tree ultrack data.db into one candidate set.

The DB produced by the multi-threshold builder contains several segmentation
trees (one per threshold). Many nodes are near-duplicates of each other, and a
nucleus appears at several "halo extents" sharing roughly the same centroid.

For each timepoint we cluster nodes whose centroids fall within --radius px of
one another (transitively) and keep only the highest-node_prob node per cluster
(ties broken toward smaller area, i.e. the tighter mask). Dropped nodes and the
``overlaps`` / ``links`` rows that reference them are cascade-deleted; the keeper
retains its own geometrically-correct constraints (we do NOT redirect the
dropped node's overlaps onto the keeper, which would over-constrain a tight core
with a halo variant's extra neighbours).

Dry-run by default; pass --apply to mutate. A backup should sit alongside the DB.

    python scripts/dedup_ultrack_db.py            # dry run, report only
    python scripts/dedup_ultrack_db.py --apply     # mutate data.db in place
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_DB = Path(
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk/"
    "pos00/2_nucleus/ultrack_workdir/data.db"
)
DEFAULT_RADIUS = 5.0


def _cluster_keepers(rows, radius):
    """rows: list of (id, y, x, node_prob, area) for one frame.

    Returns (keep_ids, drop_ids) for the frame.
    """
    n = len(rows)
    ids = [r[0] for r in rows]
    yx = np.array([(r[1], r[2]) for r in rows], dtype=float)
    prob = np.array([r[3] for r in rows], dtype=float)
    area = np.array([r[4] for r in rows], dtype=float)

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    if n > 1:
        for a, b in cKDTree(yx).query_pairs(r=radius):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    clusters = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    keep_ids, drop_ids = [], []
    for members in clusters.values():
        # keeper: highest node_prob, then smallest area (tighter mask)
        best = min(members, key=lambda i: (-prob[i], area[i]))
        keep_ids.append(ids[best])
        drop_ids.extend(ids[i] for i in members if i != best)
    return keep_ids, drop_ids


def main(db_path: Path, radius: float, apply: bool) -> None:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    by_t = defaultdict(list)
    for row in cur.execute("SELECT id, t, y, x, node_prob, area FROM nodes"):
        nid, t, y, x, prob, area = row
        by_t[t].append((nid, y, x, prob, area))

    drop_ids: list[int] = []
    n_total = 0
    for t in sorted(by_t):
        n_total += len(by_t[t])
        _, drop = _cluster_keepers(by_t[t], radius)
        drop_ids.extend(drop)

    n_drop = len(drop_ids)
    print(f"DB: {db_path}")
    print(f"radius={radius}px  frames={len(by_t)}")
    print(f"nodes: {n_total} -> {n_total - n_drop}  (drop {n_drop}, "
          f"{100 * n_drop / max(n_total,1):.1f}%)")

    n_ov = cur.execute("SELECT COUNT(*) FROM overlaps").fetchone()[0]
    n_lk = cur.execute("SELECT COUNT(*) FROM links").fetchone()[0]

    if not apply:
        print(f"overlaps: {n_ov} (cascade-delete on apply)")
        print(f"links:    {n_lk} (cascade-delete on apply)")
        print("\nDRY RUN — pass --apply to mutate.")
        con.close()
        return

    cur.execute("CREATE TEMP TABLE _del (id INTEGER PRIMARY KEY)")
    cur.executemany("INSERT INTO _del (id) VALUES (?)", [(i,) for i in drop_ids])

    cur.execute(
        "DELETE FROM overlaps WHERE node_id IN (SELECT id FROM _del) "
        "OR ancestor_id IN (SELECT id FROM _del)"
    )
    ov_del = cur.rowcount
    cur.execute(
        "DELETE FROM links WHERE source_id IN (SELECT id FROM _del) "
        "OR target_id IN (SELECT id FROM _del)"
    )
    lk_del = cur.rowcount
    cur.execute(
        "DELETE FROM cellflow_ultrack_source_nodes "
        "WHERE node_id IN (SELECT id FROM _del)"
    )
    cur.execute("DELETE FROM nodes WHERE id IN (SELECT id FROM _del)")
    nodes_del = cur.rowcount
    con.commit()

    print(f"deleted nodes:    {nodes_del}")
    print(f"overlaps: {n_ov} -> {n_ov - ov_del}  (deleted {ov_del})")
    print(f"links:    {n_lk} -> {n_lk - lk_del}  (deleted {lk_del})")
    print("\nNOTE: links were cascade-deleted; re-run linking before solving "
          "for accurate edge weights on the deduped node set.")

    cur.execute("VACUUM")
    con.commit()
    con.close()
    print("done (VACUUMed).")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    p.add_argument("--apply", action="store_true", help="mutate the DB in place")
    args = p.parse_args()
    main(args.db, args.radius, args.apply)
