#!/usr/bin/env python
"""Phase 0 probe: verify ingest → link → solve round-trip with synthetic data.

Creates a 2-frame synthetic labelmap (2 cells per frame), ingests it as if it
were two hypothesis partitions, then links and solves. Prints NodeDB/LinkDB
contents to confirm the ID scheme, mask round-trip, and solver output.

Run inside the cellflow conda env:
    python scripts/probe_ultrack_db.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import h5py

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.ingest import ingest_hypotheses_to_db

# ---------------------------------------------------------------------------
# Build a minimal synthetic hypotheses.h5
# ---------------------------------------------------------------------------
FRAME_SHAPE = (1, 64, 64)   # (Z=1, Y, X)


def _make_synthetic_h5(path: Path) -> None:
    """Two frames, two partitions each.  Each partition has 2 non-overlapping cells.

    p=0: two large cells (top-left, top-right quadrants)
    p=1: two smaller cells (slightly shifted, overlapping p=0 cells)
    """
    def _label_frame(half_height: int, offset: tuple[int, int] = (0, 0)) -> np.ndarray:
        lm = np.zeros(FRAME_SHAPE, dtype=np.uint32)
        r0, c0 = offset
        lm[0, r0:r0 + half_height, c0:c0 + 30] = 1
        lm[0, r0:r0 + half_height, c0 + 33:c0 + 63] = 2
        return lm

    with h5py.File(path, "w") as f:
        f.attrs["version"] = 2
        f.attrs["stage"] = "nucleus_hypotheses"
        f.attrs["layout"] = "hypotheses/t{t:03d}/p{p:03d}/labels"
        f.attrs["n_t"] = 2
        f.attrs["n_p"] = 2

        for t in range(2):
            t_grp = f.require_group(f"hypotheses/t{t:03d}")
            # p=0: cells in top half
            p0 = t_grp.require_group("p000")
            lm0 = _label_frame(28, offset=(2, 1))
            p0.create_dataset("labels", data=lm0)
            p0.attrs["parameter_index"] = 0
            p0.attrs["parameter_json"] = '{"method": "watershed", "threshold_pct": 30.0}'
            p0.attrs["method"] = "watershed"

            # p=1: cells slightly smaller / shifted — overlap with p=0
            p1 = t_grp.require_group("p001")
            lm1 = _label_frame(20, offset=(4, 2))
            p1.create_dataset("labels", data=lm1)
            p1.attrs["parameter_index"] = 1
            p1.attrs["parameter_json"] = '{"method": "watershed", "threshold_pct": 40.0}'
            p1.attrs["method"] = "watershed"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="probe_ultrack_") as tmpdir:
        tmp = Path(tmpdir)
        h5_path = tmp / "hypotheses.h5"
        _make_synthetic_h5(h5_path)
        print(f"Synthetic HDF5: {h5_path}")

        cfg = TrackingConfig(min_area=10, max_area=10_000)
        working_dir = tmp / "ultrack"

        # ---- Ingest --------------------------------------------------------
        print("\n=== Ingesting hypotheses ===")
        ingest_hypotheses_to_db(h5_path, working_dir, cfg, overwrite=True)

        # ---- Inspect NodeDB ------------------------------------------------
        import sqlalchemy as sqla
        from sqlalchemy.orm import Session
        from ultrack.core.database import NodeDB, OverlapDB, LinkDB

        engine = sqla.create_engine(f"sqlite:///{working_dir}/data.db")
        with Session(engine) as sess:
            nodes = list(sess.query(NodeDB).all())
            overlaps = list(sess.query(OverlapDB).all())
            print(f"\nNodeDB: {len(nodes)} rows")
            for n in nodes:
                node = n.pickle  # MaybePickleType auto-deserializes
                print(f"  t={n.t} id={n.id} p={n.t_hier_id} area={n.area} "
                      f"y={n.y:.1f} x={n.x:.1f} mask.shape={node.mask.shape}")

            print(f"\nOverlapDB: {len(overlaps)} rows")
            for ov in overlaps:
                print(f"  node_id={ov.node_id} ancestor_id={ov.ancestor_id}")

        # ---- Link ----------------------------------------------------------
        print("\n=== Linking ===")
        from cellflow.tracking_ultrack.linking import run_linking
        for step, total, label in run_linking(working_dir, cfg):
            print(f"  [{step}/{total}] {label}")

        with Session(engine) as sess:
            links = list(sess.query(LinkDB).all())
            print(f"\nLinkDB: {len(links)} rows")
            for lk in links[:10]:
                print(f"  src={lk.source_id} tgt={lk.target_id} w={lk.weight:.4f}")

        # ---- Solve ---------------------------------------------------------
        print("\n=== Solving ===")
        from cellflow.tracking_ultrack.solve import run_solve
        for step, total, label in run_solve(working_dir, cfg):
            print(f"  [{step}/{total}] {label}")

        with Session(engine) as sess:
            selected = list(sess.query(NodeDB).where(NodeDB.selected == True).all())
            print(f"\nSelected nodes: {len(selected)}")
            for n in selected:
                print(f"  t={n.t} id={n.id} parent_id={n.parent_id} area={n.area}")

        print("\n=== Probe complete ===")


if __name__ == "__main__":
    main()
