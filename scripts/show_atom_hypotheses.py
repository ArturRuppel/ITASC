#!/usr/bin/env python
"""Show every candidate hypothesis for one frame as a scrubable napari stack.

Reads the atom-based ultrack DB (built by experiment_atoms_to_db.py), unpickles
each NodeDB candidate for the chosen frame, and paints it into its own slice of
a (n_candidates, H, W) stack. Scrub the first slider to step through every
hypothesis one at a time; the foreground map sits behind each slice.

Slices are ordered atoms-first then unions, and coloured by atom-count
(1 = single atom, 2 = pair, 3 = triple, …) so over-segmentation merges stand out.

Usage
-----
    python scripts/show_atom_hypotheses.py
    python scripts/show_atom_hypotheses.py --frame 0 --workdir <dir>
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import tifffile

from experiment_residual_atoms import FG_PATH, CONTOUR_PATH, DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--workdir", type=str,
                    default=str(DATA / "2_nucleus" / "atoms_ultrack_workdir"))
    args = ap.parse_args()

    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import NodeDB

    db_path = Path(args.workdir) / "data.db"
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path} — run experiment_atoms_to_db.py first")

    fg = tifffile.imread(FG_PATH).astype(np.float32)[args.frame]
    contour = tifffile.imread(CONTOUR_PATH).astype(np.float32)[args.frame]
    H, W = fg.shape

    engine = sqla.create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        rows = (
            session.query(NodeDB.id, NodeDB.height, NodeDB.area, NodeDB.pickle)
            .filter(NodeDB.t == args.frame)
            .order_by(NodeDB.height, NodeDB.area, NodeDB.id)
            .all()
        )
    engine.dispose()

    if not rows:
        raise SystemExit(f"No candidates at frame {args.frame}.")

    n = len(rows)
    n_atoms = np.array([int(r[1]) for r in rows])
    print(f"frame {args.frame}: {n} candidate hypotheses")
    for k in sorted(set(n_atoms.tolist())):
        print(f"  {int((n_atoms == k).sum()):>5}  with {k} atom(s)")

    stack = np.zeros((n, H, W), dtype=np.uint16)
    for i, (_id, height, _area, blob) in enumerate(rows):
        # ultrack's NodeDB.pickle column auto-deserializes to a Node on read.
        node = pickle.loads(blob) if isinstance(blob, (bytes, bytearray)) else blob
        mask = node.mask.astype(bool)
        bbox = np.asarray(node.bbox)
        y0, x0, y1, x1 = bbox
        sub = stack[i, y0:y1, x0:x1]
        sub[mask] = int(height)  # colour by atom-count

    import napari
    v = napari.Viewer()
    v.add_image(fg, name="fg", colormap="gray")
    v.add_image(contour, name="contour", colormap="inferno",
                blending="additive", visible=False)
    v.add_labels(stack, name="hypotheses (scrub axis 0)", opacity=0.6)
    v.dims.set_point(0, 0)
    v.text_overlay.visible = True
    v.text_overlay.text = f"frame {args.frame}: {n} hypotheses — scrub the slider"
    napari.run()


if __name__ == "__main__":
    main()
