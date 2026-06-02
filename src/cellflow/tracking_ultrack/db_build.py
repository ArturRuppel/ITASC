"""Shared Ultrack database construction pipeline."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from cellflow.tracking_ultrack.config import TrackingConfig
from cellflow.tracking_ultrack.corrections import (
    Correction,
    annotate_anchor_tail_links,
    apply_corrections_to_database,
    corrections_from_validated_tracks,
    ensure_anchor_incident_links,
    inject_unmatched_anchor_nodes,
)
from cellflow.tracking_ultrack.ingest import _build_ultrack_config
from cellflow.tracking_ultrack.linking import run_linking
from cellflow.tracking_ultrack.seed_prior import write_seed_prior_node_probs


@dataclass(frozen=True)
class UltrackDatabaseBuildReport:
    """Result of candidate-building (segmentation + linking). No annotation state."""
    pass


@dataclass(frozen=True)
class AtomUnionDatabaseBuildReport:
    """Result of atom-union candidate-building + linking (stage ②)."""
    total_nodes: int = 0
    total_overlaps: int = 0
    n_frames: int = 0


@dataclass(frozen=True)
class AnnotateAndScoreReport:
    fake_nodes: int = 0
    anchor_nodes: int = 0
    anchor_links: int = 0
    scored_nodes: int = 0
    seed_nodes: int = 0
    anchor_incident_links_inserted: int = 0
    anchor_tail_links_annotated: int = 0
    injected_homemade_anchors: int = 0


def _reset_annotations(working_dir: str | Path) -> None:
    """Clear all NodeDB.node_annot and LinkDB.annotation back to UNKNOWN."""
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session
    from ultrack.core.database import LinkDB, NodeDB, VarAnnotation

    engine = sqla.create_engine(f"sqlite:///{Path(working_dir) / 'data.db'}")
    with Session(engine) as session:
        session.query(NodeDB).update(
            {NodeDB.node_annot: VarAnnotation.UNKNOWN},
            synchronize_session=False,
        )
        session.query(LinkDB).update(
            {LinkDB.annotation: VarAnnotation.UNKNOWN},
            synchronize_session=False,
        )
        session.commit()


def apply_annotations_and_score(
    working_dir: str | Path,
    cfg: TrackingConfig,
    *,
    score_signal_path: str | Path,
    corrections: list[Correction] | None = None,
    validated_tracks: dict[int, set[int]] | None = None,
    tracked_labels: np.ndarray | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> AnnotateAndScoreReport:
    """Reset annotations on an existing ``data.db``, apply corrections, and rescore.

    Runs after candidate-building (segmentation + linking) and before solve.
    The candidate set itself is left untouched; only ``node_annot``,
    ``link.annotation`` and ``node_prob`` change. Safe to call repeatedly when
    the user toggles validations or anchors.
    """
    if corrections is None and validated_tracks:
        if tracked_labels is None:
            raise ValueError(
                "validated_tracks requires tracked_labels for centroid derivation."
            )
        corrections = corrections_from_validated_tracks(
            validated_tracks,
            np.asarray(tracked_labels, dtype=np.uint32),
        )
    corrections = list(corrections or [])

    _notify(progress_cb, "Resetting prior annotations...")
    _reset_annotations(working_dir)

    fake_nodes = anchor_nodes = anchor_links = 0
    injected_homemade_anchors = 0
    if corrections:
        _notify(progress_cb, "Applying node annotations...")
        pre = apply_corrections_to_database(
            working_dir, corrections, cfg,
            annotate_anchor_links=False,
            tracked_labels=tracked_labels,
        )
        fake_nodes = int(pre.fake_nodes)
        anchor_nodes = int(pre.anchor_nodes)
        _notify(
            progress_cb,
            f"Marked {fake_nodes} FAKE node(s) and {anchor_nodes} anchor node(s).",
        )

        if pre.unmatched_anchors and tracked_labels is not None:
            # Some anchor corrections had no NodeDB candidate — the user is
            # anchoring a manually-drawn cell. Inject synthetic REAL nodes so
            # the ILP is forced to include them.
            _notify(
                progress_cb,
                f"Injecting {len(pre.unmatched_anchors)} homemade anchor node(s) into DB...",
            )
            inj = inject_unmatched_anchor_nodes(
                working_dir, pre.unmatched_anchors, tracked_labels, cfg,
            )
            injected_homemade_anchors = inj.injected
            _notify(
                progress_cb,
                f"Injected {inj.injected} homemade anchor node(s) "
                f"({inj.skipped_no_mask} skipped: no mask in tracked_labels).",
            )
        elif pre.unmatched_anchors:
            _notify(
                progress_cb,
                f"Warning: {len(pre.unmatched_anchors)} anchor correction(s) matched no "
                f"NodeDB candidate and tracked_labels was not provided — these anchors "
                f"will not affect the ILP solve.",
            )

    _notify(progress_cb, "Scoring node probabilities...")
    score = write_seed_prior_node_probs(working_dir, score_signal_path, cfg)
    scored_nodes = int(getattr(score, "scored", 0))
    seed_nodes = int(getattr(score, "seeds", 0))
    _notify(progress_cb, f"Scored {scored_nodes} node(s) using {seed_nodes} seed node(s).")

    if corrections:
        # Second pass: re-mark REAL anchor nodes (now including any injected
        # homemade nodes that were inserted above) and add link annotations
        # between consecutive REAL anchor nodes.
        _notify(progress_cb, "Applying link annotations...")
        post = apply_corrections_to_database(
            working_dir, corrections, cfg,
            annotate_anchor_links=True,
            tracked_labels=tracked_labels,
        )
        anchor_links = int(post.anchor_links)
        _notify(progress_cb, f"Marked {anchor_links} anchor link(s).")

    _notify(progress_cb, "Filling anchor-incident links...")
    incident = ensure_anchor_incident_links(working_dir, cfg)
    anchor_incident_links_inserted = int(incident.inserted)
    _notify(
        progress_cb,
        f"Inserted {anchor_incident_links_inserted} anchor-incident link(s) "
        f"across {int(incident.anchors_processed)} anchor node(s).",
    )

    anchor_tail_links_annotated = 0
    if corrections:
        _notify(progress_cb, "Annotating anchor tail continuation links...")
        tail = annotate_anchor_tail_links(
            working_dir,
            corrections,
            cfg,
            tracked_labels=tracked_labels,
        )
        anchor_tail_links_annotated = int(tail.annotated)
        _notify(
            progress_cb,
            f"Annotated {anchor_tail_links_annotated} anchor tail link(s).",
        )

    return AnnotateAndScoreReport(
        fake_nodes=fake_nodes,
        anchor_nodes=anchor_nodes,
        anchor_links=anchor_links,
        scored_nodes=scored_nodes,
        seed_nodes=seed_nodes,
        anchor_incident_links_inserted=anchor_incident_links_inserted,
        anchor_tail_links_annotated=anchor_tail_links_annotated,
        injected_homemade_anchors=injected_homemade_anchors,
    )


def annotate_database_from_corrections(
    working_dir: str | Path,
    cfg: TrackingConfig,
    *,
    score_signal_path: str | Path,
    corrections: list[Correction] | None = None,
    validated_tracks: dict[int, set[int]] | None = None,
    tracked_labels: np.ndarray | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> AnnotateAndScoreReport:
    """Apply correction-derived annotations to ``data.db`` and rescore nodes."""
    return apply_annotations_and_score(
        working_dir=working_dir,
        cfg=cfg,
        score_signal_path=score_signal_path,
        corrections=corrections,
        validated_tracks=validated_tracks,
        tracked_labels=tracked_labels,
        progress_cb=progress_cb,
    )


def _notify(progress_cb: Callable[[str], None] | None, message: str) -> None:
    if progress_cb is not None:
        progress_cb(message)


def build_atom_union_database(
    atoms_path: str | Path,
    working_dir: str | Path,
    cfg: TrackingConfig,
    progress_cb: Callable[[str], None] | None = None,
) -> AtomUnionDatabaseBuildReport:
    """Build candidate ``data.db`` from ``atoms.tif`` via connected-union enumeration.

    Reads ``atoms.tif`` (written by stage ①), builds NodeDB + OverlapDB rows for
    every connected union of ≤ ``cfg.atom_union_max_atoms`` atoms whose total area is
    ≤ ``cfg.atom_union_max_area``, then runs linking. Warns if the embedded fingerprint
    differs from the current atom params in ``cfg``.
    """
    import pickle

    import sqlalchemy as sqla
    from scipy import ndimage as ndi
    from sqlalchemy.orm import Session
    from ultrack.core.database import Base, NodeDB, OverlapDB, clear_all_data
    from ultrack.core.segmentation.node import Node

    from cellflow.tracking_ultrack.atoms import (
        atom_adjacency,
        enum_connected_unions,
    )

    atoms_path = Path(atoms_path)
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    _notify(progress_cb, "Reading atoms.tif...")
    atoms_stack = tifffile.imread(str(atoms_path)).astype(np.int32)
    if atoms_stack.ndim == 2:
        atoms_stack = atoms_stack[np.newaxis]

    ultrack_cfg = _build_ultrack_config(cfg, working_dir)
    db_path = ultrack_cfg.data_config.database_path
    clear_all_data(db_path)
    engine = sqla.create_engine(db_path)
    Base.metadata.create_all(engine)
    ultrack_cfg.data_config.metadata_add({"shape": list(atoms_stack.shape)})

    n_frames = atoms_stack.shape[0]
    max_seg = cfg.max_segments_per_time
    max_atoms = cfg.atom_union_max_atoms
    max_area = cfg.atom_union_max_area
    min_area = cfg.min_area
    total_nodes = total_overlaps = 0

    for t in range(n_frames):
        _notify(progress_cb, f"Building candidates frame {t + 1}/{n_frames}...")
        frame_atoms = atoms_stack[t]
        n_atoms = int(frame_atoms.max())
        if n_atoms == 0:
            continue

        slices = ndi.find_objects(frame_atoms)
        ids, counts = np.unique(frame_atoms, return_counts=True)
        areas = {int(i): int(c) for i, c in zip(ids, counts) if i != 0}
        adj = atom_adjacency(frame_atoms)
        unions = enum_connected_unions(adj, areas, max_atoms, max_area)
        unions = [
            u for u in unions
            if sum(areas[m] for m in u) >= min_area or len(u) > 1
        ]

        nodes: list[NodeDB] = []
        atom_to_node_ids: dict[int, list[int]] = {a: [] for a in areas}
        index = 1
        for members in unions:
            mlist = sorted(members)
            ys0 = min(slices[m - 1][0].start for m in mlist)
            xs0 = min(slices[m - 1][1].start for m in mlist)
            ys1 = max(slices[m - 1][0].stop for m in mlist)
            xs1 = max(slices[m - 1][1].stop for m in mlist)
            crop = frame_atoms[ys0:ys1, xs0:xs1]
            mask = np.isin(crop, mlist)
            bbox = np.array([ys0, xs0, ys1, xs1])
            node = Node.from_mask(t, mask.astype(bool), bbox=bbox)
            nid = index + (t + 1) * max_seg
            node.id = nid
            node.time = t
            centroid = node.centroid
            z, y, x = (0, *centroid) if len(centroid) == 2 else centroid
            nodes.append(NodeDB(
                id=nid, t_node_id=index, t_hier_id=1, t=t,
                z=int(z), y=int(y), x=int(x), area=int(node.area),
                frontier=-1.0, height=float(len(mlist)),
                pickle=pickle.dumps(node),
            ))
            for a in mlist:
                atom_to_node_ids[a].append(nid)
            index += 1

        # Two candidates overlap iff they share an atom (atoms are disjoint).
        seen_pairs: set[tuple[int, int]] = set()
        overlaps: list[OverlapDB] = []
        for cand_ids in atom_to_node_ids.values():
            for i in range(len(cand_ids)):
                for j in range(i + 1, len(cand_ids)):
                    a, b = cand_ids[i], cand_ids[j]
                    key = (a, b) if a < b else (b, a)
                    if key not in seen_pairs:
                        seen_pairs.add(key)
                        overlaps.append(OverlapDB(node_id=key[0], ancestor_id=key[1]))

        with Session(engine) as session:
            session.add_all(nodes)
            session.add_all(overlaps)
            session.commit()
        engine.dispose()

        total_nodes += len(nodes)
        total_overlaps += len(overlaps)

    _notify(
        progress_cb,
        f"Wrote {total_nodes} candidate nodes, {total_overlaps} overlap rows.",
    )
    _notify(progress_cb, "Linking candidates...")
    for step, total, label in run_linking(working_dir, cfg):
        _notify(progress_cb, f"[link {step}/{total}] {label}")

    return AtomUnionDatabaseBuildReport(
        total_nodes=total_nodes,
        total_overlaps=total_overlaps,
        n_frames=n_frames,
    )


def _load_ultrack_inputs(
    contour_maps_path: str | Path,
    foreground_masks_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    contours = np.asarray(tifffile.imread(str(contour_maps_path)), dtype=np.float32)
    foreground = np.asarray(tifffile.imread(str(foreground_masks_path)), dtype=np.float32)
    if contours.ndim == 4 and contours.shape[1] == 1:
        contours = contours[:, 0]
    if foreground.ndim == 4 and foreground.shape[1] == 1:
        foreground = foreground[:, 0]
    return contours, foreground


def _run_ultrack_segment(
    foreground: np.ndarray,
    contours: np.ndarray,
    ultrack_cfg,
    cfg: TrackingConfig,
) -> None:
    try:
        from ultrack.core.segmentation.processing import segment as ultrack_segment
    except ImportError as exc:
        raise ImportError(
            "ultrack must be installed (conda env cellflow) to build data.db"
        ) from exc

    ultrack_segment(
        foreground,
        contours,
        ultrack_cfg,
        max_segments_per_time=cfg.max_segments_per_time,
        overwrite=True,
    )


def build_ultrack_database(
    contour_maps_path: str | Path,
    foreground_masks_path: str | Path,
    working_dir: str | Path,
    cfg: TrackingConfig,
    progress_cb: Callable[[str], None] | None = None,
) -> UltrackDatabaseBuildReport:
    """Build candidate ``data.db`` from canonical Ultrack segmentation + linking.

    Produces NodeDB / LinkDB / OverlapDB rows with all annotations UNKNOWN and
    no node-prob scores. Pair with ``apply_annotations_and_score`` before
    ``run_solve`` to ingest validations/anchors.
    """
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    _notify(progress_cb, "Loading contour maps and foreground masks...")
    contours, foreground = _load_ultrack_inputs(contour_maps_path, foreground_masks_path)
    ultrack_cfg = _build_ultrack_config(cfg, working_dir)

    _notify(progress_cb, "Segmenting candidates (ultrack hierarchy)...")
    _run_ultrack_segment(foreground, contours, ultrack_cfg, cfg)

    _notify(progress_cb, "Linking candidates...")
    for step, total, label in run_linking(working_dir, cfg):
        _notify(progress_cb, f"[link {step}/{total}] {label}")

    return UltrackDatabaseBuildReport()
