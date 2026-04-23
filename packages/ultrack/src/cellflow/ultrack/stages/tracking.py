"""s03 — Ultrack tracking: segment, link, and solve as independent stages."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Generator, Sequence

import numpy as np
import tifffile

from cellflow.ultrack.config import TrackingConfig
from cellflow.ultrack.ingestion import (
    labels_batch_to_foreground_contours,
    load_hypothesis_labelmaps,
    prepare_hypothesis_labelmaps_for_ingestion,
    write_foreground_contours,
    write_hypothesis_labelmaps,
)
from cellflow.ultrack.pruning import prune_circularity_filtered_candidates
from cellflow.core.paths import stage_dir
from cellflow.core.protocol import StageProgress, ValidationResult


def _require_ultrack():
    """Lazy import guard — raises ImportError with a helpful message."""
    try:
        from ultrack.config import MainConfig  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ultrack is required for the tracking stage. "
            "Install it with: pip install cellflow-ultrack[tracking]"
        ) from exc


def _select_solver() -> str:
    try:
        import gurobipy  # noqa: F401
        return "GUROBI"
    except ImportError:
        return "CBC"


def _build_ultrack_config(
    cfg: TrackingConfig,
    working_dir: str | Path,
):
    """Translate our TrackingConfig into an Ultrack MainConfig."""
    from ultrack.config import MainConfig

    return MainConfig(
        data={
            "working_dir": str(working_dir),
        },
        segmentation={
            "min_area": cfg.min_area,
            "max_area": cfg.max_area,
            "min_frontier": cfg.min_frontier,
            "threshold": cfg.threshold,
            "ws_hierarchy": cfg.ws_hierarchy,
            "anisotropy_penalization": cfg.anisotropy_penalization,
            "n_workers": cfg.n_workers,
        },
        linking={
            "max_distance": cfg.max_distance,
            "max_neighbors": cfg.max_neighbors,
            "distance_weight": cfg.distance_weight,
            "n_workers": cfg.link_n_workers,
        },
        tracking={
            "solver_name": _select_solver(),
            "appear_weight": cfg.appear_weight,
            "disappear_weight": cfg.disappear_weight,
            "division_weight": cfg.division_weight,
            "link_function": cfg.link_function,
            "power": cfg.power,
            "bias": cfg.bias,
            "solution_gap": cfg.solution_gap,
            "time_limit": cfg.time_limit,
            "window_size": cfg.window_size if cfg.window_size > 0 else None,
        },
    )


def load_stack(path: str | Path) -> np.ndarray:
    """Load a TIFF stack (T, Z, Y, X) or (T, Y, X)."""
    return tifffile.imread(str(path)).astype(np.float32)


def export_tracked_labels(
    working_dir: str | Path,
    cfg: TrackingConfig,
    output_path: str | Path,
) -> None:
    """Export tracked segmentation labels as ``tracked_labels.tif``."""
    from ultrack.core.export.ctc import to_ctc

    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)
    output_path = Path(output_path)

    try:
        from ultrack.core.export.labels import to_labels  # type: ignore[import]

        labels = to_labels(ultrack_cfg)
        if hasattr(labels, "compute"):
            labels = labels.compute()
        tifffile.imwrite(str(output_path), labels.astype(np.uint32), compression="zlib")
        return
    except (ImportError, Exception):
        pass

    tmpdir = Path(tempfile.mkdtemp(prefix="ultrack_labels_"))
    try:
        to_ctc(tmpdir, ultrack_cfg, overwrite=True)
        mask_files = sorted(tmpdir.rglob("mask*.tif"))
        if not mask_files:
            mask_files = sorted(tmpdir.rglob("man_track*.tif"))
        if not mask_files:
            mask_files = sorted(tmpdir.rglob("*.tif"))
        if mask_files:
            frames = [tifffile.imread(str(f)) for f in mask_files]
            stacked = np.stack(frames, axis=0)
            tifffile.imwrite(str(output_path), stacked.astype(np.uint32), compression="zlib")
        else:
            raise RuntimeError("CTC export produced no mask files.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def get_labels_layer(working_dir: str | Path) -> np.ndarray:
    """Load ``tracked_labels.tif`` from *working_dir* for napari visualisation."""
    labels_path = Path(working_dir) / "tracked_labels.tif"
    if not labels_path.exists():
        raise FileNotFoundError(f"tracked_labels.tif not found in {working_dir}")
    return tifffile.imread(str(labels_path))


# ── Independent stage runners ─────────────────────────────────────────────────

def _to_zarr(arr: np.ndarray, store_path: Path) -> "zarr.Array":
    """Write a (T, …) numpy array to a time-chunked zarr DirectoryStore.

    Each timepoint is its own chunk so spawned worker processes can open the
    store by path and read only their slice — no large-array pickling across
    the spawn boundary.
    """
    import zarr

    chunks = (1,) + arr.shape[1:]
    z = zarr.open(
        str(store_path),
        mode="w",
        shape=arr.shape,
        dtype=arr.dtype,
        chunks=chunks,
    )
    z[:] = arr
    return z


def run_segmentation(
    foreground_path: str | Path,
    contours_path: str | Path,
    working_dir: str | Path,
    cfg: TrackingConfig,
    overwrite: bool = True,
) -> Generator[tuple[int, int, str], None, None]:
    """Run only the segmentation (add_nodes) step.

    Yields ``(step, total_steps, status_label)``.
    """
    total = 5
    fg_path = Path(foreground_path)
    ct_path = Path(contours_path)

    if not fg_path.exists():
        raise FileNotFoundError(f"Foreground stack not found: {fg_path}\nRun the Foreground stage first.")
    if not ct_path.exists():
        raise FileNotFoundError(f"Contours stack not found: {ct_path}\nRun the Contours stage first.")

    yield (0, total, "Loading stacks…")
    foreground = load_stack(fg_path)
    contours = load_stack(ct_path)

    wd = Path(working_dir)
    wd.mkdir(parents=True, exist_ok=True)
    ultrack_cfg = _build_ultrack_config(cfg, wd)

    from ultrack.core.database import clear_all_data
    from ultrack.core.segmentation.processing import segment

    if overwrite:
        yield (1, total, "Clearing existing segmentation from DB…")
        clear_all_data(ultrack_cfg.data_config.database_path)
    else:
        yield (1, total, "Skipping DB clear (overwrite=False)…")

    yield (2, total, "Running segmentation (add nodes)…")

    zarr_tmp: Path | None = None
    try:
        if cfg.n_workers > 1:
            # ultrack uses a spawn-based pool: worker processes cannot share
            # in-memory numpy arrays (they would be pickled in full for each
            # worker).  Converting to a file-backed zarr store lets each
            # spawned process open the store by path and read only its own
            # timepoint slice — no large-array serialisation.
            zarr_tmp = Path(tempfile.mkdtemp(prefix="cellflow_zarr_"))
            foreground = _to_zarr(foreground, zarr_tmp / "foreground.zarr")
            contours = _to_zarr(contours, zarr_tmp / "contours.zarr")

        try:
            segment(foreground, contours, ultrack_cfg, overwrite=overwrite)
        except ValueError as exc:
            if not overwrite and "Duplicated nodes" in str(exc):
                yield (total, total, "Segmentation already in DB, skipping.")
                return
            raise
    finally:
        if zarr_tmp is not None:
            shutil.rmtree(zarr_tmp, ignore_errors=True)

    yield (3, total, "Pruning low-circularity candidates…")
    pruned = prune_circularity_filtered_candidates(wd, cfg, n_workers=cfg.n_workers)

    if pruned:
        yield (4, total, f"Pruned {pruned} candidates and cleared links.")
    else:
        yield (4, total, "No low-circularity candidates to prune.")

    yield (total, total, "Segmentation done.")


def run_hypothesis_ingestion(
    labelmaps: Sequence[np.ndarray],
    working_dir: str | Path,
    cfg: TrackingConfig,
    *,
    overwrite: bool = True,
    stage_name: str = "tracking",
    source: str | None = None,
    smooth_sigma: float = 0.5,
) -> Generator[tuple[int, int, str], None, None]:
    """Ingest explicit label hypotheses into Ultrack.

    The hypotheses are persisted under ``labelmaps/`` together with a manifest
    so later workflow stages can inspect or replay the exact inputs that were
    used for segmentation.
    """
    total = 6
    wd = Path(working_dir)
    wd.mkdir(parents=True, exist_ok=True)
    ultrack_cfg = _build_ultrack_config(cfg, wd)

    from ultrack.core.database import clear_all_data
    from ultrack.core.segmentation.processing import segment

    if overwrite:
        yield (0, total, "Clearing existing segmentation from DB…")
        clear_all_data(ultrack_cfg.data_config.database_path)
    else:
        yield (0, total, "Skipping DB clear (overwrite=False)…")

    yield (1, total, "Writing hypothesis manifest…")
    write_hypothesis_labelmaps(wd, labelmaps, stage_name=stage_name, source=source)

    yield (2, total, "Preparing hypotheses for ingestion…")
    ingestion_labelmaps = prepare_hypothesis_labelmaps_for_ingestion(labelmaps)

    ingestion_shapes = {tuple(np.asarray(labels).shape) for labels in ingestion_labelmaps}
    if len(ingestion_shapes) != 1:
        raise ValueError(
            "All labelmaps must have the same shape for Ultrack ingestion; "
            f"got {sorted(ingestion_shapes)}"
        )

    yield (3, total, "Deriving segmentation inputs…")
    foreground, contours = labels_batch_to_foreground_contours(
        ingestion_labelmaps,
        smooth_sigma=smooth_sigma,
    )
    write_foreground_contours(wd, foreground, contours)

    zarr_tmp: Path | None = None
    try:
        if cfg.n_workers > 1:
            zarr_tmp = Path(tempfile.mkdtemp(prefix="cellflow_zarr_"))
            foreground = _to_zarr(foreground, zarr_tmp / "foreground.zarr")
            contours = _to_zarr(contours, zarr_tmp / "contours.zarr")

        yield (4, total, "Running segmentation (add nodes)…")
        try:
            segment(foreground, contours, ultrack_cfg, overwrite=overwrite)
        except ValueError as exc:
            if not overwrite and "Duplicated nodes" in str(exc):
                yield (total, total, "Segmentation already in DB, skipping.")
                return
            raise
    finally:
        if zarr_tmp is not None:
            shutil.rmtree(zarr_tmp, ignore_errors=True)

    yield (4, total, "Pruning low-circularity candidates…")
    pruned = prune_circularity_filtered_candidates(wd, cfg, n_workers=cfg.n_workers)

    if pruned:
        yield (5, total, f"Pruned {pruned} candidates and cleared links.")
    else:
        yield (5, total, "No low-circularity candidates to prune.")

    yield (total, total, "Segmentation done.")


def run_nucleus_ultrack(
    working_dir: str | Path,
    cfg: TrackingConfig,
    *,
    overwrite: bool = True,
) -> Generator[tuple[int, int, str], None, None]:
    """Run nucleus Ultrack from persisted label hypotheses."""
    wd = Path(working_dir)
    labelmaps, manifest = load_hypothesis_labelmaps(wd)
    source = manifest.get("source") if isinstance(manifest, dict) else None
    yield from run_hypothesis_ingestion(
        labelmaps,
        wd,
        cfg,
        overwrite=overwrite,
        stage_name="nucleus_ultrack",
        source=source if isinstance(source, str) else None,
    )


def run_linking(
    working_dir: str | Path,
    cfg: TrackingConfig,
    overwrite: bool = True,
) -> Generator[tuple[int, int, str], None, None]:
    """Run only the linking (add_edges) step.

    Yields ``(step, total_steps, status_label)``.
    """
    if cfg.linking_mode == "iou":
        from cellflow.ultrack.linking import run_iou_linking

        yield from run_iou_linking(working_dir, cfg, overwrite=overwrite)
        return
    if cfg.linking_mode != "default":
        raise ValueError(
            f"Unknown linking_mode={cfg.linking_mode!r}; expected 'default' or 'iou'."
        )

    total = 3
    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)

    from ultrack.core.linking.processing import link
    from ultrack.core.linking.utils import clear_linking_data

    if overwrite:
        yield (0, total, "Clearing existing links from DB…")
        clear_linking_data(ultrack_cfg.data_config.database_path)
    else:
        yield (0, total, "Skipping DB clear (overwrite=False)…")

    yield (1, total, "Running linking (add edges)…")
    link(ultrack_cfg)

    yield (total, total, "Linking done.")


def run_solve(
    working_dir: str | Path,
    cfg: TrackingConfig,
    overwrite: bool = True,
) -> Generator[tuple[int, int, str], None, None]:
    """Run only the solve (ILP) step and export results.

    Yields ``(step, total_steps, status_label)``.
    """
    from ultrack.core.solve.processing import solve
    from ultrack.core.solve.sqltracking import SQLTracking
    from ultrack.core.export.tracks_layer import to_tracks_layer
    from cellflow.ultrack.stages.project2d import export_nuclear_labels_2d

    total = 6
    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)

    if overwrite:
        yield (0, total, "Clearing existing solution from DB…")
        SQLTracking.clear_solution_from_database(ultrack_cfg.data_config.database_path)
    else:
        yield (0, total, "Skipping DB clear (overwrite=False)…")

    yield (1, total, "Running ILP solve…")
    solve(ultrack_cfg)

    yield (2, total, "Exporting tracks CSV…")
    tracks_df, _ = to_tracks_layer(ultrack_cfg)
    tracks_df.to_csv(str(wd / "tracks.csv"), index=True)

    yield (3, total, "Exporting tracked labels…")
    export_tracked_labels(wd, cfg, wd / "tracked_labels.tif")

    yield (4, total, "Projecting tracked labels to 2D…")
    export_nuclear_labels_2d(wd / "tracked_labels.tif", wd / "nuclear_labels_2d.tif")

    yield (total, total, "Solve done.")


def run(
    foreground_path: str | Path,
    contours_path: str | Path,
    working_dir: str | Path,
    cfg: TrackingConfig,
) -> Generator[tuple[int, int, str], None, None]:
    """Run the full Ultrack pipeline (segment → link → solve).

    Respects ``cfg.overwrite_segmentation``, ``cfg.overwrite_linking``,
    and ``cfg.overwrite_solve``.

    Yields ``(step, total_steps, status_label)``.
    """
    total = 12

    # Segmentation
    for step, sub_total, label in run_segmentation(
        foreground_path, contours_path, working_dir, cfg,
        overwrite=cfg.overwrite_segmentation,
    ):
        yield (int(step / max(sub_total, 1) * 4), total, f"[Seg] {label}")

    # Linking
    for step, sub_total, label in run_linking(
        working_dir, cfg, overwrite=cfg.overwrite_linking,
    ):
        yield (4 + int(step / max(sub_total, 1) * 3), total, f"[Link] {label}")

    # Solve + export
    for step, sub_total, label in run_solve(
        working_dir, cfg, overwrite=cfg.overwrite_solve,
    ):
        yield (7 + int(step / max(sub_total, 1) * 5), total, f"[Solve] {label}")

    yield (total, total, "Done")


def export_ctc(
    working_dir: str | Path,
    output_dir: str | Path,
    cfg: TrackingConfig,
) -> None:
    """Export tracking results to CTC format."""
    from ultrack.core.export.ctc import to_ctc

    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)
    to_ctc(Path(output_dir), ultrack_cfg, overwrite=True)


def get_tracks_layer(
    working_dir: str | Path,
    cfg: TrackingConfig,
):
    """Load tracks layer data for napari visualization."""
    from ultrack.core.export.tracks_layer import to_tracks_layer

    wd = Path(working_dir)
    ultrack_cfg = _build_ultrack_config(cfg, wd)
    tracks_df, graph = to_tracks_layer(ultrack_cfg)
    spatial_cols = [c for c in tracks_df.columns if c in {"track_id", "t", "z", "y", "x"}]
    return tracks_df[spatial_cols], graph


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="s03 — run Ultrack tracking pipeline from the command line",
    )
    parser.add_argument("--stage", default="all",
                        choices=["all", "segmentation", "linking", "solve"],
                        help="Which stage to run (default: all)")
    parser.add_argument("--foreground", default=None, help="Path to foreground.tif stack")
    parser.add_argument("--contours", default=None, help="Path to contours.tif stack")
    parser.add_argument("--working-dir", required=True, help="Ultrack working/output directory")
    parser.add_argument("--config", default=None, help="Path to JSON file with TrackingConfig fields")
    parser.add_argument("--overwrite-segmentation", action="store_true", default=True)
    parser.add_argument("--overwrite-linking", action="store_true", default=True)
    parser.add_argument("--overwrite-solve", action="store_true", default=True)
    args = parser.parse_args()

    # Validate required args for each stage
    if args.stage in ("all", "segmentation"):
        if not args.foreground or not args.contours:
            parser.error("--foreground and --contours are required for --stage all/segmentation")

    cfg_dict: dict = {}
    if args.config:
        cfg_dict = json.loads(Path(args.config).read_text())
    cfg_dict.setdefault("overwrite_segmentation", args.overwrite_segmentation)
    cfg_dict.setdefault("overwrite_linking", args.overwrite_linking)
    cfg_dict.setdefault("overwrite_solve", args.overwrite_solve)
    cfg = TrackingConfig(**cfg_dict)

    if args.stage == "all":
        gen = run(args.foreground, args.contours, args.working_dir, cfg)
    elif args.stage == "segmentation":
        gen = run_segmentation(args.foreground, args.contours, args.working_dir, cfg, overwrite=cfg.overwrite_segmentation)
    elif args.stage == "linking":
        gen = run_linking(args.working_dir, cfg, overwrite=cfg.overwrite_linking)
    elif args.stage == "solve":
        gen = run_solve(args.working_dir, cfg, overwrite=cfg.overwrite_solve)
    else:
        parser.error(f"Unknown stage: {args.stage}")

    for step, total, label in gen:
        print(f"[{step}/{total}] {label}", flush=True)

    sys.exit(0)


# ── StageProtocol wrapper ────────────────────────────────────────────────────


class _TrackingStageClass:
    name = "nucleus_ultrack"
    display_name = "Nucleus Ultrack"

    def __init__(self):
        self.config = TrackingConfig()

    def run(self, working_dir, cfg: TrackingConfig = None, overwrite: bool = True):
        from cellflow.core.logging import StageLogger

        cfg = cfg or self.config
        wd = Path(working_dir)
        log_file = wd.parent / "pipeline.log"
        log = StageLogger(log_file, self.name)
        with log:
            for progress in run_nucleus_ultrack(working_dir=working_dir, cfg=cfg, overwrite=overwrite):
                yield StageProgress(*progress)
            # Keep the stage-local outputs authoritative; later stages read them
            # directly from 2_nucleus_ultrack/.

    def validate_inputs(self, schema, root_dir, pos) -> ValidationResult:
        from cellflow.core.validation import validate_inputs

        stage_path = stage_dir(root_dir, pos, "nucleus_ultrack")
        manifest = stage_path / "hypotheses_manifest.json"
        labelmap_dir = stage_path / "labelmaps"
        labelmaps = sorted(labelmap_dir.glob("labelmap_*.tif")) if labelmap_dir.exists() else []
        if not manifest.exists() and not labelmaps:
            return ValidationResult(
                ok=False,
                errors=[f"No hypotheses_manifest.json or labelmaps/labelmap_*.tif in {stage_path}"],
            )
        return validate_inputs([manifest] if manifest.exists() else labelmaps)

    def is_complete(self, root_dir, pos) -> bool:
        d = stage_dir(root_dir, pos, "nucleus_ultrack")
        return (d / "tracked_labels.tif").exists()


NucleusUltrackStage = _TrackingStageClass()
TrackingStage = NucleusUltrackStage
