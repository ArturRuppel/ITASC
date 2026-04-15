"""s02 — Flow-guided watershed cell segmentation.

Expands nuclear labels from ultrack using cellpose flow field from s1b
to guide and scale expansion velocity. Produces cell segmentation stack.

Inputs (per position)
---------------------
  2_ultrack/tracked_labels_proj2d_corrected.tif  (T, H, W)    uint32  — nuclear labels (2D projection)
  1b_cellpose_cell/cell_dp.tif                   (T, 2, H, W) float32 — flow field
  1b_cellpose_cell/cell_prob.tif                 (T, H, W)    float32 — cellpose probability

Outputs (per position)
----------------------
  3_cell_segmentation/
    run_params.json
    cell_labels_raw.tif              (T, H, W)    int32  — raw segmentation (before postprocessing)
    cell_labels.tif                  (T, H, W)    int32  — final segmentation (after postprocessing)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Generator

import numpy as np
import tifffile

from cellflow.cellpose.processing.flow_watershed import flow_guided_watershed
from cellflow.cellpose.config import FlowWatershedConfig
from cellflow.core.paths import stage_dir
from cellflow.core.protocol import StageProgress, ValidationResult


def load_tracked_labels(root_dir: str | Path, pos: int) -> np.ndarray | None:
    """Load tracked nuclear labels (2D projection) from 3_tracking."""
    tracking_path = stage_dir(root_dir, pos, "tracking") / "tracked_labels_proj2d_corrected.tif"
    if not tracking_path.exists():
        print(f"[error] Tracked labels not found: {tracking_path}", file=sys.stderr)
        return None
    return tifffile.imread(str(tracking_path)).astype(np.int32)


def apply_postprocessing(
    raw_labels_stack: np.ndarray,
    cellpose_prob_stack: np.ndarray | None = None,
    opening_radius: int = 1,
    closing_radius: int = 1,
    boundary_smoothness: float = 0.5,
    fill_holes_threshold: float = 0.5,
) -> np.ndarray:
    """Apply post-processing to flow-watershed labels.

    Parameters
    ----------
    raw_labels_stack : np.ndarray
        Raw labels from flow_guided_watershed (T, H, W)
    cellpose_prob_stack : np.ndarray, optional
        Cellpose probability stack (T, H, W)
    opening_radius : int
        Morphological opening radius
    closing_radius : int
        Morphological closing radius
    boundary_smoothness : float
        Boundary smoothing factor (0-1)
    fill_holes_threshold : float
        Cellpose probability threshold for hole-filling (0-1)

    Returns
    -------
    np.ndarray
        Post-processed labels (T, H, W)
    """
    from cellflow.cellpose.processing.flow_watershed_postproc import postprocess_flow_watershed

    T = raw_labels_stack.shape[0]
    processed_stack = []

    for t in range(T):
        raw_t = raw_labels_stack[t]
        prob_t = cellpose_prob_stack[t] if cellpose_prob_stack is not None else None

        processed = postprocess_flow_watershed(
            raw_t,
            cellpose_prob=prob_t,
            opening_radius=opening_radius,
            closing_radius=closing_radius,
            boundary_smoothness=boundary_smoothness,
            fill_holes_threshold=fill_holes_threshold,
        )
        processed_stack.append(processed)

    return np.stack(processed_stack, axis=0).astype(np.int32)


def load_cellpose_data(root_dir: str | Path, pos: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load cellpose flow and probability from s1b.

    Returns raw arrays with shapes as stored on disk:
    - flow: (T, 2, H, W) or (2, H, W)
    - prob: (T, H, W) or (H, W)
    """
    cell_dir = stage_dir(root_dir, pos, "cellpose_cell")

    # Load flow field
    flow_path = cell_dir / "cell_dp.tif"
    if not flow_path.exists():
        print(f"[error] Flow field not found: {flow_path}", file=sys.stderr)
        return None, None

    flow = tifffile.imread(str(flow_path)).astype(np.float32)
    print(f"  flow shape={flow.shape}", flush=True)

    # Load probability field
    prob_path = cell_dir / "cell_prob.tif"
    prob = None
    if prob_path.exists():
        prob = tifffile.imread(str(prob_path)).astype(np.float32)
        print(f"  prob shape={prob.shape}", flush=True)

    return flow, prob


def run(
    root_dir: str | Path,
    pos: int,
    flow_scale: float = 1.0,
    cellpose_prob_threshold: float = 0.0,
    flow_smoothing_sigma: float = 0.0,
    max_iterations: int = 50,
    uniform_growth_rate: float = 0.2,
    opening_radius: int = 1,
    closing_radius: int = 1,
    boundary_smoothness: float = 0.5,
    fill_holes_threshold: float = 0.5,
    overwrite: bool = False,
    mode: str = "full",
) -> Generator[tuple[int, int, str], None, None]:
    """Run flow-guided watershed segmentation for one position (full stack).

    Yields ``(done, total, label)`` progress tuples.

    Parameters
    ----------
    root_dir : str | Path
        Root project directory.
    pos : int
        Position index.
    flow_scale : float
        Blending factor: 0.0 = uniform, 1.0+ = flow-guided (default 1.0).
    cellpose_prob_threshold : float
        Mask out regions with probability below this value (default 0.0).
    flow_smoothing_sigma : float
        Gaussian smoothing of flow field (default 0.0 = no smoothing).
    max_iterations : int
        Maximum iterations for watershed expansion (default 50).
    uniform_growth_rate : float
        Baseline expansion probability (default 0.2).
    opening_radius : int
        Morphological opening radius for noise removal (default 1).
    closing_radius : int
        Morphological closing radius for hole filling (default 1).
    boundary_smoothness : float
        Boundary smoothing factor 0-1 (default 0.5).
    fill_holes_threshold : float
        Cellpose probability threshold for hole-filling (default 0.5).
    overwrite : bool
        Overwrite existing outputs.
    mode : str
        Execution mode: "full" (segmentation + postprocessing), "seg-only" (segmentation only),
        or "postprocess-only" (postprocessing on existing raw labels). Default: "full".
    """
    root_dir = Path(root_dir)

    # Load inputs
    print(f"pos{pos:02d}: Loading tracked labels and cellpose data…", flush=True)
    nuclear_labels = load_tracked_labels(root_dir, pos)
    if nuclear_labels is None:
        print(f"pos{pos:02d}: Could not load nuclear labels — skipping.", flush=True)
        return

    flow, prob = load_cellpose_data(root_dir, pos)
    if flow is None:
        print(f"pos{pos:02d}: Could not load cellpose data — skipping.", flush=True)
        return

    # Verify input shapes and normalize to (T, H, W, 2) and (T, H, W)
    if nuclear_labels.ndim != 3:
        print(
            f"[error] Expected nuclear_labels to be (T, H, W), got {nuclear_labels.shape}",
            file=sys.stderr,
        )
        return

    T = nuclear_labels.shape[0]

    # Handle flow field: expect (T, 2, H, W) or (2, H, W)
    if flow.ndim not in (3, 4):
        print(
            f"[error] Expected flow to be (T, 2, H, W) or (2, H, W), got {flow.shape}",
            file=sys.stderr,
        )
        return

    if flow.ndim == 4:
        # (T, 2, H, W) → transpose to (T, H, W, 2)
        if flow.shape[0] != T:
            print(
                f"[error] Timepoint mismatch: nuclear_labels T={T}, flow T={flow.shape[0]}",
                file=sys.stderr,
            )
            return
        flow = np.transpose(flow, (0, 2, 3, 1)).astype(np.float32)
    else:
        # (2, H, W) → transpose to (H, W, 2) and add time dimension
        flow = np.transpose(flow, (1, 2, 0)).astype(np.float32)  # (H, W, 2)
        flow = np.repeat(flow[np.newaxis, ...], T, axis=0)  # (T, H, W, 2)

    # Handle probability field: expect (T, H, W) or (H, W)
    if prob is not None:
        if prob.ndim == 3:
            # Already (T, H, W)
            if prob.shape[0] != T:
                print(
                    f"[error] Timepoint mismatch: nuclear_labels T={T}, prob T={prob.shape[0]}",
                    file=sys.stderr,
                )
                return
        elif prob.ndim == 2:
            # (H, W) → add time dimension to get (T, H, W)
            prob = np.repeat(prob[np.newaxis, ...], T, axis=0)
        else:
            print(
                f"[error] Unexpected prob shape: {prob.shape}",
                file=sys.stderr,
            )
            prob = None

    # Setup output
    out_dir = stage_dir(root_dir, pos, "flow_watershed")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "cell_labels.tif"
    raw_path = out_dir / "cell_labels_raw.tif"

    # Handle postprocess-only mode
    if mode == "postprocess-only":
        print(f"pos{pos:02d}: Mode=postprocess-only. Loading raw labels…", flush=True)
        if not raw_path.exists():
            print(f"[error] Raw labels not found at {raw_path}", file=sys.stderr)
            return
        raw_stack = tifffile.imread(str(raw_path)).astype(np.int32)
        T = raw_stack.shape[0]
        print(f"pos{pos:02d}: Applying post-processing to {T} frames…", flush=True)

        if out_path.exists() and not overwrite:
            print(f"pos{pos:02d}: Output exists — skipping.", flush=True)
            yield (T, T, "skipped")
            return

        stack = apply_postprocessing(
            raw_stack,
            cellpose_prob_stack=prob,
            opening_radius=opening_radius,
            closing_radius=closing_radius,
            boundary_smoothness=boundary_smoothness,
            fill_holes_threshold=fill_holes_threshold,
        )
        tifffile.imwrite(
            str(out_path),
            stack,
            compression="zlib",
            metadata={"axes": "TYX"},
        )
        print(f"pos{pos:02d}: Saved {stack.shape} to {out_path.name}", flush=True)
        for t in range(T):
            yield (t + 1, T, f"t{t:03d}")
        print(f"pos{pos:02d}: Done.", flush=True)
        return

    # Segmentation phase (for "full" and "seg-only" modes)
    if mode not in ("full", "seg-only"):
        raise ValueError(f"Invalid mode: {mode}")

    output_check_path = raw_path if mode == "seg-only" else out_path
    if output_check_path.exists() and not overwrite:
        print(f"pos{pos:02d}: Output exists — skipping.", flush=True)
        yield (T, T, "skipped")
        return

    print(f"pos{pos:02d}: Processing {T} timepoints (mode={mode})", flush=True)
    print(f"  nuclear_labels shape={nuclear_labels.shape}", flush=True)
    print(f"  flow shape={flow.shape}", flush=True)
    print(f"  flow_scale={flow_scale}", flush=True)
    print(f"  cellpose_prob_threshold={cellpose_prob_threshold}", flush=True)
    print(f"  flow_smoothing_sigma={flow_smoothing_sigma}", flush=True)
    print(f"  max_iterations={max_iterations}", flush=True)
    print(f"  uniform_growth_rate={uniform_growth_rate}", flush=True)
    if mode == "full":
        print(f"  opening_radius={opening_radius}", flush=True)
        print(f"  closing_radius={closing_radius}", flush=True)
        print(f"  boundary_smoothness={boundary_smoothness}", flush=True)
        print(f"  fill_holes_threshold={fill_holes_threshold}", flush=True)

    # Save run parameters
    run_params_path = out_dir / "run_params.json"
    if not run_params_path.exists():
        run_params_path.write_text(
            json.dumps(
                {
                    "stage": "cell_segmentation",
                    "pos": pos,
                    "params": {
                        "flow_scale": flow_scale,
                        "cellpose_prob_threshold": cellpose_prob_threshold,
                        "flow_smoothing_sigma": flow_smoothing_sigma,
                        "max_iterations": max_iterations,
                        "uniform_growth_rate": uniform_growth_rate,
                        "opening_radius": opening_radius,
                        "closing_radius": closing_radius,
                        "boundary_smoothness": boundary_smoothness,
                        "fill_holes_threshold": fill_holes_threshold,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # Process all timepoints into a stack (flow and prob are now normalized to (T, H, W, 2) and (T, H, W))
    cell_labels_stack = []

    for t in range(T):
        try:
            nuc_t = nuclear_labels[t]
            flow_t = flow[t]  # (H, W, 2)
            prob_t = prob[t] if prob is not None else None  # (H, W)

            # Run segmentation
            cell_labels = flow_guided_watershed(
                nuc_t,
                flow_t,
                cellpose_prob=prob_t,
                flow_scale=flow_scale,
                cellpose_prob_threshold=cellpose_prob_threshold,
                flow_smoothing_sigma=flow_smoothing_sigma,
                max_iterations=max_iterations,
                uniform_growth_rate=uniform_growth_rate,
            )

            cell_labels_stack.append(cell_labels)

        except Exception as e:
            print(f"[error] Failed to process t{t:03d}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            # Append zeros on error
            cell_labels_stack.append(np.zeros_like(nuclear_labels[t], dtype=np.int32))

        yield (t + 1, T, f"t{t:03d}")

    # Save raw (intermediate) stack
    raw_stack = np.stack(cell_labels_stack, axis=0).astype(np.int32)
    tifffile.imwrite(
        str(raw_path),
        raw_stack,
        compression="zlib",
        metadata={"axes": "TYX"},
    )
    print(f"pos{pos:02d}: Saved raw {raw_stack.shape} to {raw_path.name}", flush=True)

    # For "seg-only" mode, we're done
    if mode == "seg-only":
        print(f"pos{pos:02d}: Done (segmentation only).", flush=True)
        return

    # Apply post-processing (only in "full" mode)
    print(f"pos{pos:02d}: Applying post-processing…", flush=True)
    stack = apply_postprocessing(
        raw_stack,
        cellpose_prob_stack=prob,
        opening_radius=opening_radius,
        closing_radius=closing_radius,
        boundary_smoothness=boundary_smoothness,
        fill_holes_threshold=fill_holes_threshold,
    )

    # Save final stack
    tifffile.imwrite(
        str(out_path),
        stack,
        compression="zlib",
        metadata={"axes": "TYX"},
    )
    print(f"pos{pos:02d}: Saved {stack.shape} to {out_path.name}", flush=True)
    print(f"pos{pos:02d}: Done.", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="s03 — Flow-guided watershed cell segmentation (full stack)",
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root project directory",
    )
    parser.add_argument(
        "--pos",
        type=int,
        required=True,
        help="Position index",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON file with FlowWatershedConfig fields (optional)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "seg-only", "postprocess-only"],
        default="full",
        help="Execution mode: 'full' (segmentation+postprocessing), 'seg-only' (segmentation only), "
             "'postprocess-only' (postprocessing on existing raw labels)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    args = parser.parse_args()

    cfg_dict: dict = {
        "flow_scale": 1.0,
        "cellpose_prob_threshold": 0.0,
        "flow_smoothing_sigma": 0.0,
        "max_iterations": 50,
        "uniform_growth_rate": 0.2,
        "opening_radius": 1,
        "closing_radius": 1,
        "boundary_smoothness": 0.5,
        "fill_holes_threshold": 0.5,
    }
    if args.config:
        cfg_dict.update(json.loads(Path(args.config).read_text()))

    for done, total, label in run(
        args.root_dir,
        args.pos,
        flow_scale=cfg_dict.get("flow_scale", 1.0),
        cellpose_prob_threshold=cfg_dict.get("cellpose_prob_threshold", 0.0),
        flow_smoothing_sigma=cfg_dict.get("flow_smoothing_sigma", 0.0),
        max_iterations=cfg_dict.get("max_iterations", 50),
        uniform_growth_rate=cfg_dict.get("uniform_growth_rate", 0.2),
        opening_radius=cfg_dict.get("opening_radius", 1),
        closing_radius=cfg_dict.get("closing_radius", 1),
        boundary_smoothness=cfg_dict.get("boundary_smoothness", 0.5),
        fill_holes_threshold=cfg_dict.get("fill_holes_threshold", 0.5),
        overwrite=args.overwrite,
        mode=args.mode,
    ):
        print(f"[{done}/{total}] {label}", flush=True)

    sys.exit(0)


# ── StageProtocol wrapper ────────────────────────────────────────────────────


class _FlowWatershedStageClass:
    name = "flow_watershed"
    display_name = "Flow-Guided Watershed"

    def __init__(self):
        self.config = FlowWatershedConfig()

    def run(self, root_dir, pos: int, cfg: FlowWatershedConfig = None, overwrite: bool = False, mode: str = "full"):
        from cellflow.core.logging import StageLogger
        from cellflow.core.paths import log_path

        cfg = cfg or self.config
        log = StageLogger(log_path(root_dir, pos), self.name)
        with log:
            for progress in run(
                root_dir=root_dir, pos=pos,
                flow_scale=cfg.flow_scale,
                cellpose_prob_threshold=cfg.cellpose_prob_threshold,
                flow_smoothing_sigma=cfg.flow_smoothing_sigma,
                max_iterations=cfg.max_iterations,
                uniform_growth_rate=cfg.uniform_growth_rate,
                opening_radius=cfg.opening_radius,
                closing_radius=cfg.closing_radius,
                boundary_smoothness=cfg.boundary_smoothness,
                fill_holes_threshold=cfg.fill_holes_threshold,
                overwrite=overwrite,
                mode=mode,
            ):
                yield StageProgress(*progress)

    def validate_inputs(self, schema, root_dir, pos) -> ValidationResult:
        from cellflow.core.validation import validate_inputs

        tracking_labels = stage_dir(root_dir, pos, "tracking") / "tracked_labels_proj2d_corrected.tif"
        cell_dir = stage_dir(root_dir, pos, "cellpose_cell")
        return validate_inputs([
            tracking_labels,
            cell_dir / "cell_dp.tif",
        ])

    def is_complete(self, root_dir, pos) -> bool:
        d = stage_dir(root_dir, pos, "flow_watershed")
        return (d / "cell_labels.tif").exists()


FlowWatershedStage = _FlowWatershedStageClass()
