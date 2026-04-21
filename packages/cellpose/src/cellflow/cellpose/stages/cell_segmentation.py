"""s04 — Gravity-flow cell segmentation.

Expands corrected nuclear labels outward using Euler integration of a blended
N-body gravitational field (from nuclear centroids) and the Cellpose flow field.

Inputs (per position)
---------------------
  3_correction/nuclear_labels_corrected.tif  (T, H, W)    uint32  — nuclear labels (2D projection)
  1b_cellpose_cell/cell_dp.tif               (T, 2, H, W) float32 — flow field
  1b_cellpose_cell/cell_prob.tif             (T, H, W)    float32 — cellpose probability

Outputs (per position)
----------------------
  4_cell_segmentation/
    run_params.json
    cell_labels.tif                  (T, H, W)    int32  — cell segmentation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Generator

import numpy as np
import tifffile

from cellflow.cellpose.processing.gravity_flow import gravity_flow_segmentation
from cellflow.cellpose.config import CellSegmentationConfig
from cellflow.core.paths import stage_dir
from cellflow.core.protocol import StageProgress, ValidationResult


def load_tracked_labels(root_dir: str | Path, pos: int) -> np.ndarray | None:
    """Load tracked nuclear labels (2D projection) from 3_correction."""
    tracking_path = stage_dir(root_dir, pos, "correction") / "nuclear_labels_corrected.tif"
    if not tracking_path.exists():
        print(f"[error] Tracked labels not found: {tracking_path}", file=sys.stderr)
        return None
    return tifffile.imread(str(tracking_path)).astype(np.int32)



def load_cellpose_data(root_dir: str | Path, pos: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load z-averaged cellpose flow and probability.

    Prefers cell_dp_zavg.tif / cell_prob_zavg.tif (per-z-slice pipeline).
    Falls back to cell_dp.tif / cell_prob.tif for older outputs.

    Returns raw arrays with shapes as stored on disk:
    - flow: (T, 2, H, W) or (2, H, W)
    - prob: (T, H, W) or (H, W)
    """
    cell_dir = stage_dir(root_dir, pos, "cellpose_cell")

    # Prefer zavg outputs (per-z-slice pipeline), fall back to legacy names
    flow_path = cell_dir / "cell_dp_zavg.tif"
    if not flow_path.exists():
        flow_path = cell_dir / "cell_dp.tif"
    if not flow_path.exists():
        print(f"[error] Flow field not found in {cell_dir}", file=sys.stderr)
        return None, None

    flow = tifffile.imread(str(flow_path)).astype(np.float32)
    print(f"  flow shape={flow.shape}  ({flow_path.name})", flush=True)

    prob_path = cell_dir / "cell_prob_zavg.tif"
    if not prob_path.exists():
        prob_path = cell_dir / "cell_prob.tif"
    prob = None
    if prob_path.exists():
        prob = tifffile.imread(str(prob_path)).astype(np.float32)
        print(f"  prob shape={prob.shape}  ({prob_path.name})", flush=True)

    return flow, prob


def run(
    root_dir: str | Path,
    pos: int,
    flow_step_scale: float = 0.2,
    capture_radius: float = 3.0,
    flow_weight: float = 0.5,
    gravity_falloff: float = 2.0,
    cellpose_prob_threshold: float = 0.0,
    flow_smoothing_sigma: float = 0.0,
    max_iterations: int = 100,
    overwrite: bool = False,
) -> Generator[tuple[int, int, str], None, None]:
    """Run gravity-flow cell segmentation for one position (full stack).

    Yields ``(done, total, label)`` progress tuples.
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
    out_dir = stage_dir(root_dir, pos, "cell_segmentation")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "cell_labels.tif"

    if out_path.exists() and not overwrite:
        print(f"pos{pos:02d}: Output exists — skipping.", flush=True)
        yield (T, T, "skipped")
        return

    print(f"pos{pos:02d}: Processing {T} timepoints", flush=True)
    print(f"  nuclear_labels shape={nuclear_labels.shape}", flush=True)
    print(f"  flow shape={flow.shape}", flush=True)
    print(f"  flow_step_scale={flow_step_scale}", flush=True)
    print(f"  capture_radius={capture_radius}", flush=True)
    print(f"  flow_weight={flow_weight}", flush=True)
    print(f"  gravity_falloff={gravity_falloff}", flush=True)
    print(f"  cellpose_prob_threshold={cellpose_prob_threshold}", flush=True)
    print(f"  flow_smoothing_sigma={flow_smoothing_sigma}", flush=True)
    print(f"  max_iterations={max_iterations}", flush=True)

    # Save run parameters
    run_params_path = out_dir / "run_params.json"
    if not run_params_path.exists():
        run_params_path.write_text(
            json.dumps(
                {
                    "stage": "cell_segmentation",
                    "pos": pos,
                    "params": {
                        "flow_step_scale":          flow_step_scale,
                        "capture_radius":           capture_radius,
                        "flow_weight":           flow_weight,
                        "gravity_falloff":          gravity_falloff,
                        "cellpose_prob_threshold":  cellpose_prob_threshold,
                        "flow_smoothing_sigma":     flow_smoothing_sigma,
                        "max_iterations":           max_iterations,
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
            cell_labels = gravity_flow_segmentation(
                nuc_t,
                flow_t,
                cellpose_prob=prob_t,
                flow_step_scale=flow_step_scale,
                capture_radius=capture_radius,
                flow_weight=flow_weight,
                gravity_falloff=gravity_falloff,
                cellpose_prob_threshold=cellpose_prob_threshold,
                flow_smoothing_sigma=flow_smoothing_sigma,
                max_iterations=max_iterations,
            )

            cell_labels_stack.append(cell_labels)

        except Exception as e:
            print(f"[error] Failed to process t{t:03d}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            # Append zeros on error
            cell_labels_stack.append(np.zeros_like(nuclear_labels[t], dtype=np.int32))

        yield (t + 1, T, f"t{t:03d}")

    stack = np.stack(cell_labels_stack, axis=0).astype(np.int32)
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
        description="s04 — Gravity-flow cell segmentation (full stack)",
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
        help="Path to JSON file with CellSegmentationConfig fields (optional)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    args = parser.parse_args()

    cfg_dict: dict = {
        "flow_step_scale":         0.2,
        "cellpose_prob_threshold": 0.0,
        "flow_smoothing_sigma":    0.0,
        "max_iterations":          100,
    }
    if args.config:
        loaded = json.loads(Path(args.config).read_text())
        for k in ("postprocess_steps", "opening_radius", "closing_radius",
                  "boundary_smoothness", "fill_holes_threshold"):
            loaded.pop(k, None)
        cfg_dict.update(loaded)

    for done, total, label in run(
        args.root_dir,
        args.pos,
        flow_step_scale=cfg_dict.get("flow_step_scale", 0.2),
        capture_radius=cfg_dict.get("capture_radius", 3.0),
        flow_weight=cfg_dict.get("flow_weight", 0.5),
        gravity_falloff=cfg_dict.get("gravity_falloff", 2.0),
        cellpose_prob_threshold=cfg_dict.get("cellpose_prob_threshold", 0.0),
        flow_smoothing_sigma=cfg_dict.get("flow_smoothing_sigma", 0.0),
        max_iterations=cfg_dict.get("max_iterations", 100),
        overwrite=args.overwrite,
    ):
        print(f"[{done}/{total}] {label}", flush=True)

    sys.exit(0)


# ── StageProtocol wrapper ────────────────────────────────────────────────────


class _CellSegmentationStageClass:
    name = "cell_segmentation"
    display_name = "Cell Segmentation"

    def __init__(self):
        self.config = CellSegmentationConfig()

    def run(self, root_dir, pos: int, cfg: CellSegmentationConfig = None, overwrite: bool = False):
        from cellflow.core.logging import StageLogger
        from cellflow.core.paths import log_path

        cfg = cfg or self.config
        log = StageLogger(log_path(root_dir, pos), self.name)
        with log:
            for progress in run(
                root_dir=root_dir, pos=pos,
                flow_step_scale=cfg.flow_step_scale,
                capture_radius=cfg.capture_radius,
                flow_weight=cfg.flow_weight,
                gravity_falloff=cfg.gravity_falloff,
                cellpose_prob_threshold=cfg.cellpose_prob_threshold,
                flow_smoothing_sigma=cfg.flow_smoothing_sigma,
                max_iterations=cfg.max_iterations,
                overwrite=overwrite,
            ):
                yield StageProgress(*progress)

    def validate_inputs(self, schema, root_dir, pos) -> ValidationResult:
        from cellflow.core.validation import validate_inputs

        tracking_labels = stage_dir(root_dir, pos, "correction") / "nuclear_labels_corrected.tif"
        cell_dir = stage_dir(root_dir, pos, "cellpose_cell")
        return validate_inputs([
            tracking_labels,
            cell_dir / "cell_dp.tif",
        ])

    def is_complete(self, root_dir, pos) -> bool:
        d = stage_dir(root_dir, pos, "cell_segmentation")
        return (d / "cell_labels.tif").exists()


CellSegmentationStage = _CellSegmentationStageClass()
