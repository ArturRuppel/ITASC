"""s01b — Cellpose 2D cell segmentation (per-z-slice).

Runs Cellpose on each z-slice individually, storing full per-slice outputs and
z-averaged outputs directly in ``1_cellpose/`` for use by downstream stages.

Inputs (per position)
---------------------
  0_input/cell_4d.tif                  (T, Z, H, W) uint16 — cell 4D stack (488 nm)
  0_input/nucleus_4d.tif               (T, Z, H, W) uint16 — nucleus 4D stack (405 nm)

Outputs (per position)
----------------------
  1_cellpose/
    run_params.json
    cell_dp.tif        (T, Z, 2, H, W) float32 — per-z-slice flow fields
    cell_prob.tif      (T, Z, H, W)    float32 — per-z-slice probability maps
    cell_dp_zavg.tif   (T, 2, H, W)    float32 — z-averaged flow fields
    cell_prob_zavg.tif (T, H, W)      float32 — z-averaged probability maps
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Generator

import numpy as np
import tifffile

from cellflow.cellpose.config import CellposeConfig
from cellflow.cellpose.stages.raw_import import (
    cell_4d_path,
    nucleus_4d_path,
)
from cellflow.core.paths import stage_dir
from cellflow.core.protocol import StageProgress, ValidationResult


def cellpose_cell_dir(root_dir, pos):
    return stage_dir(root_dir, pos, "cellpose_cell")


def cell_dp_path(root_dir, pos):
    return cellpose_cell_dir(root_dir, pos) / "cell_dp.tif"


def cell_prob_path(root_dir, pos):
    return cellpose_cell_dir(root_dir, pos) / "cell_prob.tif"


def cell_dp_zavg_path(root_dir, pos):
    return cellpose_cell_dir(root_dir, pos) / "cell_dp_zavg.tif"


def cell_prob_zavg_path(root_dir, pos):
    return cellpose_cell_dir(root_dir, pos) / "cell_prob_zavg.tif"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_model(model_type: str, use_gpu: bool):
    from cellpose.models import CellposeModel

    gpu = use_gpu
    if gpu:
        try:
            import torch

            if not torch.cuda.is_available():
                print("CUDA not available — falling back to CPU", flush=True)
                gpu = False
            else:
                print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        except ImportError:
            print("torch not importable — falling back to CPU", flush=True)
            gpu = False

    model = CellposeModel(gpu=gpu, pretrained_model=model_type)
    print(f"Model '{model_type}' loaded  (gpu={gpu})", flush=True)
    return model


def _apply_gamma(img: np.ndarray, gamma: float) -> None:
    """Gamma-correct (H, W, C) img in-place."""
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            ch_norm = (ch - ch_min) / (ch_max - ch_min)
            img[:, :, c] = (ch_norm ** gamma) * (ch_max - ch_min) + ch_min


# ── Core run function ─────────────────────────────────────────────────────────


def run(
    root_dir: str | Path,
    pos: int,
    cfg: CellposeConfig,
    overwrite: bool = False,
) -> Generator[tuple[int, int, str], None, None]:
    """Run Cellpose 2D segmentation per-z-slice for one position.

    Yields ``(done, total, label)`` progress tuples.
    """
    root_dir = Path(root_dir)

    cell_stack_path = cell_4d_path(root_dir, pos)
    nuc_stack_path = nucleus_4d_path(root_dir, pos)

    if not cell_stack_path.exists() or not nuc_stack_path.exists():
        print(f"[error] Missing cell_4d.tif or nucleus_4d.tif for pos{pos:02d}", file=sys.stderr)
        return

    cell_stack = tifffile.imread(str(cell_stack_path)).astype(np.float32)
    nuc_stack = tifffile.imread(str(nuc_stack_path)).astype(np.float32)
    if cell_stack.ndim != 4 or nuc_stack.ndim != 4:
        print(
            f"[error] Expected 4D stacks (T,Z,H,W), got cell={cell_stack.shape} nucleus={nuc_stack.shape}",
            file=sys.stderr,
        )
        return
    if cell_stack.shape[0] != nuc_stack.shape[0]:
        print(
            f"[error] Timepoint mismatch: cell={cell_stack.shape[0]} nucleus={nuc_stack.shape[0]}",
            file=sys.stderr,
        )
        return

    T = cell_stack.shape[0]
    Z = cell_stack.shape[1]
    H, W = cell_stack.shape[2], cell_stack.shape[3]
    print(f"pos{pos:02d}  T={T}  Z={Z}  H={H}  W={W}", flush=True)
    if cfg.gamma is not None:
        print(f"  gamma={cfg.gamma}", flush=True)

    # Check outputs
    out_dir = cellpose_cell_dir(root_dir, pos)
    out_dir.mkdir(parents=True, exist_ok=True)

    dp_path = cell_dp_path(root_dir, pos)
    prob_path = cell_prob_path(root_dir, pos)
    dp_zavg_path = cell_dp_zavg_path(root_dir, pos)
    prob_zavg_path = cell_prob_zavg_path(root_dir, pos)

    if not overwrite and dp_path.exists() and prob_path.exists() and dp_zavg_path.exists() and prob_zavg_path.exists():
        print(f"pos{pos:02d}: all outputs exist — skipping.", flush=True)
        yield (T, T, f"pos{pos:02d} skipped")
        return

    run_params_path = out_dir / "run_params.json"
    if not run_params_path.exists():
        run_params_path.write_text(
            json.dumps(
                {
                    "stage": "cellpose_cell",
                    "pos": pos,
                    "params": cfg.model_dump(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # Load model
    model = _load_model(cfg.model, cfg.use_gpu)

    print(
        f"  diameter={cfg.diameter}  min_size={cfg.min_size}",
        flush=True,
    )

    # t_dp_list[t][z] → dp  (2, H, W)
    t_dp_list:   list[list[np.ndarray]] = []
    t_prob_list: list[list[np.ndarray]] = []

    try:
        for t in range(T):
            yield (t, T, f"t{t:03d}")

            cell_z = cell_stack[t]  # (Z, H, W)
            nuc_z = nuc_stack[t]    # (Z, H, W)

            print(f"  [{t+1:3d}/{T}]  t{t:03d}  Z={cell_z.shape[0]} ...", flush=True)

            z_dp_list:   list[np.ndarray] = []
            z_prob_list: list[np.ndarray] = []

            for z in range(cell_z.shape[0]):
                img = np.stack([cell_z[z], nuc_z[z]], axis=-1)  # (H, W, 2)

                if cfg.gamma is not None and cfg.gamma != 1.0:
                    _apply_gamma(img, cfg.gamma)

                _, flows, _ = model.eval(
                    img,
                    diameter=cfg.diameter if cfg.diameter > 0 else None,
                    min_size=cfg.min_size,
                )
                z_dp_list.append(flows[1].astype(np.float32))    # (2, H, W)
                z_prob_list.append(flows[2].astype(np.float32))  # (H, W)

            t_dp_list.append(z_dp_list)
            t_prob_list.append(z_prob_list)
            print("  done", flush=True)
            yield (t + 1, T, f"t{t:03d}")
    finally:
        del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    print("Assembling stacks …", flush=True)

    # (T, Z, 2, H, W) and (T, Z, H, W)
    stack_dp   = np.stack([np.stack(z_list, axis=0) for z_list in t_dp_list],   axis=0)
    stack_prob = np.stack([np.stack(z_list, axis=0) for z_list in t_prob_list], axis=0)

    # Z-averaged: (T, 2, H, W) and (T, H, W)
    stack_dp_zavg   = stack_dp.mean(axis=1).astype(np.float32)
    stack_prob_zavg = stack_prob.mean(axis=1).astype(np.float32)

    tifffile.imwrite(
        str(dp_path),
        stack_dp,
        compression="zlib",
        metadata={"axes": "TZCYX"},
    )
    tifffile.imwrite(
        str(prob_path),
        stack_prob,
        compression="zlib",
        metadata={"axes": "TZYX"},
    )
    tifffile.imwrite(
        str(dp_zavg_path),
        stack_dp_zavg,
        compression="zlib",
        metadata={"axes": "TCYX"},
    )
    tifffile.imwrite(
        str(prob_zavg_path),
        stack_prob_zavg,
        compression="zlib",
        metadata={"axes": "TYX"},
    )

    print(f"  → {dp_path.name}      {stack_dp.shape}       {stack_dp.dtype}", flush=True)
    print(f"  → {prob_path.name}    {stack_prob.shape}     {stack_prob.dtype}", flush=True)
    print(f"  → {dp_zavg_path.name} {stack_dp_zavg.shape}  {stack_dp_zavg.dtype}", flush=True)
    print(f"  → {prob_zavg_path.name} {stack_prob_zavg.shape} {stack_prob_zavg.dtype}", flush=True)
    print("Done.", flush=True)
    yield (T, T, "done")


# ── CLI entry point ───────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="s01b — Cellpose 2D cell segmentation (per-z-slice)",
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root project directory",
    )
    parser.add_argument(
        "--pos",
        required=True,
        type=int,
        help="Position index",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON file with CellposeConfig fields (optional)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs",
    )
    args = parser.parse_args()

    cfg_dict: dict = {}
    if args.config:
        cfg_dict = json.loads(Path(args.config).read_text())
    cfg = CellposeConfig(**cfg_dict)

    for done, total, label in run(args.root_dir, args.pos, cfg, overwrite=args.overwrite):
        print(f"[{done}/{total}] {label}", flush=True)

    sys.exit(0)


# ── StageProtocol wrapper ────────────────────────────────────────────────────


class _CellposeCellStageClass:
    name = "cellpose_cell"
    display_name = "Cellpose Cell"

    def __init__(self):
        self.config = CellposeConfig()

    def run(self, root_dir, pos: int, cfg: CellposeConfig = None, overwrite: bool = False):
        from cellflow.core.logging import StageLogger
        from cellflow.core.paths import log_path

        cfg = cfg or self.config
        log = StageLogger(log_path(root_dir, pos), self.name)
        with log:
            yield from run(root_dir=root_dir, pos=pos, cfg=cfg, overwrite=overwrite)

    def validate_inputs(self, schema, root_dir, pos) -> ValidationResult:
        from cellflow.core.validation import validate_inputs

        return validate_inputs([
            nucleus_4d_path(root_dir, pos),
            cell_4d_path(root_dir, pos),
        ])

    def is_complete(self, root_dir, pos) -> bool:
        return (
            cell_dp_path(root_dir, pos).exists()
            and cell_prob_path(root_dir, pos).exists()
            and cell_dp_zavg_path(root_dir, pos).exists()
            and cell_prob_zavg_path(root_dir, pos).exists()
        )


CellposeCellStage = _CellposeCellStageClass()
