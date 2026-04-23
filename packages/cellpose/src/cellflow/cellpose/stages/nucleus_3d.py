"""s01a — Cellpose 3D nucleus segmentation.

Reads the raw 4D nucleus stack ``nucleus_4d.tif`` from ``0_input`` and runs
Cellpose in 3-D mode once per timepoint. The per-timepoint flow and probability
outputs are stacked into time-stacked TIFFs written directly under ``1_cellpose/``.

Usage
-----
    python -m ultrack_wrapper.stages.s01a_cellpose_nucleus \\
        --input-dir /path/to/0_input \\
        --output-dir /path/to/1_cellpose \\
        --config /tmp/cp_config.json \\
        [--overwrite]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Generator

import numpy as np
import tifffile

from cellflow.cellpose.config import CellposeConfig
from cellflow.core.paths import stage_dir
from cellflow.core.protocol import StageProgress, ValidationResult


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


def discover_input_files(input_dir: str | Path) -> list[Path]:
    """Return the nucleus stack input file(s) for *input_dir*.

    The new layout expects a single ``nucleus_4d.tif`` file. A narrow glob
    fallback is kept for older projects that still store per-timepoint inputs.
    """
    input_dir = Path(input_dir)
    nucleus_stack = input_dir / "nucleus_4d.tif"
    if nucleus_stack.exists():
        return [nucleus_stack]
    return sorted(input_dir.glob("nucleus_3d_t*.tif"))


# ── Core run function ─────────────────────────────────────────────────────────


def run(
    input_dir: str | Path,
    output_dir: str | Path,
    cfg: CellposeConfig,
    overwrite: bool = False,
) -> Generator[tuple[int, int, str], None, None]:
    """Run Cellpose 3-D segmentation on the nucleus 4D stack in *input_dir*.

    Yields ``(done, total, label)`` progress tuples.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = discover_input_files(input_dir)
    total = len(tif_files)
    if total == 0:
        print(f"[warning] No nucleus_4d.tif files found in {input_dir}", file=sys.stderr)
        return

    model = None  # lazy load once
    try:
        for done, in_path in enumerate(tif_files, start=1):
            prefix = in_path.stem.replace("_4d", "")
            dp_path = output_dir / f"{prefix}_dp_4d.tif"
            prob_path = output_dir / f"{prefix}_prob_4d.tif"

            label = in_path.name
            yield (done - 1, total, label)

            if not overwrite and dp_path.exists() and prob_path.exists():
                print(f"  {label}: outputs exist — skipping.", flush=True)
                yield (done, total, label)
                continue

            img = tifffile.imread(str(in_path))
            if img.ndim != 4:
                print(f"[error] {label}: expected (T,Z,Y,X), got {img.shape}", file=sys.stderr)
                yield (done, total, label)
                continue

            print(f"  {label}  shape={img.shape}  dtype={img.dtype}", flush=True)

            # Optional gamma correction is applied per timepoint volume.
            gamma = cfg.gamma
            if gamma is not None and gamma != 1.0:
                print(f"  gamma={gamma}", flush=True)
                img = img.astype(np.float32)
            else:
                img = img.astype(np.float32)

            if model is None:
                model = _load_model(cfg.model, cfg.use_gpu)

            print(
                f"  diameter={cfg.diameter}  anisotropy={cfg.anisotropy}"
                f"  min_size={cfg.min_size}",
                flush=True,
            )

            dp_frames: list[np.ndarray] = []
            prob_frames: list[np.ndarray] = []
            for frame in img:
                if gamma is not None and gamma != 1.0:
                    frame = frame.copy()
                    frame_min, frame_max = frame.min(), frame.max()
                    if frame_max > frame_min:
                        frame = (
                            ((frame - frame_min) / (frame_max - frame_min)) ** gamma
                            * (frame_max - frame_min)
                            + frame_min
                        )
                _, flows, _ = model.eval(
                    frame,
                    do_3D=True,
                    z_axis=0,
                    diameter=cfg.diameter if cfg.diameter > 0 else None,
                    anisotropy=cfg.anisotropy,
                    min_size=cfg.min_size,
                )
                dp_frames.append(flows[1].astype(np.float32))  # (3, Z, Y, X)
                prob_frames.append(flows[2].astype(np.float32))  # (Z, Y, X)

            dP = np.stack(dp_frames, axis=0)
            cellprob = np.stack(prob_frames, axis=0)

            tifffile.imwrite(str(dp_path), dP, compression="zlib", metadata={"axes": "TZCYX"})
            tifffile.imwrite(str(prob_path), cellprob, compression="zlib", metadata={"axes": "TZYX"})

            print(f"  → {dp_path.name}  {dP.shape}", flush=True)
            print(f"  → {prob_path.name}  {cellprob.shape}", flush=True)

            yield (done, total, label)
    finally:
        if model is not None:
            del model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    print("Done.", flush=True)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="s01a — Cellpose nucleus segmentation",
    )
    parser.add_argument("--input-dir",  required=True,
                        help="Directory containing nucleus_4d.tif (T, Z, Y, X)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write nucleus_dp_4d.tif / nucleus_prob_4d.tif outputs")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON file with CellposeConfig fields (optional)",
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing outputs")
    args = parser.parse_args()

    cfg_dict: dict = {}
    if args.config:
        cfg_dict = json.loads(Path(args.config).read_text())
    cfg = CellposeConfig(**cfg_dict)

    for done, total, label in run(args.input_dir, args.output_dir, cfg,
                                   overwrite=args.overwrite):
        print(f"[{done}/{total}] {label}", flush=True)

    sys.exit(0)


# ── StageProtocol wrapper ────────────────────────────────────────────────────


class _CellposeNucleusStageClass:
    name = "cellpose_nucleus"
    display_name = "Cellpose Nucleus"

    def __init__(self):
        self.config = CellposeConfig()

    def run(self, input_dir, output_dir, cfg: CellposeConfig = None, overwrite: bool = False):
        from cellflow.core.logging import StageLogger

        cfg = cfg or self.config
        pos_dir = Path(output_dir).parent.parent
        log = StageLogger(pos_dir / "pipeline.log", self.name)
        with log:
            for progress in run(input_dir=input_dir, output_dir=output_dir, cfg=cfg, overwrite=overwrite):
                yield StageProgress(*progress)

    def validate_inputs(self, schema, root_dir, pos) -> ValidationResult:
        from cellflow.core.validation import validate_inputs

        input_dir = stage_dir(root_dir, pos, "raw_import") / "nucleus_4d.tif"
        return validate_inputs([input_dir])

    def is_complete(self, root_dir, pos) -> bool:
        out = stage_dir(root_dir, pos, "cellpose_nucleus")
        return (
            out.exists()
            and (
                (out / "nucleus_dp_4d.tif").exists()
                or (out / "nucleus_dp.tif").exists()
            )
            and (
                (out / "nucleus_prob_4d.tif").exists()
                or (out / "nucleus_prob.tif").exists()
            )
        )


CellposeNucleusStage = _CellposeNucleusStageClass()
