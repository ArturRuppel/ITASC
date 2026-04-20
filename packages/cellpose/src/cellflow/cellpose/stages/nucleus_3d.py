"""s01a — Cellpose 3D nucleus segmentation.

Scans an input directory for ``*.tif`` volumes (Z, Y, X) and runs Cellpose in
3-D mode, writing flow (``*_dp.tif``) and probability (``*_prob.tif``) outputs
to the output directory.

Usage
-----
    python -m ultrack_wrapper.stages.s01a_cellpose_nucleus \\
        --input-dir /path/to/raw_nucleus \\
        --output-dir /path/to/1a_cellpose_nucleus \\
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
    """Return sorted list of *.tif files in *input_dir*."""
    return sorted(Path(input_dir).glob("*.tif"))


# ── Core run function ─────────────────────────────────────────────────────────


def run(
    input_dir: str | Path,
    output_dir: str | Path,
    cfg: CellposeConfig,
    overwrite: bool = False,
) -> Generator[tuple[int, int, str], None, None]:
    """Run Cellpose 3-D segmentation on all TIFFs in *input_dir*.

    Yields ``(done, total, label)`` progress tuples.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = discover_input_files(input_dir)
    total = len(tif_files)
    if total == 0:
        print(f"[warning] No .tif files found in {input_dir}", file=sys.stderr)
        return

    model = None  # lazy load once
    try:
        for done, in_path in enumerate(tif_files, start=1):
            dp_path   = output_dir / f"{in_path.stem}_dp.tif"
            prob_path = output_dir / f"{in_path.stem}_prob.tif"

            label = in_path.name
            yield (done - 1, total, label)

            if not overwrite and dp_path.exists() and prob_path.exists():
                print(f"  {label}: outputs exist — skipping.", flush=True)
                yield (done, total, label)
                continue

            img = tifffile.imread(str(in_path))
            if img.ndim != 3:
                print(f"[error] {label}: expected (Z,Y,X), got {img.shape}", file=sys.stderr)
                yield (done, total, label)
                continue

            print(f"  {label}  shape={img.shape}  dtype={img.dtype}", flush=True)

            # Optional gamma correction
            gamma = cfg.gamma
            if gamma is not None and gamma != 1.0:
                print(f"  gamma={gamma}", flush=True)
                img = img.astype(np.float32)
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = (
                        ((img - img_min) / (img_max - img_min)) ** gamma
                        * (img_max - img_min)
                        + img_min
                    )

            if model is None:
                model = _load_model(cfg.model, cfg.use_gpu)

            print(
                f"  diameter={cfg.diameter}  anisotropy={cfg.anisotropy}"
                f"  min_size={cfg.min_size}",
                flush=True,
            )

            _, flows, _ = model.eval(
                img,
                do_3D=True,
                z_axis=0,
                diameter=cfg.diameter if cfg.diameter > 0 else None,
                anisotropy=cfg.anisotropy,
                min_size=cfg.min_size,
            )

            dP       = flows[1].astype(np.float32)  # (3, Z, Y, X)
            cellprob = flows[2].astype(np.float32)  # (Z, Y, X)

            tifffile.imwrite(str(dp_path),   dP,       compression="zlib", metadata={"axes": "CZYX"})
            tifffile.imwrite(str(prob_path), cellprob, compression="zlib", metadata={"axes": "ZYX"})

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
        description="s01a — Cellpose 3-D nucleus segmentation",
    )
    parser.add_argument("--input-dir",  required=True,
                        help="Directory containing raw nucleus TIFFs (Z, Y, X)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write _dp.tif / _prob.tif outputs")
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
    display_name = "Cellpose Nucleus (3D)"

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

        input_dir = stage_dir(root_dir, pos, "raw_import")
        return validate_inputs([input_dir])

    def is_complete(self, root_dir, pos) -> bool:
        out = stage_dir(root_dir, pos, "cellpose_nucleus")
        return out.exists() and any(out.glob("*_dp.tif"))


CellposeNucleusStage = _CellposeNucleusStageClass()
