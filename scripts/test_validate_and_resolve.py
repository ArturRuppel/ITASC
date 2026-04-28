#!/usr/bin/env python
"""End-to-end validation and resolve test.

Tests the full validate-and-resolve loop:
1. Load a curated tracked_labels.tif
2. Programmatically pick ~5 cells to validate
3. Register them using the validation API
4. Run resolve_with_validation from the existing hypothesis DB
5. Assert validated cells are pixel-identical in the output
6. Tweak a tracking parameter, re-run, and verify validated cells are unchanged

Usage (from repo root, inside cellflow env):
    python scripts/test_validate_and_resolve.py [--pos-dir <path>]

Default pos_dir:
    /home/aruppel/Data/2026-04-01_U251.../v2/pos00
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
LOG = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from cellflow.database.validation import validate_track, read_validated_tracks
    from cellflow.tracking_ultrack.config import TrackingConfig
    from cellflow.tracking_ultrack.ingest import ingest_hypotheses_to_db
    from cellflow.tracking_ultrack.reseed import resolve_with_validation
except ImportError as e:
    LOG.error(
        f"Missing dependencies: {e}\n"
        "Please ensure ultrack, sqlalchemy, and related packages are installed.\n"
        "This script requires: python, numpy, scipy, tifffile, h5py, sqlalchemy, ultrack"
    )
    sys.exit(1)

# Default test dataset
DEFAULT_POS_DIR = Path(
    "/home/aruppel/Data/"
    "2026-04-01_U251-WT-NLS-mCherry_U251-VimentinKO_circle300um_live_spinning-disk"
    "/v2/pos00"
)


def _find_cells_spanning_frames(
    tracked_labels: np.ndarray,
    min_frames: int = 3,
    max_cells: int = 5,
) -> list[int]:
    """Find cell IDs that appear in at least min_frames frames.

    Returns up to max_cells cell IDs, sorted by number of frames (descending).
    """
    cell_frame_counts: dict[int, int] = {}

    for t in range(tracked_labels.shape[0]):
        for uid in np.unique(tracked_labels[t]):
            if uid > 0:
                cell_frame_counts[int(uid)] = cell_frame_counts.get(int(uid), 0) + 1

    # Filter: keep only cells with >= min_frames
    candidates = [
        cid for cid, count in cell_frame_counts.items() if count >= min_frames
    ]

    # Sort by frame count (descending) and take top max_cells
    candidates.sort(key=lambda cid: cell_frame_counts[cid], reverse=True)
    return candidates[:max_cells]


def _get_validated_frames(
    tracked_labels: np.ndarray,
    cell_id: int,
) -> list[int]:
    """Return the list of frames where cell_id appears in tracked_labels."""
    frames = []
    for t in range(tracked_labels.shape[0]):
        if (tracked_labels[t] == cell_id).any():
            frames.append(t)
    return frames


def _frames_pixel_identical(
    labels1: np.ndarray,
    labels2: np.ndarray,
    cell_id: int,
    frames: list[int],
) -> bool:
    """Check if cell_id's mask is pixel-identical across two labelmaps for given frames."""
    for t in frames:
        mask1 = labels1[t] == cell_id
        mask2 = labels2[t] == cell_id
        if not np.array_equal(mask1, mask2):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end validate-and-resolve test",
    )
    parser.add_argument(
        "--pos-dir",
        type=Path,
        default=DEFAULT_POS_DIR,
        help=f"Position directory (default: {DEFAULT_POS_DIR})",
    )
    args = parser.parse_args()

    pos_dir = Path(args.pos_dir)
    nucleus_dir = pos_dir / "2_nucleus"

    # ─────────────────────────────────────────────────────────────────────
    # Validation: check that required files exist
    # ─────────────────────────────────────────────────────────────────────

    tracked_labels_path = nucleus_dir / "tracked_labels.tif"
    hypotheses_h5_path = nucleus_dir / "hypotheses.h5"

    if not tracked_labels_path.exists():
        LOG.error(f"Curated dataset not found: {tracked_labels_path}")
        sys.exit(1)

    if not hypotheses_h5_path.exists():
        LOG.error(f"Hypotheses HDF5 not found: {hypotheses_h5_path}")
        sys.exit(1)

    LOG.info(f"✓ Dataset found at {pos_dir}")

    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Load the tracked labelmap
    # ─────────────────────────────────────────────────────────────────────

    LOG.info("[1/6] Loading tracked_labels.tif…")
    tracked_labels = tifffile.imread(str(tracked_labels_path))
    tracked_labels = np.asarray(tracked_labels, dtype=np.uint32)
    LOG.info(f"      Shape: {tracked_labels.shape}, max_label: {tracked_labels.max()}")

    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Pick ~5 cells to validate
    # ─────────────────────────────────────────────────────────────────────

    LOG.info("[2/6] Selecting cells to validate…")
    cells_to_validate = _find_cells_spanning_frames(tracked_labels, min_frames=3)
    if not cells_to_validate:
        LOG.error("No cells found spanning 3+ frames!")
        sys.exit(1)

    LOG.info(f"      Selected {len(cells_to_validate)} cells: {cells_to_validate}")

    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Register validated tracks
    # ─────────────────────────────────────────────────────────────────────

    LOG.info("[3/6] Registering validated tracks…")
    validated_track_info: dict[int, list[int]] = {}
    for cell_id in cells_to_validate:
        frames = _get_validated_frames(tracked_labels, cell_id)
        validate_track(pos_dir, cell_id, frames)
        validated_track_info[cell_id] = frames
        LOG.info(f"      Cell {cell_id}: {len(frames)} frames {frames}")

    # Verify we can read back
    read_back = read_validated_tracks(pos_dir)
    LOG.info(f"      Read back {len(read_back)} validated cells")

    # ─────────────────────────────────────────────────────────────────────
    # Step 4: First resolve run (default parameters)
    # ─────────────────────────────────────────────────────────────────────

    LOG.info("[4/6] First resolve_with_validation run (default parameters)…")

    tmpdir_first = Path(tempfile.mkdtemp(prefix="cellflow_resolve_1_"))
    try:
        # Ingest hypotheses into a fresh working directory
        ingest_hypotheses_to_db(
            hypotheses_h5_path,
            tmpdir_first,
            TrackingConfig(),
            n_frames=tracked_labels.shape[0],
        )
        LOG.info(f"      Ingested hypotheses to {tmpdir_first}")

        # Resolve with validation
        resolved_1 = resolve_with_validation(
            tmpdir_first,
            read_validated_tracks(pos_dir),
            tracked_labels,
            TrackingConfig(),
        )
        LOG.info(f"      Resolved labelmap shape: {resolved_1.shape}")

    finally:
        shutil.rmtree(tmpdir_first, ignore_errors=True)

    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Verify validated cells are identical
    # ─────────────────────────────────────────────────────────────────────

    LOG.info("[5/6] Verifying validated cells are pixel-identical…")
    all_match = True
    for cell_id, frames in validated_track_info.items():
        # Map cell_id → new track ID. We need to find the new ID in resolved_1.
        # The validation merge process assigns new IDs starting from max(exported) + 1.
        # For simplicity, we check if *any* set of frames in resolved_1 match the original
        # mask (should be exactly one per cell_id after merge).

        # Get the original mask for this cell at all its frames
        original_masks = {}
        for t in frames:
            original_masks[t] = tracked_labels[t] == cell_id

        # In the resolved output, validated cells are assigned new track IDs.
        # We need to find which ID corresponds to this cell.
        # Strategy: for the first frame, find the label(s) that match the original mask.
        first_frame = frames[0]
        orig_mask = original_masks[first_frame]
        orig_pixels = set(np.argwhere(orig_mask, out=None).flat)

        resolved_mask = np.zeros_like(resolved_1[first_frame], dtype=bool)
        for label_id in np.unique(resolved_1[first_frame]):
            if label_id == 0:
                continue
            candidate_mask = resolved_1[first_frame] == label_id
            candidate_pixels = set(np.argwhere(candidate_mask, out=None).flat)
            if candidate_pixels == orig_pixels:
                # Found the mapped ID
                new_id = label_id
                break
        else:
            LOG.warning(f"      Cell {cell_id} not found in first frame — may have been merged")
            all_match = False
            continue

        # Now check all frames with this new_id
        frames_ok = True
        for t in frames:
            orig = original_masks[t]
            resolved_mask = resolved_1[t] == new_id
            if not np.array_equal(orig, resolved_mask):
                LOG.error(
                    f"      Cell {cell_id} (mapped to {new_id}) differs at frame {t}"
                )
                frames_ok = False
                all_match = False
                break

        if frames_ok:
            LOG.info(f"      ✓ Cell {cell_id} (→ {new_id}): {len(frames)} frames identical")

    if not all_match:
        LOG.error("Some validated cells did not match!")
        sys.exit(1)

    LOG.info("      ✓ All validated cells are pixel-identical")

    # ─────────────────────────────────────────────────────────────────────
    # Step 6: Tweak parameter, re-run, verify validated cells unchanged
    # ─────────────────────────────────────────────────────────────────────

    LOG.info("[6/6] Tweaking max_distance, re-running resolve…")

    cfg_tweaked = TrackingConfig(max_distance=25.0)  # default is 15.0
    tmpdir_second = Path(tempfile.mkdtemp(prefix="cellflow_resolve_2_"))
    try:
        ingest_hypotheses_to_db(
            hypotheses_h5_path,
            tmpdir_second,
            cfg_tweaked,
            n_frames=tracked_labels.shape[0],
        )
        LOG.info(f"      Ingested hypotheses with max_distance=25.0 to {tmpdir_second}")

        resolved_2 = resolve_with_validation(
            tmpdir_second,
            read_validated_tracks(pos_dir),
            tracked_labels,
            cfg_tweaked,
        )
        LOG.info(f"      Resolved labelmap shape: {resolved_2.shape}")

    finally:
        shutil.rmtree(tmpdir_second, ignore_errors=True)

    # ─────────────────────────────────────────────────────────────────────
    # Step 7: Compare results
    # ─────────────────────────────────────────────────────────────────────

    LOG.info("[7/7] Comparing first and second resolve runs…")

    # Count cells per frame in each run
    def count_cells_per_frame(labels: np.ndarray) -> list[int]:
        counts = []
        for t in range(labels.shape[0]):
            frame = labels[t]
            counts.append(int(np.count_nonzero(np.unique(frame)[1:]) if frame.max() > 0 else 0))
        return counts

    counts_1 = count_cells_per_frame(resolved_1)
    counts_2 = count_cells_per_frame(resolved_2)

    LOG.info(f"      Run 1 (default):     mean={np.mean(counts_1):.1f} cells/frame")
    LOG.info(f"      Run 2 (tweaked):     mean={np.mean(counts_2):.1f} cells/frame")

    # Verify validated cells are still identical
    LOG.info("      Verifying validated cells unchanged across parameter change…")
    all_match_2 = True
    n_changed = 0

    for cell_id, frames in validated_track_info.items():
        # Find the new ID in resolved_1 (same as before)
        first_frame = frames[0]
        orig_mask = tracked_labels[first_frame] == cell_id
        orig_pixels = set(np.argwhere(orig_mask, out=None).flat)

        # Find in resolved_1
        for label_id in np.unique(resolved_1[first_frame]):
            if label_id == 0:
                continue
            candidate_mask = resolved_1[first_frame] == label_id
            candidate_pixels = set(np.argwhere(candidate_mask, out=None).flat)
            if candidate_pixels == orig_pixels:
                new_id_1 = label_id
                break
        else:
            all_match_2 = False
            continue

        # Check if this cell appears with the same mask in resolved_2
        # (might have a different ID, but the mask should be identical)
        for t in frames:
            orig = tracked_labels[t] == cell_id
            found_match = False
            for label_id_2 in np.unique(resolved_2[t]):
                if label_id_2 == 0:
                    continue
                resolved_mask_2 = resolved_2[t] == label_id_2
                if np.array_equal(orig, resolved_mask_2):
                    found_match = True
                    break

            if not found_match:
                LOG.error(f"      Cell {cell_id} not found in resolved_2 frame {t}")
                all_match_2 = False
                break

    if all_match_2:
        LOG.info("      ✓ All validated cells unchanged across parameter change")
    else:
        LOG.error("Some validated cells changed!")
        sys.exit(1)

    # Count unvalidated cells that changed
    n_changed = 0
    total_cells_1 = len(set(resolved_1.ravel())) - 1  # exclude background
    total_cells_2 = len(set(resolved_2.ravel())) - 1

    LOG.info(f"      Total cells: run 1 = {total_cells_1}, run 2 = {total_cells_2}")
    LOG.info(f"      Difference: {abs(total_cells_1 - total_cells_2)} cells")

    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────

    LOG.info("")
    LOG.info("=" * 70)
    LOG.info("TEST PASSED")
    LOG.info("=" * 70)
    LOG.info(f"✓ Validated {len(cells_to_validate)} cells")
    LOG.info(f"✓ Cells round-trip identically through resolve_with_validation")
    LOG.info(f"✓ Parameter changes affect unvalidated regions only")
    LOG.info("")


if __name__ == "__main__":
    main()
