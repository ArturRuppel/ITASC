"""Generate synthetic sample data for testing napariTissueFlow.

Creates:
  - sample_data/segmentation_labels.tif  (T x H x W label stack)
  - sample_data/nuclear_tracks.csv       (trackpy-style CSV)
"""
import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import Voronoi
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "sample_data"
RNG = np.random.default_rng(42)

# Parameters
N_FRAMES = 20
IMAGE_SIZE = 512
N_CELLS_SIDE = 8
SPACING = IMAGE_SIZE / (N_CELLS_SIDE + 1)
DRIFT_SPEED = 0.8  # pixels per frame
NOISE_AMPLITUDE = 1.0


def generate_cell_positions(n_frames, n_side, spacing, image_size):
    """Generate cell positions with slow drift and jitter over time."""
    # Initial grid
    yy, xx = np.meshgrid(
        np.arange(n_side) * spacing + spacing,
        np.arange(n_side) * spacing + spacing,
        indexing="ij",
    )
    base_positions = np.column_stack([yy.ravel(), xx.ravel()])
    n_cells = len(base_positions)

    # Random persistent drift direction per cell
    drift_angles = RNG.uniform(0, 2 * np.pi, n_cells)
    drift_dirs = np.column_stack([np.sin(drift_angles), np.cos(drift_angles)])

    all_positions = []
    for f in range(n_frames):
        drift = drift_dirs * DRIFT_SPEED * f
        jitter = RNG.normal(0, NOISE_AMPLITUDE, base_positions.shape)
        positions = base_positions + drift + jitter
        # Clamp to image bounds with margin
        margin = spacing * 0.3
        positions = np.clip(positions, margin, image_size - margin)
        all_positions.append(positions)

    return all_positions, n_cells


def positions_to_labels(positions, image_size):
    """Create a Voronoi-based label image from 2D positions."""
    labels = np.zeros((image_size, image_size), dtype=np.int32)
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    coords = np.column_stack([yy.ravel(), xx.ravel()])

    # Assign each pixel to nearest cell (1-indexed labels)
    dists = np.sum((coords[:, None, :] - positions[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(dists, axis=1) + 1
    labels = nearest.reshape(image_size, image_size)

    return labels


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Generating {N_FRAMES} frames, {N_CELLS_SIDE}x{N_CELLS_SIDE} = {N_CELLS_SIDE**2} cells...")
    all_positions, n_cells = generate_cell_positions(
        N_FRAMES, N_CELLS_SIDE, SPACING, IMAGE_SIZE
    )

    # --- Segmentation labels (.tif) ---
    print("Creating segmentation label stack...")
    label_stack = np.zeros((N_FRAMES, IMAGE_SIZE, IMAGE_SIZE), dtype=np.int32)
    for f in range(N_FRAMES):
        label_stack[f] = positions_to_labels(all_positions[f], IMAGE_SIZE)

    tif_path = OUTPUT_DIR / "segmentation_labels.tif"
    tifffile.imwrite(str(tif_path), label_stack)
    print(f"  Saved: {tif_path} — shape {label_stack.shape}")

    # --- Nuclear tracks (trackpy-style CSV) ---
    # trackpy output columns: x, y, mass, size, ecc, signal, raw_mass, ep, frame, particle
    print("Creating trackpy-style tracks CSV...")
    rows = []
    for f in range(N_FRAMES):
        for cell_id in range(n_cells):
            y, x = all_positions[f][cell_id]
            rows.append({
                "x": x,
                "y": y,
                "mass": RNG.normal(1000, 50),
                "size": RNG.normal(3.0, 0.2),
                "ecc": RNG.uniform(0, 0.3),
                "signal": RNG.normal(500, 30),
                "raw_mass": RNG.normal(1200, 60),
                "ep": RNG.uniform(0, 0.1),
                "frame": f,
                "particle": cell_id,
            })

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "nuclear_tracks.csv"
    df.to_csv(str(csv_path), index=False)
    print(f"  Saved: {csv_path} — {len(df)} rows, {n_cells} particles, {N_FRAMES} frames")

    print("\nDone! Load in napari:")
    print(f"  Labels:  File → Open → {tif_path}")
    print(f"  Tracks:  pd.read_csv('{csv_path}') then add as Points layer")


if __name__ == "__main__":
    main()
