"""Shared fixtures and synthetic data generators for testing."""
import numpy as np
import pytest


def make_grid_positions(nx: int, ny: int, spacing: float = 20.0, noise: float = 0.0) -> np.ndarray:
    """Create a regular grid of 2D positions with optional jitter.

    Returns Nx2 array of (y, x) positions.
    """
    yy, xx = np.meshgrid(
        np.arange(ny) * spacing + spacing,
        np.arange(nx) * spacing + spacing,
        indexing="ij",
    )
    positions = np.column_stack([yy.ravel(), xx.ravel()])
    if noise > 0:
        rng = np.random.default_rng(42)
        positions += rng.normal(0, noise, positions.shape)
    return positions


def make_label_frame(n_cells_side: int = 4, image_size: int = 200) -> np.ndarray:
    """Create a label frame from a Voronoi tessellation of random points.

    Returns a 2D integer array where each cell has a unique label (1-indexed).
    """
    rng = np.random.default_rng(42)
    spacing = image_size / (n_cells_side + 1)
    positions = make_grid_positions(n_cells_side, n_cells_side, spacing=spacing, noise=spacing * 0.1)

    # Assign each pixel to nearest cell
    frame = np.zeros((image_size, image_size), dtype=np.int32)
    yy, xx = np.mgrid[0:image_size, 0:image_size]
    coords = np.column_stack([yy.ravel(), xx.ravel()])

    # Distance to each cell center
    dists = np.sum((coords[:, None, :] - positions[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1) + 1  # 1-indexed
    frame = labels.reshape(image_size, image_size)

    return frame


def make_label_stack(n_frames: int = 3, n_cells_side: int = 4, image_size: int = 200) -> np.ndarray:
    """Create a time series of label frames (identical for simplicity)."""
    frame = make_label_frame(n_cells_side, image_size)
    return np.stack([frame] * n_frames)


@pytest.fixture
def grid_positions():
    return make_grid_positions(4, 4, spacing=20.0)


@pytest.fixture
def label_frame():
    return make_label_frame(n_cells_side=4, image_size=200)


@pytest.fixture
def label_stack():
    return make_label_stack(n_frames=3, n_cells_side=4, image_size=200)


def make_label_stack_4d(
    n_tissues: int = 2, n_frames: int = 3, n_cells_side: int = 4, image_size: int = 200
) -> np.ndarray:
    """Create a 4D label stack (n_tissues, T, H, W) with slightly different tissues."""
    rng = np.random.default_rng(42)
    stacks = []
    for t in range(n_tissues):
        frame = make_label_frame(n_cells_side, image_size)
        stacks.append(np.stack([frame] * n_frames))
    return np.stack(stacks)


def make_label_stacks_ragged(
    n_frames_per_tissue: list[int] = [5, 8],
    n_cells_side: int = 4,
    image_size: int = 200,
) -> list[np.ndarray]:
    """Create a list of 3D label stacks with different frame counts per tissue."""
    stacks = []
    for n_frames in n_frames_per_tissue:
        frame = make_label_frame(n_cells_side, image_size)
        stacks.append(np.stack([frame] * n_frames))
    return stacks


@pytest.fixture
def label_stack_4d():
    return make_label_stack_4d(n_tissues=2, n_frames=3, n_cells_side=4, image_size=200)


@pytest.fixture
def label_stacks_ragged():
    return make_label_stacks_ragged(n_frames_per_tissue=[5, 8])


