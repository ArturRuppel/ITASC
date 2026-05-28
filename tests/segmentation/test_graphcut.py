# tests/segmentation/test_graphcut.py
import importlib.util
from pathlib import Path

import numpy as np
import pytest

# Load the script module without executing main()
_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "experiment_cell_2d_graphcut.py"

pytestmark = pytest.mark.skipif(
    not _SCRIPT.exists(),
    reason="archived exploratory graphcut script is outside the maintained test surface",
)


@pytest.fixture(scope="module")
def graphcut_module():
    spec = importlib.util.spec_from_file_location("experiment_cell_2d_graphcut", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_alpha_expansion_hard_pins_seed_pixels(graphcut_module):
    """Seed pixels must retain their label after expansion."""
    contours = np.zeros((6, 6), dtype=np.float32)
    foreground = np.ones((6, 6), dtype=bool)
    seeds = np.zeros((6, 6), dtype=np.uint32)
    seeds[0, 0] = 1
    seeds[5, 5] = 2

    labels = graphcut_module._run_alpha_expansion(
        contours, foreground, seeds, smoothness_weight=10.0, max_rounds=3
    )

    assert labels[0, 0] == 1
    assert labels[5, 5] == 2
    assert np.all(labels[foreground] > 0), "all foreground pixels must be labeled"


def test_alpha_expansion_background_pixels_stay_zero(graphcut_module):
    """Pixels outside the foreground mask must stay 0."""
    contours = np.zeros((4, 4), dtype=np.float32)
    foreground = np.ones((4, 4), dtype=bool)
    foreground[3, :] = False  # bottom row is background
    seeds = np.zeros((4, 4), dtype=np.uint32)
    seeds[0, 0] = 1
    seeds[1, 3] = 2

    labels = graphcut_module._run_alpha_expansion(
        contours, foreground, seeds, smoothness_weight=5.0, max_rounds=3
    )

    assert np.all(labels[~foreground] == 0)
    assert labels[0, 0] == 1
    assert labels[1, 3] == 2


def test_alpha_expansion_contour_barrier_splits_labels(graphcut_module):
    """A perfect contour barrier (value=1.0) makes crossing it free, concentrating
    the boundary there rather than elsewhere."""
    # 1-row, 8-column strip. Strong contour barrier between columns 3 and 4.
    contours = np.zeros((1, 8), dtype=np.float32)
    contours[0, 3] = 1.0
    contours[0, 4] = 1.0
    foreground = np.ones((1, 8), dtype=bool)
    seeds = np.zeros((1, 8), dtype=np.uint32)
    seeds[0, 0] = 1
    seeds[0, 7] = 2

    labels = graphcut_module._run_alpha_expansion(
        contours, foreground, seeds, smoothness_weight=50.0, max_rounds=5
    )

    assert labels[0, 0] == 1
    assert labels[0, 7] == 2
    # Left side of barrier should be label 1, right side label 2
    assert labels[0, 1] == 1
    assert labels[0, 2] == 1
    assert labels[0, 5] == 2
    assert labels[0, 6] == 2
