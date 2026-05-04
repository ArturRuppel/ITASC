import sys
import types

import numpy as np

# Stub torch and cellpose so tests run without GPU/torch installed
_torch_stub = types.ModuleType("torch")
_torch_stub.device = lambda *a, **kw: "cpu"
_torch_cuda_stub = types.ModuleType("torch.cuda")
_torch_cuda_stub.is_available = lambda: False
_torch_stub.cuda = _torch_cuda_stub
sys.modules.setdefault("torch", _torch_stub)
sys.modules.setdefault("torch.cuda", _torch_cuda_stub)
_cp_ver_stub = types.ModuleType("cellpose.version")
_cp_ver_stub.version = "0.0.0"
_cp_ver_stub.version_str = "0.0.0"
_cp_dyn_stub = types.ModuleType("cellpose.dynamics")
_cp_dyn_stub.compute_masks = None
_cp_stub = types.ModuleType("cellpose")
sys.modules.setdefault("cellpose.version", _cp_ver_stub)
sys.modules.setdefault("cellpose.dynamics", _cp_dyn_stub)
sys.modules.setdefault("cellpose", _cp_stub)

from cellflow import segmentation


def test_fill_and_close_labels_fills_per_label_bounding_boxes(monkeypatch):
    labels = np.zeros((20, 20), dtype=np.uint32)
    labels[2:5, 3:6] = 1
    labels[3, 4] = 0
    labels[12:14, 15:17] = 2
    labels[12, 16] = 0
    seen_shapes = []

    def fake_fill(mask):
        seen_shapes.append(mask.shape)
        return np.ones_like(mask, dtype=bool)

    monkeypatch.setattr("scipy.ndimage.binary_fill_holes", fake_fill)

    result = segmentation._fill_and_close_labels(labels)

    assert result[2:5, 3:6].tolist() == [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    assert result[12:14, 15:17].tolist() == [[2, 2], [2, 2]]
    assert seen_shapes == [(3, 3), (2, 2)]


def test_build_mean_z_consensus_boundary_returns_correct_shapes(monkeypatch):
    import numpy as np
    from unittest.mock import patch

    rng = np.random.default_rng(0)
    n_z, n_y, n_x = 3, 8, 10
    prob_zyx = rng.standard_normal((n_z, n_y, n_x)).astype(np.float32)
    dp_zcyx = rng.standard_normal((n_z, 2, n_y, n_x)).astype(np.float32)
    thresholds = [-2.0, -1.0, 0.0]
    gammas = [0.8, 1.0]

    call_log = []

    def fake_compute_masks(dp, prob, cellprob_threshold, flow_threshold, niter, do_3D, device):
        call_log.append((dp.shape, prob.shape, cellprob_threshold))
        masks = np.zeros(prob.shape, dtype=np.uint32)
        masks[2:5, 3:8] = 1
        return masks

    with patch("cellpose.dynamics.compute_masks", fake_compute_masks):
        from cellflow.segmentation import build_mean_z_consensus_boundary
        boundary, foreground = build_mean_z_consensus_boundary(
            prob_zyx, dp_zcyx, thresholds, gammas
        )

    assert boundary.shape == (n_y, n_x), f"Expected ({n_y},{n_x}), got {boundary.shape}"
    assert foreground.shape == (n_y, n_x)
    assert boundary.dtype == np.float32
    assert foreground.dtype == np.float32
    assert 0.0 <= float(boundary.min()) <= float(boundary.max()) <= 1.0
    assert 0.0 <= float(foreground.min()) <= float(foreground.max()) <= 1.0
    # 2 gammas × 3 thresholds = 6 calls, all on 2D projected inputs
    assert len(call_log) == 6
    assert all(dp_shape == (2, n_y, n_x) for dp_shape, _, _ in call_log)
    assert all(prob_shape == (n_y, n_x) for _, prob_shape, _ in call_log)
    assert sorted(set(t for _, _, t in call_log)) == thresholds


def test_build_mean_z_consensus_boundary_single_gamma_default(monkeypatch):
    import numpy as np
    from unittest.mock import patch

    n_z, n_y, n_x = 2, 6, 6
    prob_zyx = np.zeros((n_z, n_y, n_x), dtype=np.float32)
    dp_zcyx = np.zeros((n_z, 2, n_y, n_x), dtype=np.float32)

    call_log = []

    def fake_compute_masks(dp, prob, cellprob_threshold, flow_threshold, niter, do_3D, device):
        call_log.append(cellprob_threshold)
        return np.zeros(prob.shape, dtype=np.uint32)

    with patch("cellpose.dynamics.compute_masks", fake_compute_masks):
        from cellflow.segmentation import build_mean_z_consensus_boundary
        boundary, foreground = build_mean_z_consensus_boundary(
            prob_zyx, dp_zcyx, [-1.0, 0.0]
        )

    # Default gammas=(1.0,) → 1 gamma × 2 thresholds = 2 calls
    assert len(call_log) == 2
    assert boundary.shape == (n_y, n_x)


def test_build_mean_z_consensus_boundary_invokes_mask_callback(monkeypatch):
    import numpy as np
    from unittest.mock import patch

    n_z, n_y, n_x = 2, 5, 5
    prob_zyx = np.zeros((n_z, n_y, n_x), dtype=np.float32)
    dp_zcyx = np.zeros((n_z, 2, n_y, n_x), dtype=np.float32)

    def fake_compute_masks(dp, prob, **kwargs):
        return np.zeros(prob.shape, dtype=np.uint32)

    cb_calls = []

    def my_callback(masks, gamma_idx, thresh_idx):
        cb_calls.append((masks.shape, gamma_idx, thresh_idx))

    with patch("cellpose.dynamics.compute_masks", fake_compute_masks):
        from cellflow.segmentation import build_mean_z_consensus_boundary
        build_mean_z_consensus_boundary(
            prob_zyx, dp_zcyx, [-1.0, 0.0], [1.0, 1.2], mask_callback=my_callback
        )

    # 2 gammas × 2 thresholds → 4 callback invocations
    assert len(cb_calls) == 4
    assert all(shape == (n_y, n_x) for shape, _, _ in cb_calls)
    gamma_idx_vals = sorted(set(gi for _, gi, _ in cb_calls))
    thresh_idx_vals = sorted(set(ti for _, _, ti in cb_calls))
    assert gamma_idx_vals == [0, 1]
    assert thresh_idx_vals == [0, 1]
