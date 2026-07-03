from __future__ import annotations

import numpy as np

from cellflow.core import label_store
from cellflow.core.label_store import (
    read_full_tracked_stack,
    write_full_tracked_stack,
)


def _make_stack() -> np.ndarray:
    stack = np.zeros((4, 6, 6), dtype=np.uint32)
    stack[0, 1:3, 1:3] = 5
    stack[2, 3:5, 3:5] = 70000  # exercise a >uint16 id
    return stack


def test_write_full_tracked_stack_round_trips(tmp_path):
    path = tmp_path / "tracked_labels.tif"
    stack = _make_stack()
    write_full_tracked_stack(path, stack)
    loaded = read_full_tracked_stack(path)
    assert loaded.dtype == np.uint32
    np.testing.assert_array_equal(loaded, stack)


def test_full_write_encodes_once(tmp_path, monkeypatch):
    """Saving a whole stack must hit the TIFF encoder exactly once, not once per
    frame (the O(T^2) regression in the per-frame save loop)."""
    calls: list[int] = []
    real = label_store.imwrite_grayscale

    def counting(path, data, **kw):
        calls.append(int(np.asarray(data).shape[0]))
        return real(path, data, **kw)

    monkeypatch.setattr(label_store, "imwrite_grayscale", counting)

    stack = _make_stack()
    write_full_tracked_stack(tmp_path / "tracked_labels.tif", stack)
    assert calls == [stack.shape[0]]  # one encode of the full T-frame stack


def test_full_write_squeezes_singleton_z(tmp_path):
    path = tmp_path / "tracked_labels.tif"
    stack = np.zeros((3, 1, 5, 5), dtype=np.uint32)  # (T, Z=1, Y, X)
    stack[1, 0, 1:3, 1:3] = 9
    write_full_tracked_stack(path, stack)
    loaded = read_full_tracked_stack(path)
    assert loaded.shape == (3, 5, 5)
    assert int(loaded[1].max()) == 9
