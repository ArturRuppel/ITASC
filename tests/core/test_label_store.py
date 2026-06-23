from __future__ import annotations

import numpy as np
import tifffile

from cellflow.core import label_store
from cellflow.core.label_store import (
    read_full_tracked_stack,
    tracked_frame_exists,
    write_full_tracked_stack,
    write_tracked_frame,
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


def test_full_write_matches_per_frame_loop(tmp_path):
    """The batched writer must produce a byte-for-frame-identical stack to the
    legacy per-frame loop it replaces."""
    stack = _make_stack()

    loop_path = tmp_path / "loop.tif"
    for t in range(stack.shape[0]):
        write_tracked_frame(loop_path, t, stack[t])

    batch_path = tmp_path / "batch.tif"
    write_full_tracked_stack(batch_path, stack)

    np.testing.assert_array_equal(
        read_full_tracked_stack(loop_path), read_full_tracked_stack(batch_path)
    )


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


def test_tracked_frame_exists_true_for_all_background_written_frame(tmp_path):
    """A legitimately-tracked frame that happens to be all background must read
    as existing — existence is about being written, not about having labels."""
    path = tmp_path / "tracked_labels.tif"
    stack = np.zeros((3, 5, 5), dtype=np.uint32)
    stack[0, 1, 1] = 4
    # frame 1 is intentionally all background; frame 2 has a label
    stack[2, 2, 2] = 8
    write_full_tracked_stack(path, stack)

    assert tracked_frame_exists(path, 0) is True
    assert tracked_frame_exists(path, 1) is True  # all-background but written
    assert tracked_frame_exists(path, 2) is True
    assert tracked_frame_exists(path, 3) is False  # never written


def test_tracked_frame_exists_via_per_frame_write(tmp_path):
    path = tmp_path / "tracked_labels.tif"
    write_tracked_frame(path, 0, np.zeros((5, 5), dtype=np.uint32))  # all bg
    assert tracked_frame_exists(path, 0) is True
    assert tracked_frame_exists(path, 1) is False


def test_tracked_frame_exists_missing_file(tmp_path):
    assert tracked_frame_exists(tmp_path / "nope.tif", 0) is False


def test_tracked_frame_exists_legacy_no_sidecar(tmp_path):
    """Files written before the sidecar existed fall back to the content
    heuristic so old projects keep working."""
    path = tmp_path / "tracked_labels.tif"
    stack = np.zeros((2, 5, 5), dtype=np.uint32)
    stack[0, 1, 1] = 4
    tifffile.imwrite(str(path), stack)  # raw write, no sidecar
    assert tracked_frame_exists(path, 0) is True   # has a label
    assert tracked_frame_exists(path, 1) is False  # all background, legacy


def test_full_write_squeezes_singleton_z(tmp_path):
    path = tmp_path / "tracked_labels.tif"
    stack = np.zeros((3, 1, 5, 5), dtype=np.uint32)  # (T, Z=1, Y, X)
    stack[1, 0, 1:3, 1:3] = 9
    write_full_tracked_stack(path, stack)
    loaded = read_full_tracked_stack(path)
    assert loaded.shape == (3, 5, 5)
    assert int(loaded[1].max()) == 9
