"""The final-output commit contract: promote a working label file to a stable
base-folder name, and report the working-vs-committed state."""
from __future__ import annotations

import os

import numpy as np
import pytest
import tifffile

from cellflow.core.commit import commit_state, promote_labels


def _write_labels(path, value=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((4, 4), dtype=np.uint16)
    arr[1:3, 1:3] = value
    tifffile.imwrite(str(path), arr)
    return arr


def test_promote_copies_labels_to_destination(tmp_path):
    src = tmp_path / "2_nucleus" / "tracked_labels.tif"
    dst = tmp_path / "nucleus_labels.tif"
    arr = _write_labels(src, value=7)

    out = promote_labels(src, dst)

    assert out == dst
    assert dst.is_file()
    np.testing.assert_array_equal(tifffile.imread(str(dst)), arr)


def test_promote_creates_destination_parent(tmp_path):
    src = tmp_path / "3_cell" / "tracked_labels.tif"
    _write_labels(src)
    dst = tmp_path / "sub" / "cell_labels.tif"

    promote_labels(src, dst)

    assert dst.is_file()


def test_promote_overwrites_existing_destination(tmp_path):
    src = tmp_path / "tracked_labels.tif"
    _write_labels(src, value=3)
    dst = tmp_path / "cell_labels.tif"
    _write_labels(dst, value=99)

    promote_labels(src, dst)

    assert int(tifffile.imread(str(dst)).max()) == 3


def test_promote_missing_source_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        promote_labels(tmp_path / "nope.tif", tmp_path / "out.tif")


def test_promote_rejects_non_label_image(tmp_path):
    src = tmp_path / "float.tif"
    src.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(src), np.zeros((4, 4), dtype=np.float32))

    with pytest.raises(ValueError):
        promote_labels(src, tmp_path / "out.tif")


def test_commit_state_missing_when_no_working_file(tmp_path):
    assert commit_state(tmp_path / "work.tif", tmp_path / "final.tif") == "missing"


def test_commit_state_uncommitted_when_only_working(tmp_path):
    work = tmp_path / "work.tif"
    _write_labels(work)
    assert commit_state(work, tmp_path / "final.tif") == "uncommitted"


def test_commit_state_committed_after_promote(tmp_path):
    work = tmp_path / "2_nucleus" / "tracked_labels.tif"
    _write_labels(work)
    final = tmp_path / "nucleus_labels.tif"
    promote_labels(work, final)

    assert commit_state(work, final) == "committed"


def test_commit_state_stale_when_working_newer(tmp_path):
    work = tmp_path / "work.tif"
    final = tmp_path / "final.tif"
    _write_labels(work)
    _write_labels(final)
    # Working re-run after commit → strictly newer than the committed copy.
    os.utime(final, (1000, 1000))
    os.utime(work, (2000, 2000))

    assert commit_state(work, final) == "stale"
