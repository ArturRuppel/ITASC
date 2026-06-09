"""Unit tests for the shared live-preview per-frame cache."""
from cellflow.napari._preview_cache import FramePreviewCache


def test_get_returns_none_before_anything_is_stored():
    cache = FramePreviewCache()
    cache.sync(("params", 1))
    assert cache.get(0) is None


def test_put_then_get_roundtrips_under_same_key():
    cache = FramePreviewCache()
    key = ("params", 1)
    cache.sync(key)
    cache.put(key, 3, "result-3")
    assert cache.get(3) == "result-3"
    assert cache.get(4) is None


def test_changing_the_signature_drops_every_frame():
    cache = FramePreviewCache()
    cache.sync(("p", 1))
    cache.put(("p", 1), 0, "a")
    cache.put(("p", 1), 1, "b")

    cache.sync(("p", 2))  # a param edit
    assert cache.get(0) is None
    assert cache.get(1) is None


def test_stale_put_under_an_old_key_is_ignored():
    """A worker that finishes after an edit must not poison the re-keyed cache."""
    cache = FramePreviewCache()
    cache.sync(("p", 1))
    cache.put(("p", 1), 0, "fresh")

    # The signature moves on (edit) before a slow worker started under the old
    # key returns.
    cache.sync(("p", 2))
    cache.put(("p", 1), 5, "stale")

    assert cache.get(5) is None  # the stale write was dropped
    assert cache.get(0) is None  # and the old frame is gone with the re-key


def test_clear_forgets_frames_and_signature():
    cache = FramePreviewCache()
    cache.sync(("p", 1))
    cache.put(("p", 1), 0, "a")
    cache.clear()
    # After clear, the same key is "new" again, so nothing survives.
    cache.sync(("p", 1))
    assert cache.get(0) is None
