"""A per-frame result cache for the live-preview widgets.

Every live preview recomputes the current frame off the GUI thread on each param
edit or time scrub. Without a cache, scrubbing back to a frame that was just
computed re-runs the (often heavy) pipeline for an identical result. This cache
holds whatever a compute produced for each visited frame, keyed on a *signature*
of the parameters that determine the result, so:

* scrubbing back to an already-computed frame repaints instantly, and
* any param edit (a changed signature) drops every cached frame at once.

It mirrors the per-frame cache the cell-segmentation preview
(:mod:`itasc.napari.cell_workflow_widget`) grew first; the other previews
share this implementation rather than re-deriving it.

Stale-write safety: a worker started under an old signature may finish *after* an
edit has moved the cache to a new one. The owner adopts the current signature up
front with :meth:`sync`, reads with :meth:`get`, and stores with :meth:`put`
passing the signature the result was computed under — :meth:`put` ignores the
write when that signature is no longer current, so a late worker can never poison
the fresh cache.
"""
from __future__ import annotations

from typing import Any

_UNSET = object()


class FramePreviewCache:
    """Frame ``t`` → result, valid only while the parameter signature holds."""

    def __init__(self) -> None:
        self._key: Any = _UNSET
        self._frames: dict[int, Any] = {}

    def sync(self, key: Any) -> None:
        """Adopt *key* as the current signature, dropping all frames if it changed."""
        if key != self._key:
            self._key = key
            self._frames = {}

    def get(self, t: int) -> Any | None:
        """The cached result for frame *t* under the current signature, or ``None``."""
        return self._frames.get(t)

    def put(self, key: Any, t: int, value: Any) -> None:
        """Store *value* for frame *t* — ignored when *key* is no longer current.

        Call :meth:`sync` with the live signature before reading; pass the
        signature the *value* was computed under here so a worker that finishes
        after an edit cannot overwrite the re-keyed cache.
        """
        if key == self._key:
            self._frames[t] = value

    def clear(self) -> None:
        """Forget every frame and the signature (e.g. when the preview deactivates)."""
        self._key = _UNSET
        self._frames = {}
