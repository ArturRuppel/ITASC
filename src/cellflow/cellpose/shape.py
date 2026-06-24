"""Layout-free input canonicalisation for the standalone tool.

The standalone Cellpose tool supports neither stitching-as-a-mode nor true 3D
segmentation, so it never needs the 4-way ``2D/2D+t/3D/3D+t`` layout: it segments
every plane individually. The only place axis *identity* matters is tracking,
which links the **z** axis by spatial overlap (stitch) and the **t** axis by
motion (laptrack). By convention the shorter of the two leading axes is ``z``
(few z-slices, many timepoints is the common case) and the longer is ``t``.

:func:`to_canonical_tzyx` applies that convention so the rest of the pipeline can
keep working in canonical ``(T, Z, Y, X)`` without the user declaring a layout.
"""
from __future__ import annotations

import numpy as np

__all__ = ["to_canonical_tzyx", "describe_axes"]


def to_canonical_tzyx(arr: np.ndarray) -> np.ndarray:
    """Coerce any 2-D..4-D input to canonical ``(T, Z, Y, X)`` without a layout.

    - 2-D ``(Y, X)`` → ``(1, 1, Y, X)``.
    - 3-D → one leading axis, read as **time** (the common 2D+t case); ``Z`` is a
      singleton: ``(T, 1, Y, X)``.
    - 4-D → two leading axes; the **shorter** is ``Z`` and the longer is ``T``, so
      the result is reordered to ``(T, Z, Y, X)`` (a no-op when already so).
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[np.newaxis, np.newaxis]
    if arr.ndim == 3:
        return arr[:, np.newaxis]
    if arr.ndim == 4:
        if arr.shape[0] < arr.shape[1]:
            # leading axis 0 is the shorter one → it is Z; swap to (T, Z, Y, X).
            arr = np.swapaxes(arr, 0, 1)
        return arr
    raise ValueError(f"expected a 2-D..4-D array, got shape {arr.shape}")


def describe_axes(shape: tuple[int, ...]) -> str:
    """Human-readable summary of how an input shape is interpreted.

    Used for widget status so the user can see the inferred ``T``/``Z`` without a
    picker, e.g. ``"6×512×512 → T=6, Z=1"`` or ``"40×5×512×512 → T=40, Z=5"``.
    """
    canon = to_canonical_tzyx(np.empty(shape, dtype=np.uint8))
    T, Z = int(canon.shape[0]), int(canon.shape[1])
    dims = "×".join(str(s) for s in shape)
    return f"{dims} → T={T}, Z={Z}"
