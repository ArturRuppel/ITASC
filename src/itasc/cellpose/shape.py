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

import logging

import numpy as np

__all__ = ["to_canonical_tzyx", "describe_axes"]

LOG = logging.getLogger(__name__)


def to_canonical_tzyx(
    arr: np.ndarray,
    *,
    z_axis: int | None = None,
    warn: bool = True,
) -> np.ndarray:
    """Coerce any 2-D..4-D input to canonical ``(T, Z, Y, X)`` without a layout.

    - 2-D ``(Y, X)`` → ``(1, 1, Y, X)``.
    - 3-D → one leading axis, read as **time** (the common 2D+t case); ``Z`` is a
      singleton: ``(T, 1, Y, X)``.
    - 4-D → two leading axes; the **shorter** is ``Z`` and the longer is ``T``, so
      the result is reordered to ``(T, Z, Y, X)`` (a no-op when already so).

    Parameters
    ----------
    z_axis:
        For 4-D input, explicitly declare which leading axis is ``Z`` (``0`` or
        ``1``), bypassing the shorter-axis-is-Z heuristic. Use this when the
        acquisition has fewer timepoints than z-slices, where the heuristic would
        otherwise transpose ``T`` and ``Z`` and corrupt tracking.
    warn:
        When ``True`` (default), log a warning if a 4-D input's ``T``/``Z``
        identity is *guessed* from axis lengths (both leading axes non-singleton
        and no ``z_axis`` given), since the guess can be wrong and is otherwise
        silent. Descriptive callers (e.g. :func:`describe_axes`) pass ``False``.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr[np.newaxis, np.newaxis]
    if arr.ndim == 3:
        return arr[:, np.newaxis]
    if arr.ndim == 4:
        if z_axis is not None:
            if z_axis not in (0, 1):
                raise ValueError(
                    f"z_axis for a 4-D array must be 0 or 1, got {z_axis}"
                )
            if z_axis == 0:
                arr = np.swapaxes(arr, 0, 1)
            return arr
        if warn and arr.shape[0] > 1 and arr.shape[1] > 1:
            # Both leading axes are non-singleton, so T vs Z cannot be read off
            # the data; the shorter-axis-is-Z fallback is a guess that silently
            # transposes an acquisition with fewer timepoints than z-slices.
            longer, shorter = (0, 1) if arr.shape[0] >= arr.shape[1] else (1, 0)
            LOG.warning(
                "4-D input %s has ambiguous leading axes; assuming axis %d is T "
                "(len %d) and axis %d is Z (len %d) by the shorter-axis-is-Z "
                "convention. Pass z_axis=0 or z_axis=1 to to_canonical_tzyx to "
                "declare the layout if this is wrong.",
                tuple(int(s) for s in arr.shape),
                longer, int(arr.shape[longer]),
                shorter, int(arr.shape[shorter]),
            )
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
    canon = to_canonical_tzyx(np.empty(shape, dtype=np.uint8), warn=False)
    T, Z = int(canon.shape[0]), int(canon.shape[1])
    dims = "×".join(str(s) for s in shape)
    return f"{dims} → T={T}, Z={Z}"
