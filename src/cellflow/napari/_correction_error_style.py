"""Severity colour ramp shared by the correction error views.

Pure module (no Qt): maps an error ``score`` in ``[0, 1]`` to an RGBA *heat*
colour so the lineage canvas and the worklist grade flagged cells the same way —
a faint warm yellow for low severity through to a saturated, opaque red for high.
Callers wrap the returned tuple in a ``QColor``.

Encoding severity in both hue *and* opacity is what makes a flagged frame read as
heat rather than a binary on/off mark: borderline flags stay faint and obvious
errors pop.
"""
from __future__ import annotations

# Heat stops as (score, r, g, b): low -> faint warm yellow, mid -> orange,
# high -> saturated red. Interpolated piecewise across the score.
_STOPS: tuple[tuple[float, int, int, int], ...] = (
    (0.0, 250, 215, 70),
    (0.5, 245, 150, 40),
    (1.0, 215, 45, 40),
)
_ALPHA_MIN = 110  # alpha at score 0 (faint but visible)
_ALPHA_MAX = 255  # alpha at score 1 (fully opaque)


def severity_rgba(score: float) -> tuple[int, int, int, int]:
    """Map ``score`` in ``[0, 1]`` to an ``(r, g, b, a)`` heat colour.

    Higher scores are redder and more opaque; scores outside ``[0, 1]`` clamp to
    the ends. Alpha ramps linearly with the score so low-severity flags stay
    faint and high-severity flags read bold.
    """
    s = max(0.0, min(1.0, float(score)))
    r, g, b = _STOPS[-1][1:]
    for (s0, r0, g0, b0), (s1, r1, g1, b1) in zip(_STOPS, _STOPS[1:]):
        if s <= s1:
            f = 0.0 if s1 == s0 else (s - s0) / (s1 - s0)
            r = round(r0 + (r1 - r0) * f)
            g = round(g0 + (g1 - g0) * f)
            b = round(b0 + (b1 - b0) * f)
            break
    a = round(_ALPHA_MIN + (_ALPHA_MAX - _ALPHA_MIN) * s)
    return (r, g, b, a)


__all__ = ["severity_rgba"]
