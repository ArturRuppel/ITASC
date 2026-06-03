"""Unit tests for the shared severity heat ramp (pure, no Qt)."""
from __future__ import annotations

import pytest

from cellflow.napari._correction_error_style import severity_rgba


def test_returns_rgba_quad_of_bytes() -> None:
    r, g, b, a = severity_rgba(0.5)
    assert all(0 <= c <= 255 for c in (r, g, b, a))


def test_alpha_is_monotonic_in_score() -> None:
    # Severity reads as opacity: a worse score is more opaque.
    alphas = [severity_rgba(s)[3] for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert alphas == sorted(alphas)
    assert alphas[0] < alphas[-1]


def test_low_and_high_differ_in_hue() -> None:
    low = severity_rgba(0.0)[:3]
    high = severity_rgba(1.0)[:3]
    assert low != high
    # The ramp shifts warm-yellow -> red: green recedes and red dominates green
    # far more strongly at the high end (a redder hue).
    assert high[1] < low[1]                       # less green than the warm low end
    assert (high[0] - high[1]) > (low[0] - low[1])  # red:green gap widens -> redder


def test_scores_clamp_outside_unit_interval() -> None:
    assert severity_rgba(-1.0) == severity_rgba(0.0)
    assert severity_rgba(2.0) == severity_rgba(1.0)


@pytest.mark.parametrize("score", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_full_score_is_fully_opaque_and_zero_is_faint(score: float) -> None:
    a = severity_rgba(score)[3]
    assert a == 255 if score == 1.0 else a < 255
