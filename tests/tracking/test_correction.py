
from cellflow.correction.labels import apply_gamma
import numpy as np
import pytest

def test_gamma_identity():
    logits = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(apply_gamma(logits, 1.0), logits)

def test_gamma_values():
    logits = np.array([0.0]) # prob 0.5
    # gamma 2.0 -> prob 0.25 -> logit log(0.25/0.75) = log(1/3) = -1.0986
    corrected = apply_gamma(logits, 2.0)
    assert np.isclose(corrected, np.log(1/3))
    
    # gamma 0.5 -> prob 0.707 -> logit log(0.707/0.293) = 0.857
    corrected = apply_gamma(logits, 0.5)
    assert np.isclose(corrected, np.log(np.sqrt(0.5) / (1 - np.sqrt(0.5))))
