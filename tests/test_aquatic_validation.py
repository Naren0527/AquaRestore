"""Unit tests for the aquatic image validation contracts."""

import numpy as np
import pytest

from aquarestore.classification.inference import ClassificationResult
from aquarestore.classification.validation import validate_aquatic_image


def test_validation_input_types():
    with pytest.raises(TypeError, match="must be a NumPy array"):
        validate_aquatic_image("not_an_array")  # type: ignore

    with pytest.raises(TypeError, match="must have dtype uint8"):
        validate_aquatic_image(np.zeros((100, 100, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="must have shape"):
        validate_aquatic_image(np.zeros((100, 100), dtype=np.uint8))

    with pytest.raises(ValueError, match="must not be empty"):
        validate_aquatic_image(np.zeros((0, 100, 3), dtype=np.uint8))


def test_validation_result_structure():
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    res = validate_aquatic_image(img, threshold=0.85)

    assert isinstance(res, ClassificationResult)
    assert isinstance(res.is_aquatic, bool)
    assert isinstance(res.accepted, bool)
    assert 0.0 <= res.confidence <= 1.0
    assert 0.0 <= res.aquatic_probability <= 1.0
    assert 0.0 <= res.non_aquatic_probability <= 1.0
    assert res.predicted_class in ("aquatic", "non_aquatic")
    assert isinstance(res.message, str)
    assert res.threshold == 0.85


def test_validation_threshold_parameter():
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    res_high = validate_aquatic_image(img, threshold=0.99)
    assert res_high.threshold == 0.99

    res_low = validate_aquatic_image(img, threshold=0.50)
    assert res_low.threshold == 0.50
