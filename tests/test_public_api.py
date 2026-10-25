
import numpy as np

from aquarestore.contracts import (
    correct_color,
    enhance_contrast,
)
from aquarestore.enhancement.color_correction import (
    correct_color as implementation_correct_color,
)
from aquarestore.enhancement.contrast import (
    enhance_contrast as implementation_enhance_contrast,
)


def test_public_color_correction_matches_implementation():
    image = np.random.default_rng(42).integers(
        0, 256, (60, 80, 3), dtype=np.uint8
    )

    result = correct_color(image)
    expected = implementation_correct_color(image)

    np.testing.assert_array_equal(result, expected)


def test_public_contrast_matches_implementation():
    image = np.random.default_rng(42).integers(
        0, 256, (60, 80, 3), dtype=np.uint8
    )

    result = enhance_contrast(image)
    expected = implementation_enhance_contrast(image)

    np.testing.assert_array_equal(result, expected)


def test_public_api_preserves_input():
    image = np.random.default_rng(42).integers(
        0, 256, (60, 80, 3), dtype=np.uint8
    )
    original = image.copy()

    correct_color(image)
    enhance_contrast(image)

    np.testing.assert_array_equal(image, original)

