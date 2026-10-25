
import numpy as np
import pytest

from aquarestore.enhancement.adaptive import (
    measure_luminance_contrast,
    select_clip_limit,
)


def test_contrast_measurement_returns_valid_range():
    image = np.random.default_rng(42).integers(
        0, 256, (80, 100, 3), dtype=np.uint8
    )

    contrast = measure_luminance_contrast(image)

    assert isinstance(contrast, float)
    assert 0.0 <= contrast <= 1.0


def test_uniform_image_has_zero_contrast():
    image = np.full((50, 50, 3), 100, dtype=np.uint8)

    contrast = measure_luminance_contrast(image)

    assert contrast == 0.0


def test_contrast_measurement_does_not_modify_input():
    image = np.random.default_rng(42).integers(
        0, 256, (50, 60, 3), dtype=np.uint8
    )
    original = image.copy()

    measure_luminance_contrast(image)

    np.testing.assert_array_equal(image, original)


def test_low_contrast_selects_higher_clip_limit():
    low = select_clip_limit(0.1)
    high = select_clip_limit(0.8)

    assert low > high


def test_clip_limit_is_bounded():
    for contrast in (0.0, 0.25, 0.5, 0.75, 1.0):
        result = select_clip_limit(contrast)
        assert 1.0 <= result <= 4.0


def test_mapping_is_deterministic():
    assert select_clip_limit(0.35) == select_clip_limit(0.35)


def test_custom_bounds_are_respected():
    result = select_clip_limit(0.5, minimum=2.0, maximum=6.0)

    assert result == 4.0


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_contrast_is_rejected(value):
    with pytest.raises(ValueError):
        select_clip_limit(value)


def test_non_numeric_contrast_is_rejected():
    with pytest.raises(TypeError):
        select_clip_limit("low")


def test_invalid_image_dtype_is_rejected():
    image = np.zeros((30, 30, 3), dtype=np.float32)

    with pytest.raises(TypeError):
        measure_luminance_contrast(image)


def test_grayscale_image_is_rejected():
    image = np.zeros((30, 30), dtype=np.uint8)

    with pytest.raises(ValueError):
        measure_luminance_contrast(image)


def test_non_array_image_is_rejected():
    with pytest.raises(TypeError):
        measure_luminance_contrast([[1, 2, 3]])


def test_invalid_bounds_are_rejected():
    with pytest.raises(ValueError):
        select_clip_limit(0.5, minimum=0, maximum=4)

    with pytest.raises(ValueError):
        select_clip_limit(0.5, minimum=5, maximum=4)
