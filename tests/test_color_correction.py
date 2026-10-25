
import numpy as np
import pytest

from aquarestore.enhancement.color_correction import correct_color


def test_output_shape_and_dtype():
    image = np.full((40, 60, 3), [40, 80, 120], dtype=np.uint8)

    result = correct_color(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_input_is_not_modified():
    image = np.full((30, 30, 3), [40, 80, 120], dtype=np.uint8)
    original = image.copy()

    correct_color(image)

    np.testing.assert_array_equal(image, original)


def test_gray_world_reduces_channel_imbalance():
    image = np.empty((50, 50, 3), dtype=np.uint8)
    image[:] = [40, 80, 120]

    result = correct_color(image)

    before = image.astype(np.float32).mean(axis=(0, 1))
    after = result.astype(np.float32).mean(axis=(0, 1))

    before_spread = np.ptp(before)
    after_spread = np.ptp(after)

    assert after_spread < before_spread


def test_uniform_gray_image_remains_unchanged():
    image = np.full((25, 25, 3), 100, dtype=np.uint8)

    result = correct_color(image)

    np.testing.assert_array_equal(result, image)


def test_invalid_dtype_is_rejected():
    image = np.zeros((20, 20, 3), dtype=np.float32)

    with pytest.raises(TypeError):
        correct_color(image)


def test_grayscale_image_is_rejected():
    image = np.zeros((20, 20), dtype=np.uint8)

    with pytest.raises(ValueError):
        correct_color(image)


def test_empty_image_is_rejected():
    image = np.zeros((0, 20, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        correct_color(image)


def test_non_array_input_is_rejected():
    with pytest.raises(TypeError):
        correct_color([[1, 2, 3]])


def test_zero_strength_returns_identical_copy():
    image = np.random.default_rng(10).integers(
        0, 256, (40, 50, 3), dtype=np.uint8
    )

    result = correct_color(image, strength=0.0)

    np.testing.assert_array_equal(result, image)
    assert not np.shares_memory(result, image)


def test_full_strength_and_partial_strength_are_valid():
    image = np.random.default_rng(11).integers(
        0, 256, (40, 50, 3), dtype=np.uint8
    )

    partial = correct_color(image, strength=0.35)
    full = correct_color(image, strength=1.0)

    assert partial.shape == image.shape
    assert full.shape == image.shape
    assert partial.dtype == np.uint8
    assert full.dtype == np.uint8


def test_invalid_strength_is_rejected():
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        correct_color(image, strength=-0.1)

    with pytest.raises(ValueError):
        correct_color(image, strength=1.1)

    with pytest.raises(TypeError):
        correct_color(image, strength=True)
