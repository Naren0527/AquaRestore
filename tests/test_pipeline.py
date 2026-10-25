
import numpy as np
import pytest

from aquarestore.enhancement.pipeline import enhance_image


def test_output_shape_and_dtype():
    image = np.random.default_rng(42).integers(
        0, 256, (80, 100, 3), dtype=np.uint8
    )

    result = enhance_image(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_input_is_not_modified():
    image = np.random.default_rng(42).integers(
        0, 256, (80, 100, 3), dtype=np.uint8
    )
    original = image.copy()

    enhance_image(image)

    np.testing.assert_array_equal(image, original)


def test_output_is_a_separate_array():
    image = np.random.default_rng(42).integers(
        0, 256, (80, 100, 3), dtype=np.uint8
    )

    result = enhance_image(image)

    assert not np.shares_memory(result, image)


def test_invalid_dtype_is_rejected():
    image = np.zeros((30, 30, 3), dtype=np.float32)

    with pytest.raises(TypeError):
        enhance_image(image)


def test_grayscale_input_is_rejected():
    image = np.zeros((30, 30), dtype=np.uint8)

    with pytest.raises(ValueError):
        enhance_image(image)


def test_empty_image_is_rejected():
    image = np.zeros((0, 30, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        enhance_image(image)


def test_non_array_input_is_rejected():
    with pytest.raises(TypeError):
        enhance_image([[1, 2, 3]])


def test_invalid_adaptive_argument_is_rejected():
    image = np.zeros((30, 30, 3), dtype=np.uint8)

    with pytest.raises(TypeError):
        enhance_image(image, adaptive="yes")

def test_adaptive_mode_runs_successfully():
    image = np.random.default_rng(42).integers(
        0, 256, (80, 100, 3), dtype=np.uint8
    )

    result = enhance_image(image, adaptive=True)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_fixed_and_adaptive_modes_do_not_modify_input():
    image = np.random.default_rng(42).integers(
        0, 256, (80, 100, 3), dtype=np.uint8
    )
    original = image.copy()

    enhance_image(image, adaptive=False)
    enhance_image(image, adaptive=True)

    np.testing.assert_array_equal(image, original)
