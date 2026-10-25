
import numpy as np
import pytest

from aquarestore.enhancement.contrast import enhance_contrast


def test_output_shape_and_dtype():
    image = np.full((80, 100, 3), [50, 80, 110], dtype=np.uint8)

    result = enhance_contrast(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_input_is_not_modified():
    image = np.random.default_rng(42).integers(
        0, 256, (60, 80, 3), dtype=np.uint8
    )
    original = image.copy()

    enhance_contrast(image)

    np.testing.assert_array_equal(image, original)


def test_grayscale_input_is_rejected():
    image = np.zeros((30, 30), dtype=np.uint8)

    with pytest.raises(ValueError):
        enhance_contrast(image)


def test_invalid_dtype_is_rejected():
    image = np.zeros((30, 30, 3), dtype=np.float32)

    with pytest.raises(TypeError):
        enhance_contrast(image)


def test_non_array_input_is_rejected():
    with pytest.raises(TypeError):
        enhance_contrast([[1, 2, 3]])


def test_invalid_clip_limit_is_rejected():
    image = np.zeros((30, 30, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        enhance_contrast(image, clip_limit=0)

    with pytest.raises(ValueError):
        enhance_contrast(image, clip_limit=float("inf"))


def test_invalid_tile_grid_is_rejected():
    image = np.zeros((30, 30, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        enhance_contrast(image, tile_grid_size=(8, 0))


def test_non_integer_tile_grid_is_rejected():
    image = np.zeros((30, 30, 3), dtype=np.uint8)

    with pytest.raises(TypeError):
        enhance_contrast(image, tile_grid_size=(8, 4.5))
