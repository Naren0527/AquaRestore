import numpy as np
import pytest

from aquarestore.contracts import QualityMetrics
from aquarestore.diagnostics.denoising import denoise_image
from aquarestore.diagnostics.quality import calculate_quality_metrics


def sample_image():
    return np.random.default_rng(42).integers(
        0, 256, (64, 80, 3), dtype=np.uint8
    )


def test_denoising_preserves_shape_dtype_and_input():
    image = sample_image()
    original = image.copy()

    result = denoise_image(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result is not image
    np.testing.assert_array_equal(image, original)


def test_denoising_rejects_invalid_diameter():
    with pytest.raises(ValueError):
        denoise_image(sample_image(), diameter=0)


def test_denoising_rejects_invalid_sigma():
    with pytest.raises(ValueError):
        denoise_image(sample_image(), sigma_color=float("inf"))


def test_quality_metrics_returns_dataclass():
    metrics = calculate_quality_metrics(sample_image())

    assert isinstance(metrics, QualityMetrics)
    assert 0 <= metrics.mean_luminance <= 1
    assert 0 <= metrics.luminance_contrast <= 1
    assert 0 <= metrics.mean_saturation <= 1
    assert metrics.edge_strength >= 0
    assert metrics.uiqm is None
    assert metrics.uciqe is None


def test_quality_metrics_rejects_grayscale_image():
    image = np.zeros((20, 20), dtype=np.uint8)

    with pytest.raises(ValueError):
        calculate_quality_metrics(image)


def test_quality_metrics_handles_black_image():
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    metrics = calculate_quality_metrics(image)

    assert metrics.mean_luminance == 0
    assert metrics.luminance_contrast == 0
    assert metrics.mean_saturation == 0
    assert metrics.edge_strength == 0