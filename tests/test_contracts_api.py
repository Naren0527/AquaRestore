
import numpy as np
import pytest

from aquarestore.contracts import (
    EnhancementResult,
    QualityMetrics,
    calculate_quality_metrics,
    compare_images,
    denoise_image,
    enhance_image,
)


def sample_image():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)


def test_quality_metrics_returns_dataclass():
    metrics = calculate_quality_metrics(sample_image())

    assert isinstance(metrics, QualityMetrics)
    assert 0 <= metrics.mean_luminance <= 1
    assert 0 <= metrics.luminance_contrast <= 1
    assert 0 <= metrics.mean_saturation <= 1
    assert metrics.uiqm is None
    assert metrics.uciqe is None


def test_denoising_preserves_shape_dtype_and_input():
    image = sample_image()
    original = image.copy()

    result = denoise_image(image)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(image, original)


def test_public_enhance_image_returns_enhancement_result():
    result = enhance_image(sample_image(), adaptive=False)

    assert isinstance(result, EnhancementResult)
    assert result.method == "fixed"
    assert result.image.shape == (64, 80, 3)
    assert result.parameters_used["clip_limit"] == 2.0
    assert isinstance(result.metrics_before, QualityMetrics)
    assert isinstance(result.metrics_after, QualityMetrics)


def test_adaptive_public_api_records_adaptive_mode():
    result = enhance_image(sample_image(), adaptive=True)

    assert result.method == "adaptive"
    assert result.parameters_used["adaptive"] is True
    assert 1.0 <= result.parameters_used["clip_limit"] <= 3.0


def test_compare_images_returns_metric_differences():
    original = sample_image()
    enhanced = enhance_image(original, adaptive=False).image

    comparison = compare_images(original, enhanced)

    assert "metrics_before" in comparison
    assert "metrics_after" in comparison
    assert "differences" in comparison
    assert "luminance_contrast" in comparison["differences"]


def test_compare_images_rejects_different_shapes():
    with pytest.raises(ValueError):
        compare_images(
            np.zeros((20, 20, 3), dtype=np.uint8),
            np.zeros((21, 20, 3), dtype=np.uint8),
        )


def test_metrics_reject_invalid_input():
    with pytest.raises(ValueError):
        calculate_quality_metrics(
            np.zeros((20, 20), dtype=np.uint8)
        )


def test_denoising_rejects_invalid_parameters():
    image = sample_image()

    with pytest.raises(ValueError):
        denoise_image(image, diameter=0)

    with pytest.raises(ValueError):
        denoise_image(image, sigma_color=float("inf"))
